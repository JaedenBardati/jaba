import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np

from jabapy.apps import blr
from jabapy.snapshot import GIZMO_Snapshot
from jabapy.utils import constants as c
from jabapy.utils import units as u


class TestSnapshotDatasets(unittest.TestCase):
    def setUp(self):

        # create gas and stellar datasets with known physical units
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'snapshot.hdf5'
        with h5py.File(self.path, 'w') as file:
            file.create_group('Header').attrs.update(Time=1., ComovingIntegrationOn=0, HubbleParam=1.,
                UnitMass_In_CGS=1., UnitLength_In_CGS=1., UnitVelocity_In_CGS=1.,
                NumPart_Total=[3, 0, 0, 0, 3, 0])
            for part in (0, 4):
                group = file.create_group(f'PartType{part}')
                group['Coordinates'] = np.arange(1., 10.).reshape(3, 3)+part
                group['Velocities'] = np.ones((3, 3))
                group['Masses'] = [1., 2., 3.]
            gas = file['PartType0']
            gas['Density'] = np.array([1., 2., 3.])*1e-20
            gas['Temperature'] = [1e4, 2e4, 3e4]
            gas['Metallicity'] = [[.02, .28], [.01, .24], [.03, .25]]
            gas['ElectronAbundance'] = [1., .5, .25]
            gas['DustToGasRatio_Local'] = [.04, .02, .03]
            gas['Dust_Temperature'] = [500., 1500., 2000.]
            gas['KernelMaxRadius'] = [1., 2., 3.]
        self.snap = GIZMO_Snapshot(self.path)

    def test_aliases_share_raw_cache_and_preserve_lookup_formats(self):
        snap = self.snap
        self.assertEqual(snap.loaded_datasets, set())
        standard = snap.pos
        self.assertEqual(snap.loaded_datasets, set())
        positions = standard[0]
        for result in (snap.pos0, snap.pos['PartType0'], snap[0, 'Coordinates'], snap[0, 'pos']):
            self.assertIs(result, positions)
        self.assertEqual(snap.loaded_datasets, {('PartType0', 'Coordinates')})
        self.assertEqual(snap.derived_datasets, set())
        self.assertIn('Coordinates', snap[0])
        snap.particle_type_meanings = {'PartType0': 'gas', 'PartType4': 'stars'}
        self.assertEqual(snap['Coordinates'], [('PartType0', 'Coordinates'), ('PartType4', 'Coordinates')])
        self.assertIn(('PartType0', 'Coordinates'), snap.available_datasets)
        self.assertIs(snap.smooth0, snap[0, 'KernelMaxRadius'])

    def test_callable_is_lazy_and_cached_per_particle_type_and_snapshot(self):
        callback = Mock(side_effect=lambda snap, part: snap.mass[part]*2)
        with patch.dict(GIZMO_Snapshot._STANDARD_DSET_POSSIBLE_DATASET_NAMES, {'ne': callback}):
            standard = self.snap.ne
            standard._get_dataset_info_for_particle_type(0)
            callback.assert_not_called()
            first = self.snap[0, 'ne']
            self.assertIs(standard[0], first)
            self.assertIs(self.snap.ne0, first)
            self.assertIs(self.snap.ne['PartType0'], first)
            callback.assert_called_once_with(self.snap, 'PartType0')
            self.assertIn(('PartType0', 'ne'), self.snap.derived_datasets)
            self.assertNotIn(('PartType0', 'ne'), self.snap.loaded_datasets)
            self.assertNotIn(('PartType0', 'ne'), self.snap.available_datasets)
            _ = self.snap.ne4
            self.assertEqual(callback.call_count, 2)
            other = GIZMO_Snapshot(self.path)
            self.assertIsNot(other.ne0, first)
            self.assertEqual(callback.call_count, 3)

    def test_electron_density_and_particle_meanings(self):
        snap = self.snap
        snap.particle_type_meanings = {'PartType0': 'gas', 'PartType4': 'stars'}
        result = snap['gas', 'ne']
        expected = np.array([1., 2., 3.])*1e-20*np.array([.70, .75, .72])*np.array([1., .5, .25])/c.m_p.to_value(u.g)
        np.testing.assert_allclose(result.to_value(u.cm**-3), expected)
        self.assertIs(snap.ne['gas'], result)
        self.assertIs(snap.ne0, result)
        self.assertIn((0, 'ne'), snap)
        self.assertNotIn((4, 'ne'), snap)
        self.assertNotIn((4, 'dtg'), snap)
        with self.assertRaises(KeyError): _ = snap.ne1

    def test_alias_assignment_attaches_units_and_unloads_without_writing(self):
        snap = self.snap
        original = snap.pos0.copy()
        snap.pos[0] = np.ones((3, 3))
        np.testing.assert_allclose(snap[0, 'Coordinates'].to_value(u.cm), 1.)
        snap.pos0 = np.full((3, 3), 2.)*u.m
        np.testing.assert_allclose(snap.pos[0].to_value(u.cm), 200.)
        with self.assertRaises(ValueError): snap.pos0 = np.ones((3, 3))*u.s
        np.testing.assert_allclose(GIZMO_Snapshot(self.path).pos0.to_value(u.cm), original.to_value(u.cm))
        del snap.pos[0]
        self.assertNotIn(('PartType0', 'Coordinates'), snap.loaded_datasets)
        np.testing.assert_allclose(snap.pos0.to_value(u.cm), original.to_value(u.cm))
        del snap.pos0
        with self.assertRaises(AttributeError): del snap.pos0

    def test_assignment_before_loading_handles_dataset_units(self):
        snap = self.snap
        snap.mass0 = [4., 5., 6.]
        snap.temp0 = [100., 200., 300.]
        snap.fe0 = [.1, .2, .3]
        np.testing.assert_allclose(snap.mass0.to_value(u.g), [4., 5., 6.])
        np.testing.assert_allclose(snap.temp0.to_value(u.K), [100., 200., 300.])
        np.testing.assert_allclose(snap.fe0.to_value(u.dimensionless_unscaled), [.1, .2, .3])

    def test_derived_assignment_and_all_unload_styles(self):
        snap = self.snap
        original = snap.ne0.copy()
        for style in ('tuple', 'indexed', 'numbered'):
            with self.subTest(style=style):
                if style == 'tuple': snap[0, 'ne'] = np.ones(3)/u.m**3
                elif style == 'indexed': snap.ne[0] = np.ones(3)/u.m**3
                else: snap.ne0 = np.ones(3)/u.m**3
                np.testing.assert_allclose(snap.ne0.to_value(u.cm**-3), 1e-6)
                self.assertIn(('PartType0', 'ne'), snap.derived_datasets)
                if style == 'tuple': del snap[0, 'ne']
                elif style == 'indexed': del snap.ne[0]
                else: del snap.ne0
                self.assertNotIn(('PartType0', 'ne'), snap.derived_datasets)
                np.testing.assert_allclose(snap.ne0.value, original.value)
        snap.ne0 = np.full(3, 2.)
        np.testing.assert_allclose(snap.ne0.to_value(u.cm**-3), 2.)
        with self.assertRaises(ValueError): snap.ne0 = np.ones(3)*u.g

    def test_source_updates_and_unloads_invalidate_derived_cache(self):
        snap = self.snap
        original = snap.ne0.copy()
        snap[0, 'Density'] = snap.dens0*2
        self.assertEqual(snap.derived_datasets, set())
        np.testing.assert_allclose(snap.ne0.value, original.value*2)
        del snap[0, 'dens']
        self.assertEqual(snap.derived_datasets, set())
        np.testing.assert_allclose(snap.ne0.value, original.value)

    def test_transforms_recompute_functions_in_current_frame(self):
        callback = Mock(side_effect=lambda snap, part: np.linalg.norm(snap.pos[part], axis=1))
        with patch.dict(GIZMO_Snapshot._STANDARD_DSET_POSSIBLE_DATASET_NAMES, {'ne': callback}):
            original = self.snap.ne0.copy()
            self.snap.center_pos(np.ones(3)*u.cm)
            self.assertEqual(self.snap.derived_datasets, set())
            np.testing.assert_allclose(self.snap.ne0.value, np.linalg.norm(self.snap.pos0.value, axis=1))
            self.snap.reset_transform()
            self.assertEqual(self.snap.derived_datasets, set())
            np.testing.assert_allclose(self.snap.ne0.value, original.value)
            self.assertEqual(callback.call_count, 3)

    def test_property_deletion_and_unspecified_particle_assignment(self):
        snap = self.snap
        _ = snap.pos0, snap.pos4, snap.ne0
        del snap.pos
        self.assertNotIn(('PartType0', 'Coordinates'), snap.loaded_datasets)
        self.assertNotIn(('PartType4', 'Coordinates'), snap.loaded_datasets)
        self.assertEqual(snap.derived_datasets, set())
        _ = snap.ne0
        del snap.ne
        self.assertEqual(snap.derived_datasets, set())
        with self.assertRaises(NotImplementedError): snap.ne = np.ones(3)

    def test_failed_callable_is_not_cached(self):
        callback = Mock(side_effect=[RuntimeError('calculation failed'), np.ones(3)])
        with patch.dict(GIZMO_Snapshot._STANDARD_DSET_POSSIBLE_DATASET_NAMES, {'ne': callback}):
            with self.assertRaises(RuntimeError): _ = self.snap.ne0
            self.assertNotIn(('PartType0', 'ne'), self.snap.derived_datasets)
            np.testing.assert_array_equal(self.snap.ne0, np.ones(3))
            self.assertEqual(callback.call_count, 2)

    def test_missing_and_ambiguous_aliases(self):
        with self.assertRaises(AttributeError): _ = self.snap.acc0
        with h5py.File(self.path, 'a') as file:
            file['PartType0']['SmoothingLength'] = [1., 2., 3.]
        snap = GIZMO_Snapshot(self.path)
        with self.assertRaisesRegex(Exception, 'multiple possible datasets'): _ = snap.smooth0

    def test_load_standard_data_skips_unavailable_particle_datasets(self):
        self.snap.load_standard_data()
        self.assertIn(('PartType0', 'ne'), self.snap.derived_datasets)
        self.assertIn(('PartType0', 'dtg'), self.snap.derived_datasets)
        self.assertIn(('PartType4', 'Masses'), self.snap.loaded_datasets)

    def test_dust_formulas_preserve_user_implementation(self):
        np.testing.assert_allclose(self.snap.dtg0, .4*np.array([.04, .02, .03]))
        with h5py.File(self.path, 'a') as file:
            del file['PartType0']['DustToGasRatio_Local']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            snap = GIZMO_Snapshot(self.path)
            expected = .4*np.array([.02, .01, .03])/(1+np.exp((np.array([500., 1500., 2000.])-1500)/75))
            np.testing.assert_allclose(snap.dtg0, expected)
            with h5py.File(self.path, 'a') as file:
                del file['PartType0']['Dust_Temperature']
            np.testing.assert_allclose(GIZMO_Snapshot(self.path).dtg0, .4*np.array([.02, .01, .03]))
            with h5py.File(self.path, 'a') as file:
                del file['PartType0']['Metallicity']
            np.testing.assert_allclose(GIZMO_Snapshot(self.path).dtg0, .02*np.ones(3))

    def test_blr_uses_lazy_electron_density_and_dust(self):
        snap = self.snap
        gas = blr.get_Halpha_alpha_emissivity(snap, dust=False)
        self.assertIn(('PartType0', 'ne'), snap.derived_datasets)
        self.assertNotIn(('PartType0', 'dtg'), snap.derived_datasets)
        dusty = blr.get_Halpha_alpha_emissivity(snap)
        self.assertIn(('PartType0', 'dtg'), snap.derived_datasets)
        k_abs, k_sca = blr.rad.kappa_dust_MRN_cgs(blr.rad.transition_frequency_cgs['HI 2-3'], dust_to_gas=1.)
        rho_dust = snap.dens0.to_value(u.g/u.cm**3)*(.4*np.array([.04, .02, .03]))
        np.testing.assert_allclose(dusty[0].value, gas[0].value)
        np.testing.assert_allclose((dusty[1]-gas[1]).to_value(u.cm**-1), rho_dust*k_abs)
        np.testing.assert_allclose((dusty[2]-gas[2]).to_value(u.cm**-1), rho_dust*k_sca)

    def test_blr_preserves_transition_and_explicit_coefficients(self):
        snap = self.snap
        with patch.object(blr.rad, 'raytrace_line_profile', return_value={}) as trace:
            blr.get_Halpha_line_profile(snap, np.zeros((1, 3))*u.cm, np.ones((1, 3))*u.cm, 1*u.cm**2,
                emissivity=np.zeros(3)*u.erg/u.s/u.cm**3, alpha_abs=np.zeros(3)/u.cm,
                alpha_eff=np.zeros(3)/u.cm, transition='HI 2-4', thermal_broadening=False, doppler_shift=False)
            self.assertEqual(snap.derived_datasets, set())
            np.testing.assert_allclose(trace.call_args.args[5].to_value(u.AA), 4861.)
            self.assertIs(trace.call_args.kwargs['pos'], snap.pos0)


if __name__ == '__main__':
    unittest.main()
