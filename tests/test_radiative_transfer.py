import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from jabapy.physics import radiation as rad
from jabapy.apps import blr
from jabapy.utils import grid, units as u, constants as c


class TestParticleTransfer(unittest.TestCase):
    def setUp(self):
        self.pos = np.stack(np.meshgrid(*([np.linspace(-2, 2, 7)]*3), indexing='ij'), axis=-1).reshape(-1, 3)
        self.tree = cKDTree(self.pos)
        self.x0 = np.array([[0., 0., -1.], [0.2, 0.3, -1.]])
        self.x1 = self.x0.copy(); self.x1[:, 2] = 1
        self.kw = dict(tree=self.tree, N_NGB=8, workers=1, query_chunk_size=1, step_fraction=0.25)
        self.ones = np.ones(len(self.pos))

    def test_slab_and_zero_opacity(self):
        for reverse in (False, True):
            for alpha in (0., 0.3):
                I, tau = rad.raytrace_particles(self.x0, self.x1, 2*self.ones, alpha*self.ones,
                                                I0=0.7, reverse=reverse, ret_tau=True, **self.kw)
                expected = 0.7+4 if alpha == 0 else 0.7*np.exp(-0.6)+2/0.3*(-np.expm1(-0.6))
                np.testing.assert_allclose(I, expected)
                np.testing.assert_allclose(tau, 2*alpha)

    def test_particle_contributions(self):
        alpha = np.column_stack((0.3*self.ones, 0.6*self.ones))
        I, tau, parts = rad.raytrace_particles(self.x0, self.x1, 2*self.ones, alpha, I0=0.7,
                                              ret_tau=True, ret_particle_contributions=True, pixel_area=[2., 3.], **self.kw)
        np.testing.assert_allclose(parts.sum(axis=0), ((I-0.7*np.exp(-tau))*np.array([2., 3.])[:, None]).sum(axis=0))

    def test_transformed_path(self):
        transform = lambda xyz: xyz*np.array([1., 1., 2.])
        I, tau = rad.raytrace_particles(self.x0, self.x1, 2*self.ones, 0.3*self.ones,
                                        grid_to_particle=transform, ret_tau=True, **self.kw)
        np.testing.assert_allclose(tau, 1.2)
        np.testing.assert_allclose(I, 2/0.3*(-np.expm1(-1.2)))

    def test_factored_opacity_and_shared_spectral_queries(self):
        rng = np.random.default_rng(12)
        j = rng.random((len(self.pos), 3)); density = rng.random(len(self.pos))
        factors = np.array([[0., .2, .4], [0., .4, .6], [0., .8, 1.]])
        repeated_j = np.repeat(j[..., None], 3, axis=-1).reshape(len(j), -1)
        repeated_alpha = (density[:, None, None]*factors).reshape(len(j), -1)
        for reverse in (False, True):
            expected = rad.raytrace_particles(self.x0, self.x1, repeated_j, repeated_alpha,
                I0=.7, reverse=reverse, ret_tau=True, **self.kw)
            for chunk in (1, 3):
                with patch.object(grid, 'mfm_value_at_pos', wraps=grid.mfm_value_at_pos) as query:
                    actual = rad.raytrace_particles(self.x0, self.x1, j, density, opacity_factors=factors,
                        frequency_chunk_size=chunk, I0=.7, reverse=reverse, ret_tau=True, **self.kw)
                    if chunk == 1: queries = query.call_count
                    else: self.assertEqual(query.call_count, queries)
                for value, reference in zip(actual, expected):
                    np.testing.assert_allclose(value.reshape(2, -1), reference, rtol=1e-13)
        I, tau, parts = rad.raytrace_particles(self.x0, self.x1, j, density, opacity_factors=factors,
            frequency_chunk_size=2, I0=.7, ret_tau=True, ret_particle_contributions=True, pixel_area=[2., 3.], **self.kw)
        np.testing.assert_allclose(parts.sum(axis=0), ((I-.7*np.exp(-tau))*np.array([2., 3.])[:, None, None]).sum(axis=0))

    def test_cartesian_queries_and_transfer_match_tree(self):
        rng = np.random.default_rng(42)
        for axes in [(np.arange(4.),), (np.arange(3.), np.linspace(-10, 2, 7)),
                     (np.arange(3.), np.linspace(-10, 2, 5), np.array([-2., .1, .2, .4, 3.]))]:
            points = rng.normal(size=(50, len(axes)))*5
            coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(axes))
            cartesian, tree = grid.CartesianGrid(axes), cKDTree(coords)
            for k in (1, min(8, len(coords)), len(coords)):
                distances, indices = cartesian.query(points, k=k)
                np.testing.assert_allclose(distances, tree.query(points, k=k)[0], rtol=1e-14)
                delta = coords[indices]-(points[:, None] if k > 1 else points)
                np.testing.assert_allclose(distances, np.linalg.norm(delta, axis=-1))
        kw = dict(self.kw, tree=grid.CartesianGrid([np.linspace(-2, 2, 7)]*3))
        j, alpha = rng.random((len(self.pos), 3)), rng.random((len(self.pos), 3))
        expected = rad.raytrace_particles(self.x0, self.x1, j, alpha, ret_tau=True, reverse=True, **self.kw)
        actual = rad.raytrace_particles(self.x0, self.x1, j, alpha, ret_tau=True, reverse=True, **kw)
        for value, reference in zip(actual, expected): np.testing.assert_allclose(value, reference, rtol=1e-13)

    def test_generic_line_and_dispersion(self):
        kwargs = dict(emissivity=2*self.ones, alpha=0*self.ones, wavelength0=5007*u.AA,
                      doppler_shift=False, thermal_broadening=False, case_names=['line'], n_spectral=101,
                      spectral_width=1000, **self.kw)
        delta = rad.raytrace_line_profile(self.x0, self.x1, 1*u.cm**2, **kwargs)['profiles']['line']
        broad = rad.raytrace_line_profile(self.x0, self.x1, 1*u.cm**2, velocity_dispersion=30*u.km/u.s, **kwargs)['profiles']['line']
        np.testing.assert_allclose(delta['line_luminosity'].value, 8)
        np.testing.assert_allclose(delta['line_luminosity'], broad['line_luminosity'])
        np.testing.assert_allclose(broad['window_fraction'], 1)
        self.assertGreater(np.count_nonzero(broad['dL_dv'].value), np.count_nonzero(delta['dL_dv'].value))
        np.testing.assert_allclose(delta['intensity'].value, 4/(4*np.pi))

    def test_halpha_partial_defaults_and_explicit_zeros(self):
        snap = SimpleNamespace(pos0=self.pos*u.cm)
        derived = (2*self.ones*u.erg/u.s/u.cm**3, .3*self.ones/u.cm, .1*self.ones/u.cm, .4*self.ones/u.cm)
        explicit = (np.zeros_like(self.ones)*u.erg/u.s/u.cm**3, np.zeros_like(self.ones)/u.cm, np.zeros_like(self.ones)/u.cm)
        for mask in range(8):
            values = [explicit[i] if mask & (1 << i) else None for i in range(3)]
            with patch.object(blr, 'get_Halpha_alpha_emissivity', return_value=derived) as coefficients, \
                 patch.object(blr.rad, 'raytrace_line_profile', return_value={}) as trace:
                blr.get_Halpha_line_profile(snap, self.x0, self.x1, 1*u.cm**2, *values,
                    thermal_broadening=False, doppler_shift=False, source='test', g_ff=1.2, d=2., dust=False)
                if mask == 7: coefficients.assert_not_called()
                else: coefficients.assert_called_once_with(snap, transition='HI 2-3', source='test', g_ff=1.2, d=2., dust=False)
                args = trace.call_args.args
                expected = [explicit[i] if values[i] is not None else derived[j] for i, j in enumerate((0, 1, 3))]
                np.testing.assert_allclose(args[3], expected[0])
                np.testing.assert_allclose(args[4][:, 0], 0/u.cm)
                np.testing.assert_allclose(args[4][:, 1], expected[1])
                np.testing.assert_allclose(args[4][:, 2], expected[2])

    def test_halpha_dust_opacities_use_derived_ratio(self):
        class Snapshot(dict):
            pass
        snap = Snapshot({(0, 'Metallicity'): np.array([[.02, .28], [.01, .28], [.03, .28], [0., .28]]),
                         (0, 'ElectronAbundance'): np.ones(4)})
        snap.dens0 = np.array([1., 2., 3., 4.])*1e-20*u.g/u.cm**3
        snap.temp0 = np.full(4, 1e4)*u.K
        snap[0, 'ne'] = (snap.dens0/c.m_p).to(u.cm**-3)*(1-snap[0, 'Metallicity'].sum(axis=1))*snap[0, 'ElectronAbundance']
        gas = blr.get_Halpha_alpha_emissivity(snap, dust=False)
        for local_ratio in (None, np.array([.04, .02, .03, 0.])):
            if local_ratio is not None: snap[0, 'DustToGasRatio_Local'] = local_ratio.copy()
            for temperatures in (None, np.array([0., 1500., 1e8, 500.])*u.K):
                if temperatures is not None: snap[0, 'Dust_Temperature'] = temperatures
                elif (0, 'Dust_Temperature') in snap: del snap[0, 'Dust_Temperature']

                # supply a derived ratio; snapshot tests cover the actual dust prescription
                ratio = 0.5*snap[0, 'Metallicity'][:, 0] if local_ratio is None else local_ratio
                survival = 1. if temperatures is None else np.array([1/(1+np.exp(-20)), .5, 0., 1/(1+np.exp(-1000/75))])
                snap[0, 'dtg'] = np.clip(ratio*survival, 0, 1)
                result = blr.get_Halpha_alpha_emissivity(snap)
                density = snap.dens0.to_value(u.g/u.cm**3)*snap[0, 'dtg']
                k_abs, k_sca = rad.kappa_dust_MRN_cgs(rad.transition_frequency_cgs['HI 2-3'], dust_to_gas=1.)
                np.testing.assert_array_equal(result[0], gas[0])
                np.testing.assert_allclose(result[1], gas[1]+density*k_abs/u.cm, rtol=1e-13, atol=0)
                np.testing.assert_allclose(result[2], gas[2]+density*k_sca/u.cm, rtol=1e-13, atol=0)
                effective = rad.effective_extinction_optical_depth(result[1].value, result[2].value, d=3.)/u.cm
                np.testing.assert_allclose(result[3], effective, rtol=1e-13, atol=0)
                if local_ratio is not None: np.testing.assert_array_equal(snap[0, 'DustToGasRatio_Local'], local_ratio)
        with patch.object(blr.rad, 'kappa_dust_MRN_cgs', side_effect=AssertionError('Dust evaluated with dust=False')):
            again = blr.get_Halpha_alpha_emissivity(snap, dust=False)
        for value, expected in zip(again, gas): np.testing.assert_array_equal(value, expected)

    def test_halpha_adapter_with_and_without_coefficients(self):
        snap = SimpleNamespace(pos0=self.pos*u.cm, vel0=np.zeros_like(self.pos)*u.km/u.s, temp0=self.ones*1e4*u.K)
        args = (snap, self.x0, self.x1, 1*u.cm**2)
        coefficients = (2*self.ones*u.erg/u.s/u.cm**3, .3*self.ones/u.cm, .1*self.ones/u.cm, .4*self.ones/u.cm)
        with patch.object(blr, 'get_Halpha_alpha_emissivity', return_value=coefficients) as derived:
            result = blr.get_Halpha_line_profile(*args, n_spectral=30, **self.kw)
            derived.assert_called_once_with(snap, transition='HI 2-3', source='Pequignot_etal_1991', g_ff=1.0, d=3.0, dust=True)
            explicit = blr.get_Halpha_line_profile(*args, coefficients[0], coefficients[1], coefficients[3], n_spectral=30, **self.kw)
            self.assertEqual(derived.call_count, 1)
        for name in result['profiles']:
            np.testing.assert_allclose(result['profiles'][name]['dL_dv'], explicit['profiles'][name]['dL_dv'])


if __name__ == '__main__':
    unittest.main()
