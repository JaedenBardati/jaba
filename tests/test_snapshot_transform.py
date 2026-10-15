import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from jabapy.snapshot import GIZMO_Snapshot
from jabapy.utils import units as u


class TestSnapshotTransform(unittest.TestCase):
    def setUp(self):

        # create a small snapshot with distinct particle positions and velocities
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'snapshot.hdf5'
        with h5py.File(path, 'w') as file:
            file.create_group('Header').attrs.update(Time=1., ComovingIntegrationOn=0, HubbleParam=1.,
                UnitMass_In_CGS=u.Msun.to(u.g), UnitLength_In_CGS=u.kpc.to(u.cm),
                UnitVelocity_In_CGS=(u.km/u.s).to(u.cm/u.s))
            for part in (0, 3, 4):
                group = file.create_group(f'PartType{part}')
                group['Coordinates'] = np.array([[10., 20., 30.], [12., 25., 37.]])+part
                group['Velocities'] = np.array([[2., 4., 6.], [3., 7., 9.]])+part
        self.path = path

    def test_repeated_center_and_reset_with_loaded_and_lazy_data(self):

        # compare each centering cycle to the original snapshot independently of load order
        for preload in (False, True):
            with self.subTest(preload=preload):
                snap = GIZMO_Snapshot(self.path)
                original = GIZMO_Snapshot(self.path)
                positions = {part: original.pos[part].copy() for part in (0, 3, 4)}
                velocities = {part: original.vel[part].copy() for part in (0, 3, 4)}
                if preload:
                    for part in (0, 3, 4):
                        _ = snap.pos[part], snap.vel[part]
                for cycle in range(5):
                    snap.reset_transform()
                    for part in (0, 3, 4) if cycle else ():
                        np.testing.assert_allclose(snap.pos[part].to_value(u.kpc), positions[part].to_value(u.kpc), atol=1e-12)
                        np.testing.assert_allclose(snap.vel[part].to_value(u.km/u.s), velocities[part].to_value(u.km/u.s), atol=1e-12)
                    snap.center_on(3)
                    for part in (0, 3, 4):
                        np.testing.assert_allclose(snap.pos[part].to_value(u.kpc), (positions[part]-positions[3][0]).to_value(u.kpc), atol=1e-12)
                        np.testing.assert_allclose(snap.vel[part].to_value(u.km/u.s), (velocities[part]-velocities[3][0]).to_value(u.km/u.s), atol=1e-12)

    def test_center_after_rotation_matches_lazy_loading_and_resets(self):

        # loaded datasets need centers in the current frame while lazy loading uses the original frame
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for preload in (False, True):
            with self.subTest(preload=preload):
                snap = GIZMO_Snapshot(self.path)
                original = GIZMO_Snapshot(self.path)
                if preload:
                    for part in (0, 3, 4):
                        _ = snap.pos[part], snap.vel[part]
                snap.rotate_z(90)
                snap.center_on(3)
                for part in (0, 3, 4):
                    expected_pos = (original.pos[part]-original.pos[3][0]).to_value(u.kpc) @ rotation.T
                    expected_vel = (original.vel[part]-original.vel[3][0]).to_value(u.km/u.s) @ rotation.T
                    np.testing.assert_allclose(snap.pos[part].to_value(u.kpc), expected_pos, atol=1e-12)
                    np.testing.assert_allclose(snap.vel[part].to_value(u.km/u.s), expected_vel, atol=1e-12)
                snap.reset_transform()
                for part in (0, 3, 4):
                    np.testing.assert_allclose(snap.pos[part].to_value(u.kpc), original.pos[part].to_value(u.kpc), atol=1e-12)
                    np.testing.assert_allclose(snap.vel[part].to_value(u.km/u.s), original.vel[part].to_value(u.km/u.s), atol=1e-12)
