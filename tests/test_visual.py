import io
import unittest
import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from jabapy.utils import visual as jv


class TestFixedWidth(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')

    def test_exported_width_for_lines_and_fixed_aspect_images(self):
        for dpi in (40, 100):
            for kind in ('line', 'square', 'wide', 'tall', 'square_without_layout'):
                with self.subTest(dpi=dpi, kind=kind), plt.rc_context({'savefig.dpi': dpi, 'savefig.pad_inches': .05, 'text.usetex': False}):
                    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='none' if kind == 'square_without_layout' else 'constrained')
                    if kind == 'line': ax.plot([0., 1.], [0., 1.])
                    else:
                        extent = [-10., 10., -5., 5.] if kind == 'wide' else [-5., 5., -10., 10.] if kind == 'tall' else [-5., 5., -5., 5.]
                        ax.imshow(np.arange(6).reshape(2, 3), extent=extent, aspect='equal')
                    ax.set(xlabel='x [kpc]', ylabel='y [kpc]', title=kind)
                    fontsize, aspect = ax.title.get_fontsize(), ax.get_aspect()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        jv.iterate_height_to_get_fixed_width(fig, fig_aspect=None)
                    self.assertFalse(caught, [str(w.message) for w in caught])
                    self.assertEqual(ax.title.get_fontsize(), fontsize)
                    self.assertEqual(ax.get_aspect(), aspect)
                    if kind in ('square', 'tall'): self.assertGreater(fig.get_figheight(), 4.8)
                    output = io.BytesIO()
                    fig.savefig(output, format='png', bbox_inches='tight')
                    output.seek(0)
                    pixels = plt.imread(output, format='png')
                    self.assertEqual(pixels.shape[1], round(6.4*dpi))
                    plt.close(fig)

    def test_iteration_limit_warns_and_keeps_the_best_measured_size(self):
        with plt.rc_context({'savefig.dpi': 40, 'text.usetex': False}):
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.imshow(np.ones((2, 2)))
            original = fig.get_size_inches().copy()
            with self.assertWarnsRegex(UserWarning, 'after 1 iterations.*target 256 pixels'):
                jv.iterate_height_to_get_fixed_width(fig, max_iter=1, fig_aspect=None)
            np.testing.assert_array_equal(fig.get_size_inches(), original)
            with patch.object(fig, 'get_tightbbox', return_value=Bbox.from_bounds(0, 0, 2, 2)) as bbox:
                with self.assertWarnsRegex(UserWarning, 'after 3 iterations.*84.000 pixels, target 256 pixels'):
                    jv.iterate_height_to_get_fixed_width(fig)
                self.assertEqual(bbox.call_count, 3)
            np.testing.assert_array_equal(fig.get_size_inches(), original)

    def test_figure_dpi_setting(self):
        with plt.rc_context({'savefig.dpi': 'figure', 'text.usetex': False}):
            fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=72, layout='constrained')
            ax.imshow(np.ones((2, 2)))
            jv.iterate_height_to_get_fixed_width(fig)
            output = io.BytesIO(); fig.savefig(output, format='png', bbox_inches='tight'); output.seek(0)
            self.assertEqual(plt.imread(output, format='png').shape[1], round(6.4*72))
