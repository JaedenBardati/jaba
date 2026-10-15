import io
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from jabapy.utils import visual as jv


class TestMultipanelFigure(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')

    def test_panel_helpers_defaults_and_overlays(self):

        # exercise the primary indexing api with shared and per-panel settings
        fig = jv.Figure(2, 2, figsize=(8, 6), xlabel='shared x', show_legend_on_ax_num=False)
        self.assertEqual(len(fig.fig.axes), 0)
        line = fig[0, 0].plot([0, 1], [1, 2], title='lines', color='red')[0]
        fig[0, 0].plot([0, 1], [2, 3], label='overlay')
        points = fig[0, 1].scatter([0, 1], [2, 1], xlabel='scatter x')
        counts = fig[1, 0].hist([0, 0, 1], bins=2)[0]
        image = fig[1, 1].imshow(np.ones((2, 3)), show_colorbar=False)
        self.assertEqual(line.get_color(), 'red')
        self.assertEqual(len(fig[0, 0].lines), 2)
        self.assertEqual(fig[0, 0].get_title(), 'lines')
        self.assertEqual(fig[0, 0].get_xlabel(), 'shared x')
        self.assertEqual(fig[0, 1].get_xlabel(), 'scatter x')
        self.assertIs(points.axes, fig[0, 1].ax)
        self.assertEqual(counts.sum(), 3)
        self.assertIs(image.axes, fig[1, 1].ax)
        np.testing.assert_allclose(fig.fig.get_size_inches(), [8, 6])

    def test_automatic_reading_order_and_explicit_overlay(self):

        # default calls advance only when a panel contains data
        fig = jv.Figure(2, 2)
        fig[0, 0].set_title('reserved title')
        for index in range(4):
            line = fig.plot([0, 1], [index, index+1])[0]
            self.assertIs(line.axes, fig[index].ax)
        with self.assertRaisesRegex(ValueError, 'All panels'): fig.plot([0, 1], [0, 1])
        fig[0, 0].plot([0, 1], [2, 3])
        self.assertEqual(len(fig[0, 0].lines), 2)

    def test_slices_spans_and_indices(self):

        # spanning axes occupy several cells but are plotted only once by a selection
        fig = jv.Figure(grid=(2, 3))
        span = fig.add_axis(0, slice(None))
        self.assertIs(fig[0, 2].ax, span)
        self.assertIs(fig.add_axis(0, slice(None)), span)
        results = fig[:, :].plot([0, 1], [0, 1])
        self.assertEqual(len(results), 4)
        self.assertEqual(len(span.lines), 1)
        self.assertEqual(fig.get_axis_index(slice(None), 1), [1, 4])
        self.assertIs(fig[-1].ax, fig[1, -1].ax)
        self.assertEqual(len(fig[1, ::2].axes), 2)
        with self.assertRaisesRegex(ValueError, 'overlaps'): fig.add_axis(slice(None), 0)
        with self.assertRaises(IndexError): fig[2, 0]
        with self.assertRaises(ValueError): fig[0:0, :]
        with self.assertRaises(ValueError): fig[:, :].ax
        with self.assertRaises(ValueError): jv.Figure(1, 3).add_axis(0, slice(None, None, 2))

    def test_adopt_existing_figure(self):

        # recover common grids and preserve existing plots and empty cells
        original, axes = plt.subplots(2, 2, figsize=(7, 5))
        axes[1, 0].plot([0, 1], [1, 0])
        fig = jv.Figure(fig=original)
        self.assertEqual((fig.nrows, fig.ncols), (2, 2))
        self.assertIs(fig[1, 0].ax, axes[1, 0])
        fig[0, 1].scatter([0], [1])
        np.testing.assert_allclose(original.get_size_inches(), [7, 5])
        with self.assertRaises(ValueError): jv.Figure(fig=original, figsize=(3, 3))
        other = jv.Figure()
        with self.assertRaises(ValueError): fig.plot([0], [1], ax=other[0])

    def test_draw_save_clear_and_close_target_figure(self):

        # export an actual multipanel image while another figure is current
        fig = jv.Figure(1, 2, figsize=(6, 3), savefig_dpi=40, text_usetex=False)
        fig[0].plot([0, 1], [0, 1], title='line', label='line')
        fig[1].imshow(np.ones((3, 2)), show_colorbar=False, aspect='equal', title='image')
        other, other_ax = plt.subplots()
        other_ax.plot([0, 1], [1, 0])
        output = io.BytesIO()
        fig.draw(out=output, show_legend=True)
        output.seek(0)
        pixels = plt.imread(output, format='png')
        self.assertEqual(pixels.shape[1], 240)
        self.assertEqual(fig[0].get_title(), 'line')
        self.assertEqual(fig[1].get_aspect(), 1.)
        self.assertIsNotNone(fig[0].get_legend())
        self.assertEqual(len(other_ax.lines), 1)
        with patch.object(plt, 'show') as show: fig.draw(show=True)
        show.assert_called_once_with()
        stale = fig[0]
        fig.draw(clf_after=True, close_after=True)
        self.assertFalse(plt.fignum_exists(fig.fig.number))
        self.assertTrue(plt.fignum_exists(other.number))
        self.assertEqual(len(other_ax.lines), 1)
        self.assertEqual(fig.axes, [None, None])
        with self.assertRaisesRegex(ValueError, 'cleared'): stale.plot([0], [1])

    def test_colorbar_draw_and_selection_display(self):

        # finalizing a figure must not duplicate colorbars or show once per selected panel
        fig = jv.Figure(1, 2)
        fig[0].imshow(np.ones((2, 2)))
        count = len(fig.fig.axes)
        fig.draw()
        fig.draw(xlabel='x')
        self.assertEqual(len(fig.fig.axes), count)
        with patch.object(plt, 'show') as show:
            fig[:, :].plot([0, 1], [0, 1], show=True)
        show.assert_called_once_with()
        self.assertEqual(len(fig[0].images), 1)

    def test_validation_and_clear_reuse(self):

        # invalid construction and selection fail without silently replacing axes
        for kwargs in ({'nrows': 0}, {'ncols': 1.5}, {'unknown': True}, {'grid': (1, 2), 'nrows': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): jv.Figure(**kwargs)
        fig = jv.Figure(1, 2)
        fig[0].plot([0, 1], [0, 1])
        fig.clear()
        self.assertEqual(len(fig.fig.axes), 0)
        fig.plot([1, 2], [1, 2])
        self.assertEqual(len(fig[0].lines), 1)
        with self.assertRaises(ValueError): fig.draw(unknown=True)

    def test_inline_display_retains_figure_manager(self):

        # notebook display must leave the manager available for subsequent draws
        fig = jv.Figure()
        fig[0, 0].plot([0, 1], [0, 1])
        with patch.object(matplotlib, 'get_backend', return_value='module://matplotlib_inline.backend_inline'), patch.object(plt, 'show') as show:
            fig.draw(show=True)
            fig.draw(show=True)
        self.assertEqual(show.call_count, 2)
        show.assert_called_with(close=False)
        self.assertTrue(plt.fignum_exists(fig.fig.number))
