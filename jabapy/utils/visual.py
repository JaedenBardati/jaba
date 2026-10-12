import io, functools, warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.ticker import AutoMinorLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

###############################################
############## FIGURE SETTINGS  ###############
###############################################
LETTER_PAPER_SIZE          = (8.5, 11)       # width x height (inches)
A4_PAPER_SIZE              = (8.27, 11.69)
MATPLOTLIB_DEFAULT_FIGSIZE = (6.4, 4.8)

## Custom paper settings for different journals
CUSTOM_PAPER_SETTINGS = dict()

# ApJ twocolumn preprint
CUSTOM_PAPER_SETTINGS['APJ_TWOCOLUMN'] = {
    'font_size': 8.5,  # ApJ preprint font size is ~9, but i prefer a bit smaller for figures
    'subtypes': ['doublecolumn', 'singlecolumn'],
    'doublecolumn_size': (LETTER_PAPER_SIZE[0] - 2*18/25.4, LETTER_PAPER_SIZE[1] - 2*25/25.4),
    'singlecolumn_size': ((LETTER_PAPER_SIZE[0] - 2*18/25.4 - 10/25.4)/2.0, LETTER_PAPER_SIZE[1] - 2*25/25.4), 
    'default_subtype': 'doublecolumn',
}
CUSTOM_PAPER_SETTINGS['ApJ'] = CUSTOM_PAPER_SETTINGS['APJ_TWOCOLUMN'] # alias

# ... add more paper types here as needed ...


def construct_paper_figsize(paper, subtype=None, aspect=None):
    """Construct a figure size (width, height) in inches based on the paper type, aspect ratio, and column type."""
    if paper not in CUSTOM_PAPER_SETTINGS:
        raise ValueError('Invalid paper type: {}'.format(paper))
    if aspect is None:
        aspect = 'equal'
    if subtype is None:
        subtype = CUSTOM_PAPER_SETTINGS[paper]['default_subtype']

    if isinstance(subtype, int):
        if 0 <= subtype < len(CUSTOM_PAPER_SETTINGS[paper]['subtypes']):
            subtype = CUSTOM_PAPER_SETTINGS[paper]['subtypes'][subtype]
        else:
            raise ValueError('Invalid subtype index: {} (max: {})'.format(subtype, len(CUSTOM_PAPER_SETTINGS[paper]['subtypes']) - 1))
        
    elif not isinstance(subtype, str) and subtype in CUSTOM_PAPER_SETTINGS:
        raise ValueError('Invalid subtype: {}'.format(subtype))

    size = CUSTOM_PAPER_SETTINGS[paper].get(subtype + '_size', None)
    if size is None:
        raise ValueError('Invalid subtype: {}'.format(subtype))
    width, height = size

    if aspect == 'auto':
        # if auto, assume full possible height and will be cut later to fit (via e.g. bbox_inches)
        return (width, height)
    elif aspect == 'equal':
        aspect = 1.0
    elif not isinstance(aspect, (int, float)):
        raise ValueError('Invalid aspect: {}'.format(aspect))
    
    _height = aspect * width
    height = min(_height, height)
    if _height > height:
        warnings.warn("The specified aspect ratio results in a height that exceeds the maximum height for the paper type you used, you will likely have to adjust the aspect or use a smaller figure size.")
    
    return (width, height)

def set_style_defaults(settings={}, style=None, substyle=None, fig_aspect=None):
    """Set the default figure size and font size."""
    # settings are rc params only currently
    if style not in CUSTOM_PAPER_SETTINGS and style is not None:
        raise ValueError('Invalid paper type: {}'.format(style))
    
    if style is None:
        # set to default mpl settings
        figsize = MATPLOTLIB_DEFAULT_FIGSIZE
        fontsize = 10
    elif style in CUSTOM_PAPER_SETTINGS:
        figsize = construct_paper_figsize(style, subtype=substyle, aspect=fig_aspect)
        fontsize = CUSTOM_PAPER_SETTINGS[style]['font_size']
    else:
        raise ValueError('Invalid jv style: {}'.format(style))

    settings['figure.figsize'] = figsize
    settings['font.size'] = fontsize
    settings['xtick.labelsize'] = fontsize
    settings['ytick.labelsize'] = fontsize
    settings['legend.fontsize'] = fontsize
    return settings

###############################################
############ DEFAULT RC PARAMETERS ############
###############################################

## Actually adjust parameters
matplotlib.rcParams['figure.dpi'] = 100   # plt.show resolution
matplotlib.rcParams['savefig.dpi'] = 300  # plt.savefig resolution
if 'figure.constrained_layout.use' in matplotlib.rcParams:
    matplotlib.rcParams['figure.constrained_layout.use'] = True
else:
    matplotlib.rcParams['figure.autolayout'] = True  # legacy alternative to constrained_layout (use if above does not work)
    warnings.warn('Your version of matplotlib does not have constrained_layout, using autolayout instead. Consider updating matplotlib to a newer version for better layout control.')
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams['savefig.pad_inches'] = 0.05  # mpl default is 0.1

matplotlib.rcParams['axes.linewidth'] = 1.25
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 3
matplotlib.rcParams['ytick.minor.size'] = 3
matplotlib.rcParams['xtick.major.width'] = 1.25
matplotlib.rcParams['ytick.major.width'] = 1.25
matplotlib.rcParams['xtick.minor.width'] = 1.25
matplotlib.rcParams['ytick.minor.width'] = 1.25
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['ytick.right'] = True

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams["legend.borderpad"] = 0.8   # default is normally 0.4

###############################################
############ BASIC FIGURE WRAPPER #############
###############################################

# any of the below can be passed as settings to functions with basic figure wrapper
DEFAULT_CUSTOM_SUBPLOT_SETTINGS = {
    # meta settings (settings that override other settings)
    'fig_style': 'ApJ',                          # if set, will override figsize and fontsize settings with that style (e.g., paper defaults)
    'fig_substyle': None,                        # if set, will override figure substyle (singlecol/doublecol) with this value using paper default width
    'fig_aspect': None,                          # if set, will override figure aspect ratio (width/height) with this value using paper default width 

    'no_ticks': False,                           # if true, will set all tick sizes to 0 (overrides any other tick size settings)
    'no_minor_ticks': False,                     # if true, will set all minor tick sizes to 0 (overrides any other minor tick size settings)
    'no_tick_labels': False,                     # if true, will set all tick label sizes to 0 (overrides any other tick label size settings)
    'force_minor_ticks': False,                  # if true, will try to force minor ticks on x and y axes
    'force_regular_log_major_ticks': False,      # if true, will force regular log major ticks on x and y axes (i.e. 1, 10, 100, etc. not 1, 100, etc.)
    'default_ticks_on_colorbar': True,           # if true, will force default ticks on colorbar
    
    'loglog': None,                              # if set, will set x and y axes to log or linear scale 
    'xlog': None,
    'ylog': None,
    'xminor_nsubs': None,         # if set, will use AutoMinorLocator with this number of subdivisions for the x-axis minor ticks (overrides any other x-axis minor tick locator settings)
    'yminor_nsubs': None,
    'xmin': None,                 # if set, will set the x-axis lower limit (overrides any other xlim settings)
    'xmax': None,                 # if set, will set the x-axis upper limit (overrides any other xlim settings)
    'ymin': None,                 # if set, will set the y-axis lower limit (overrides any other ylim settings)
    'ymax': None,                 # if set, will set the y-axis upper limit (overrides any other ylim settings)

    # add optional components
    'show_legend': False,
    'show_colorbar': False,

    # legend settings
    'show_legend_on_ax_num': 0,   # if set, will show legend on the axes with this index (0-indexed) in the figure, if -1 it will only show if the plot is done, if True it will show on every axes, if False it will not show on any axes (unless overridden by show_legend)
    
    # tick subdivision settings
    'xminor_locator': None,     # if set, will use this locator for the x-axis minor ticks
    'yminor_locator': None,
    'xmajor_locator': None,     # if set, will use this locator for the x-axis major ticks
    'ymajor_locator': None,

    # > colorbar
    'colorbar_label': None,
    'colorbar_on_fig': False,    # True -> colorbar on figure, False -> colorbar on subplots
    'colorbar_rcparams': {},     # things set here will override other rc parameter settings for the colorbar only
    
    # closing settings
    'clf_after': None,           # if None: clf figure only if done with plotting (i.e. out or show is set)
    'close_after': None,         # if None: plt.closes figure only if done with plotting (i.e. out or show is set)
    'out': None,                 # if set: triggers plot end and savefig at provided path
    'show': False,               # if True: triggers plot end and shows to screen

    # other settings
    'clf_before': False,           # if True: clear figure before plotting
    'fixed_width': True,           # if True: iteratively adjust figure size to achieve a fixed width with tight bbox, but variable height/figure aspect ratio
    'fixed_width_max_iter': 100,   # max iterations to adjust figure size to achieve fixed width
}
DEFAULT_SET_SETTINGS = {
    # axes.set settings
    'alpha': None,
    'aspect': 'auto', # aspect of the axes, not the figure (which is set via figsize)
    'xscale': 'linear',
    'yscale': 'linear',
    'xlabel': None,
    'ylabel': None,
    'title': None,
    # 'xlim': None,
    # 'ylim': None,
    # ...
}

# any matplotlib rcParams keys can also be used as settings under the following transformation
format_setting_to_rc_params = lambda k: k.replace('_', '.')
format_rcparams_to_settings = lambda k: k.replace('.', '_')
EXCLUDED_SET_SETTINGS = {'label', 'zorder', 'alpha'} # excluded label so that it can be used in the plotting function instead
EXCLUDED_RC_PARAMS = {'path.effects'} # excluded due to e.g. overlap with other settings
ALLRCPARAMKEYS = set(k for k in plt.rcParams.keys() if k not in EXCLUDED_RC_PARAMS)
ALLRCPARAMKEYS_REFORMATTED = set(format_rcparams_to_settings(k) for k in ALLRCPARAMKEYS)
ALL_SET_SETTINGS = set([k for k in plt.gca().properties().keys() if k not in EXCLUDED_SET_SETTINGS]); plt.close()
ALL_POSSIBLE_SETTINGS = set(DEFAULT_CUSTOM_SUBPLOT_SETTINGS.keys()).union(ALL_SET_SETTINGS).union(ALLRCPARAMKEYS_REFORMATTED)
assert len(ALL_POSSIBLE_SETTINGS) == len(DEFAULT_CUSTOM_SUBPLOT_SETTINGS) + len(ALL_SET_SETTINGS) + len(ALLRCPARAMKEYS_REFORMATTED), 'There is overlap in the possible settings, please check ALL_POSSIBLE_SETTINGS.'


# style context managers
def set_defaults(**kwargs):
    """Set the default settings used for all future jv plots."""
    global DEFAULT_CUSTOM_SUBPLOT_SETTINGS, DEFAULT_SET_SETTINGS
    for k, v in kwargs.items():
        if k in DEFAULT_CUSTOM_SUBPLOT_SETTINGS:
            DEFAULT_CUSTOM_SUBPLOT_SETTINGS[k] = v
        elif k in DEFAULT_SET_SETTINGS:
            DEFAULT_SET_SETTINGS[k] = v
        elif k in ALLRCPARAMKEYS_REFORMATTED:
            warnings.warn('You are setting a matplotlib rcParam directly, this will affect all future plots, not just those using jv functions.')
            matplotlib.rcParams[format_setting_to_rc_params(k)] = v
        else:
            raise ValueError('Invalid setting: {}'.format(k))

def set_style(style, substyle=None, fig_aspect=None):
    """Switch the default figure size used for all future jv plots."""
    if style not in CUSTOM_PAPER_SETTINGS and style is not None:
        raise ValueError('Invalid paper type: {}'.format(style))
    set_defaults(fig_style=style, fig_substyle=substyle, fig_aspect=fig_aspect)
use_paper=set_style # deprecated alias

class style_context: # TODO: expand this to other settings also and make it more general?
    def __init__(self, style=None, substyle=None, fig_aspect=None):
        self.org_style = (
            DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_style'], 
            DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_substyle'], 
            DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_aspect']
        )
        self.new_style = (style, substyle, fig_aspect)

    def __enter__(self):
        set_style(*self.new_style)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_style(*self.org_style)

# standard functionality decorator

def iterate_height_to_get_fixed_width(fig, max_iter=100, fig_aspect=1.0, fallback_aspect=1.0, _debug=False):
    """Resize the canvas proportionally to match its original width after tight cropping.

    Both dimensions may change so height-limited, equal-aspect axes can grow too; font
    sizes and axes aspect constraints are preserved. Width is matched within one output
    pixel. Nonconvergence keeps the closest measured size and warns instead of failing.
    fig_aspect/fallback_aspect are used only to reset an unsafe canvas-size proposal.
    """

    # measure at the export resolution and target the center of the desired pixel bin
    if max_iter < 1: raise ValueError('max_iter must be positive.')
    dpi = matplotlib.rcParams['savefig.dpi']
    dpi = fig.get_dpi() if dpi == 'figure' else float(dpi)
    fig.set_dpi(dpi)
    original_size = fig.get_size_inches().copy()
    goal_width_pixels = max(1, int(round(original_size[0]*dpi)))
    target_width = goal_width_pixels+0.5
    best_size, best_width, best_error = original_size.copy(), np.nan, np.inf
    stalled = 0

    # resize both dimensions without changing font sizes or the aspect of the plotted data
    for iteration in range(max_iter):
        fig.canvas.draw()
        tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        real_width_pixels = (tight_bbox.width+2*matplotlib.rcParams['savefig.pad_inches'])*dpi
        if not np.isfinite(real_width_pixels) or real_width_pixels <= 0:
            warnings.warn('Invalid tight bounding-box width; keeping the closest measured figure size.')
            fig.set_size_inches(best_size)
            return
        error = abs(real_width_pixels-target_width)
        improved = error < best_error-0.01
        stalled = 0 if improved else stalled+1
        if error < best_error:
            best_size, best_width, best_error = fig.get_size_inches().copy(), real_width_pixels, error
        if _debug: print('iteration {}: cropped width {:.3f} pixels, target {} pixels'.format(iteration+1, real_width_pixels, goal_width_pixels))
        if error <= 0.25: return
        if stalled >= 2 or iteration+1 == max_iter: break
        next_size = fig.get_size_inches()*np.clip(target_width/real_width_pixels, 0.5, 2.0)

        # bound pathological resizing and retain the original fallback aspect controls
        if np.any(next_size/original_size > 100) or np.any(next_size/original_size < 0.01):
            aspect = fallback_aspect if fig_aspect is None else fig_aspect
            height = original_size[1] if aspect == 'auto' else original_size[0]/(1.0 if aspect == 'equal' else aspect)
            warnings.warn('Large discrepancy in figure width; resetting the canvas to its original width.')
            fig.set_size_inches(original_size[0], height)
            return
        fig.set_size_inches(next_size)

    # preserve the best result when the layout cannot reach the requested width
    fig.set_size_inches(best_size)
    warnings.warn('Failed to achieve fixed width after {} iterations, saving with closest width ({:.3f} pixels, target {} pixels).'.format(iteration+1, best_width, goal_width_pixels))


def basic_figure_wrapper(**wrapper_settings):
    """General wrapper that encapulates many commonly repeated parts when plotting."""
    for wkwarg in wrapper_settings:
        if wkwarg not in ALL_POSSIBLE_SETTINGS:
            raise ValueError('Invalid custom figure setting: {}'.format(wkwarg))

    def decorator_real(plotting_function): # intermediate decorator to allow for custom settings to be passed in
        @functools.wraps(plotting_function)
        def wrapper_function(*args, ax=None, **kwargs):
            # update kwargs with defaults
            for wkwarg in wrapper_settings:
                kwargs.setdefault(wkwarg, wrapper_settings[wkwarg])
            for wkwarg in DEFAULT_CUSTOM_SUBPLOT_SETTINGS:
                kwargs.setdefault(wkwarg, DEFAULT_CUSTOM_SUBPLOT_SETTINGS[wkwarg])
            for wkwarg in DEFAULT_SET_SETTINGS:
                kwargs.setdefault(wkwarg, DEFAULT_SET_SETTINGS[wkwarg])
            
            # extract the kwargs going to the plotting function and other settings
            custom_settings = {k: v for k, v in kwargs.items() if k in DEFAULT_CUSTOM_SUBPLOT_SETTINGS}
            rc_params = {format_setting_to_rc_params(k): v for k, v in kwargs.items() if k not in custom_settings and k in ALLRCPARAMKEYS_REFORMATTED}
            set_settings = {k: v for k, v in kwargs.items() if k in ALL_SET_SETTINGS and k not in custom_settings and k not in ALLRCPARAMKEYS_REFORMATTED}
            function_kwargs = {k: v for k, v in kwargs.items() if k not in set_settings and k not in custom_settings and k not in ALLRCPARAMKEYS_REFORMATTED}

            # DEBUG
            # print()
            # print('[DEGUG] ALL_POSSIBLE_SETTINGS:', ALL_POSSIBLE_SETTINGS)
            # print('[DEGUG] wrapper_settings:', wrapper_settings)
            # print()
            # print('[DEGUG 1] custom_settings:', custom_settings)
            # print('[DEGUG 1] rc_params:', rc_params)
            # print('[DEGUG 1] set_settings:', set_settings)
            # print('[DEGUG 1] function_kwargs:', function_kwargs)
            # print()

            # update kwargs with any overriding "meta" settings
            if custom_settings['fig_style'] is not None:
                rc_params = set_style_defaults(
                    settings=rc_params,
                    style=custom_settings['fig_style'],  
                    substyle=custom_settings['fig_substyle'],
                    fig_aspect=custom_settings['fig_aspect'],
                )
            else: # if "default" scheme
                custom_settings['fixed_width'] = False # no fixed width bbox adjustment if matplotlib default
            if kwargs.get('no_ticks', False):
                rc_params['xtick.major.size'] = 0
                rc_params['ytick.major.size'] = 0
                rc_params['xtick.minor.size'] = 0
                rc_params['ytick.minor.size'] = 0
                custom_settings.pop("no_ticks", None) # remove any meta settings after so they don't interfere with any other settings
            if kwargs.get('no_minor_ticks', False):
                rc_params['xtick.minor.size'] = 0
                rc_params['ytick.minor.size'] = 0
                custom_settings.pop("no_minor_ticks", None)
            if kwargs.get('no_tick_labels', False):
                rc_params['xtick.labelsize'] = 0
                rc_params['ytick.labelsize'] = 0
                custom_settings.pop("no_tick_labels", None)
            if kwargs.get('force_minor_ticks', False):
                custom_settings['xminor_nsubs'] = 10
                custom_settings['yminor_nsubs'] = 10
                custom_settings.pop("force_minor_ticks", None)
            if kwargs.get('default_ticks_on_colorbar', False):
                custom_settings['colorbar_rcparams']['xtick.major.size'] = matplotlib.rcParams['xtick.major.size']
                custom_settings['colorbar_rcparams']['ytick.major.size'] = matplotlib.rcParams['ytick.major.size']
                custom_settings['colorbar_rcparams']['xtick.minor.size'] = matplotlib.rcParams['xtick.minor.size']
                custom_settings['colorbar_rcparams']['ytick.minor.size'] = matplotlib.rcParams['ytick.minor.size']
                custom_settings.pop("default_ticks_on_colorbar", None)
            if kwargs.get('loglog', None) is not None:
                set_settings['xscale'] = 'log' if kwargs['loglog'] else 'linear'
                set_settings['yscale'] = 'log' if kwargs['loglog'] else 'linear'
                custom_settings.pop("loglog", None)
            if kwargs.get('xlog', None) is not None:
                set_settings['xscale'] = 'log' if kwargs['xlog'] else 'linear'
                custom_settings.pop("xlog", None)
            if kwargs.get('ylog', None) is not None:
                set_settings['yscale'] = 'log' if kwargs['ylog'] else 'linear'
                custom_settings.pop("ylog", None)
            if kwargs.get('xmin', None) is not None:
                set_settings['xlim'] = (kwargs['xmin'], set_settings.get('xlim', (None, None))[1])
            if kwargs.get('xmax', None) is not None:
                set_settings['xlim'] = (set_settings.get('xlim', (None, None))[0], kwargs['xmax'])
            if kwargs.get('ymin', None) is not None:
                set_settings['ylim'] = (kwargs['ymin'], set_settings.get('ylim', (None, None))[1])
            if kwargs.get('ymax', None) is not None:
                set_settings['ylim'] = (set_settings.get('ylim', (None, None))[0], kwargs['ymax'])
            # ... add more meta settings here as needed ...

            # DEBUG
            # print()
            # print('[DEGUG 2] custom_settings:', custom_settings)
            # print('[DEGUG 2] rc_params:', rc_params)
            # print('[DEGUG 2] set_settings:', set_settings)
            # print('[DEGUG 2] function_kwargs:', function_kwargs)
            # print()

            # use rc_context to temporarily set rc parameters for this plot
            with matplotlib.rc_context(rc_params):
                ## get figure and axes
                if custom_settings['clf_before']:
                    plt.clf()

                if ax is None:
                    ax = plt.gca()  # will create a figure/axes if one does not exist, otherwise grab the current one
                    assert ax is not None, 'Failed to create axes.'
                
                fig = ax.figure
                if fig is None:
                    raise ValueError('Failed to get figure from axes.')
                fig.set_size_inches(matplotlib.rcParams['figure.figsize']) # ensure that the figure size is the size we want (avoids plt.gca() returning a figure with a different size than we want)

                ## plotting function call
                ret = plotting_function(*args, ax=ax, **function_kwargs)

                ## axes settings
                ax.set(**set_settings)

                ## custom settings ##
                is_done = custom_settings['out'] is not None or custom_settings['show']

                # legend settings
                show_legend = custom_settings['show_legend']
                if type(custom_settings['show_legend_on_ax_num']) is bool:
                    show_legend = custom_settings['show_legend_on_ax_num']
                elif type(custom_settings['show_legend_on_ax_num']) is int:
                    ax_num = fig.axes.index(ax)
                    if custom_settings['show_legend_on_ax_num'] == ax_num:
                        show_legend = True
                    if custom_settings['show_legend_on_ax_num'] == -1 and is_done:
                        show_legend = True

                if show_legend: 
                    handles, labels = ax.get_legend_handles_labels()
                    if handles and labels:
                        ax.legend()

                # colorbar settings
                if custom_settings['show_colorbar'] and bool(ax.images or ax.collections or ax.contours):
                    with matplotlib.rc_context(custom_settings['colorbar_rcparams']):
                        if custom_settings['colorbar_on_fig']:
                            axes = fig.get_axes()
                            images = [im for ax in axes for im in ax.get_images()]
                        else:
                            images = ax.get_images()

                        if len(images) == 0:
                            raise ValueError('No images found for colorbar.')
                        if len(images) > 1:
                            raise ValueError('Multiple images found for colorbar, currently not supported.')
                        
                        if custom_settings['colorbar_on_fig']:
                            fig.colorbar(images[0], label=custom_settings['colorbar_label'])
                        else:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(images[0], cax=cax, label=custom_settings['colorbar_label'])
                
                # major tick settings
                log_xbase = 2.0 if 'log2' in ax.get_xscale() else 10.0 if 'log' in ax.get_xscale() else None
                log_ybase = 2.0 if 'log2' in ax.get_yscale() else 10.0 if 'log' in ax.get_yscale() else None
                if custom_settings['force_regular_log_major_ticks']:
                    if 'log' in ax.get_xscale() and log_xbase is not None:
                        ax.xaxis.set_major_locator(LogLocator(base=log_xbase, subs=(1.0,), numticks=9999)) # arbitrarily large numticks to ensure all ticks are shown
                    if 'log' in ax.get_yscale() and log_ybase is not None:
                        ax.yaxis.set_major_locator(LogLocator(base=log_ybase, subs=(1.0,), numticks=9999)) 
                if custom_settings['xmajor_locator'] is not None:
                    ax.xaxis.set_major_locator(custom_settings['xmajor_locator'])
                if custom_settings['ymajor_locator'] is not None:
                    ax.yaxis.set_major_locator(custom_settings['ymajor_locator'])

                # minor tick settings
                if custom_settings['xminor_nsubs'] is not None:
                    if custom_settings['xminor_locator'] is not None:
                        warnings.warn('Both xminor_nsubs and xminor_locator are set, using xminor_locator and ignoring xminor_nsubs.')
                    else:
                        if log_xbase is not None:
                            subs = list(range(2, int(log_xbase))) if log_xbase > 2 and custom_settings['xminor_nsubs'] == 10 else list(log_xbase ** (x / custom_settings['xminor_nsubs']) for x in range(1, custom_settings['xminor_nsubs']))
                            numticks = (len(ax.xaxis.get_majorticklocs()) + 2) * len(subs)
                            custom_settings['xminor_locator'] = LogLocator(base=log_xbase, subs=subs, numticks=numticks)
                        else:
                            custom_settings['xminor_locator'] = AutoMinorLocator(custom_settings['xminor_nsubs'])
                if custom_settings['yminor_nsubs'] is not None:
                    if custom_settings['yminor_locator'] is not None:
                        warnings.warn('Both yminor_nsubs and yminor_locator are set, using yminor_locator and ignoring yminor_nsubs.')
                    else:
                        if log_ybase is not None:
                            subs = list(range(2, int(log_ybase))) if log_ybase > 2 and custom_settings['yminor_nsubs'] == 10 else list(log_ybase ** (x / custom_settings['yminor_nsubs']) for x in range(1, custom_settings['yminor_nsubs']))
                            numticks = (len(ax.yaxis.get_majorticklocs()) + 2) * len(subs)
                            custom_settings['yminor_locator'] = LogLocator(base=log_ybase, subs=subs, numticks=numticks)
                        else:
                            custom_settings['yminor_locator'] = AutoMinorLocator(custom_settings['yminor_nsubs'])
                if custom_settings['xminor_locator'] is not None:
                    ax.xaxis.set_minor_locator(custom_settings['xminor_locator'])
                if custom_settings['yminor_locator'] is not None:
                    ax.yaxis.set_minor_locator(custom_settings['yminor_locator'])

                # closing settings
                if custom_settings['out'] is not None:
                    if custom_settings['fixed_width']:  # custom iteration to fix the resulting width of saved figure and have tight bbox
                        #import time #DEBUG
                        #t= time.time() #DEBUG
                        #print('new output: {}'.format(custom_settings['out'])) # DEBUG
                        iterate_height_to_get_fixed_width(fig, max_iter=custom_settings['fixed_width_max_iter'], fig_aspect=custom_settings['fig_aspect'])
                        #print('iteration time:', time.time()-t) #DEBUG

                        ##save image
                        #img.save(custom_settings['out'])
                        #t = time.time() #DEBUG
                        fig.savefig(custom_settings['out'], dpi=matplotlib.rcParams['savefig.dpi'], bbox_inches='tight')
                        #print('plot time:', time.time()-t) #DEBUG
                    
                    else:
                        fig.savefig(custom_settings['out'])
                if custom_settings['show']:
                    plt.show()
                
                if custom_settings['clf_after'] is None:
                    custom_settings['clf_after'] = is_done
                if custom_settings['clf_after']:
                    plt.clf()

                if custom_settings['close_after'] is None:
                    custom_settings['close_after'] = is_done
                if custom_settings['close_after']:
                    plt.close(fig)
                
                return ret
        return wrapper_function
    return decorator_real



def general_1d_wrapper(plotting_function_name, _possible_changing_kwargs={'y2','label','color','ls','lw','zorder','alpha'}, **wrapper_settings):
    """Wrapper for 1D plotting functions that takes care of the figure and axes creation."""
    @basic_figure_wrapper(**wrapper_settings)
    def wrapper_function(*args, ax=None, **kwargs):
        n_lines = max(
            max(len(x) if isinstance(x, tuple) else 1 for x in args),
            max(len(kwargs[k]) if k in kwargs and isinstance(kwargs[k], tuple) else 1 for k in _possible_changing_kwargs)
        )
        new_args = []
        for i in range(len(args)):
            if not isinstance(args[i], tuple):
                new_args.append((args[i],) * n_lines)
            else:
                new_args.append(args[i])
        for k in _possible_changing_kwargs:
            if k in kwargs and not isinstance(kwargs[k], tuple):
                kwargs[k] = (kwargs[k],) * n_lines
        rets = []
        for i in range(n_lines):
            local_args = [new_args[j][i] for j in range(len(new_args))]
            local_kwargs = {k: kwargs[k][i] if k in _possible_changing_kwargs else kwargs[k] for k in kwargs}
            rets.append(getattr(ax, plotting_function_name)(*local_args, **local_kwargs))
        if n_lines == 1:
            return rets[0]
        return rets
    return wrapper_function



# general plotting functions 
plot = general_1d_wrapper('plot')
fill_between = general_1d_wrapper('fill_between')

def loglog(*args, **kwargs):
    """General 1D plotting function for lines on log-log axes."""
    # Note that if you override this setting at a later point before ending the figure, it will override this.
    kwargs['loglog'] = True
    return plot(*args, **kwargs)

@basic_figure_wrapper()
def scatter(x, y, ax=None, **kwargs): 
    """General scatter plot function."""
    return ax.scatter(x, y, **kwargs)

@basic_figure_wrapper()
def hist(x, ax=None, **kwargs): 
    """General histogram plot function."""
    return ax.hist(x, **kwargs)

@basic_figure_wrapper(show_colorbar=True, no_minor_ticks=True)
def imshow(X, ax=None, colorbar=True, **kwargs): 
    """General 2D plotting function for images."""
    return ax.imshow(X, **kwargs)

@basic_figure_wrapper()
def blank_plot(**kwargs):
    pass
close=blank_plot # alias for ending a plot without plotting anything else in that line





#############################################################################
### MULTIPANEL FIGURES

class Figure:
    """Manage a lazy subplot grid using the same settings as the standalone helpers.

    Figure(2, 3) or Figure(grid=(2, 3)) creates a grid. Plotting uses the first empty panel;
    ax accepts a flat index, a (row, column) pair, a slice selection, or owned axes.
    fig[row, col].plot(...) uses the helpers; its .ax exposes the Matplotlib axes.
    Slices select existing or newly created panels, deduplicating spanning axes.
    add_axis(slice(...), slice(...)) creates a single panel spanning empty cells.
    Per-call settings override panel history and constructor defaults. Plot methods
    return the helper's result, or a list of results for multiple selected panels.
    draw(show=True) displays; draw(out=path) saves. Neither clears nor closes by
    default; use clear(), close(), clf_after=True, or close_after=True explicitly.
    Existing figures must use one shared GridSpec, excluding colorbar axes.
    """
    _creation_settings = {'figsize', 'dpi', 'facecolor', 'edgecolor', 'frameon', 'layout', 'tight_layout', 'constrained_layout'}
    _grid_settings = {'width_ratios', 'height_ratios', 'wspace', 'hspace', 'left', 'right', 'top', 'bottom'}
    _end_settings = {'out', 'show', 'clf_after', 'close_after', 'clf_before'}
    _plot_methods = {'plot', 'scatter', 'imshow', 'hist', 'fill_between', 'loglog'}

    def __init__(self, nrows=1, ncols=1, fig=None, grid=None, **settings):

        # separate construction options from reusable plotting defaults
        if grid is not None:
            if (nrows, ncols) != (1, 1) or len(grid) != 2: raise ValueError('Use either grid=(nrows, ncols) or nrows and ncols.')
            nrows, ncols = grid
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1 for n in (nrows, ncols)):
            raise ValueError('nrows and ncols must be positive integers.')
        allowed = ALL_POSSIBLE_SETTINGS | self._creation_settings | self._grid_settings
        if settings.keys()-allowed: raise ValueError(f'Invalid figure or subplot settings: {sorted(settings.keys()-allowed)}')
        creation = {key: value for key, value in settings.items() if key in self._creation_settings}
        self._grid_kwargs = {key: value for key, value in settings.items() if key in self._grid_settings}
        self.settings = {key: value for key, value in settings.items() if key not in self._creation_settings | self._grid_settings}
        self._panel_settings = {}

        # create a pyplot-managed figure or recover the shared grid of an existing figure
        if fig is None:
            with matplotlib.rc_context(self._rc_settings(self.settings)):
                fig = plt.figure(**creation)
            spec = fig.add_gridspec(nrows, ncols, **self._grid_kwargs)
            axes = [None]*(nrows*ncols)
        else:
            if not isinstance(fig, MatplotlibFigure): raise TypeError('fig must be a Matplotlib figure.')
            if (nrows, ncols) != (1, 1) or grid is not None or creation or self._grid_kwargs:
                raise ValueError('An existing figure supplies its own grid and creation settings.')
            panels = [ax for ax in fig.axes if getattr(ax, '_colorbar', None) is None and ax.get_subplotspec() is not None]
            if not panels: raise ValueError('The existing figure must contain subplot axes.')
            spec = panels[0].get_subplotspec().get_gridspec()
            if any(ax.get_subplotspec().get_gridspec() is not spec for ax in panels):
                raise ValueError('Existing subplot axes must share one GridSpec.')
            nrows, ncols = spec.get_geometry()
            axes = [None]*(nrows*ncols)
            for ax in panels:
                sub = ax.get_subplotspec()
                for row in sub.rowspan:
                    for col in sub.colspan:
                        idx = row*ncols+col
                        if axes[idx] is not None: raise ValueError('Existing subplot axes must not overlap.')
                        axes[idx] = ax
            self._grid_kwargs = dict(width_ratios=spec.get_width_ratios(), height_ratios=spec.get_height_ratios())
        self.fig, self.spec, self.axes = fig, spec, axes
        self.nrows, self.ncols = nrows, ncols

    def _rc_settings(self, settings):

        # resolve the same paper style and rc parameters as standalone plotting helpers
        rc = {format_setting_to_rc_params(key): value for key, value in settings.items() if key in ALLRCPARAMKEYS_REFORMATTED}
        style = settings.get('fig_style', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_style'])
        if style is not None:
            rc = set_style_defaults(rc, style=style,
                substyle=settings.get('fig_substyle', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_substyle']),
                fig_aspect=settings.get('fig_aspect', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_aspect']))
        if hasattr(self, 'fig'): rc['figure.figsize'] = self.fig.get_size_inches()
        return rc

    def get_axis_index(self, *index):
        """Return a flat cell index or a list of cell indices in reading order."""
        if len(index) == 1 and isinstance(index[0], tuple): index = index[0]
        if len(index) not in (1, 2): raise ValueError('Use a flat index or a row and column selection.')

        # normalize ordinary python indices and slices against the grid dimensions
        selected = []
        for key, size in zip(index, (self.nrows*self.ncols,) if len(index) == 1 else (self.nrows, self.ncols)):
            if isinstance(key, slice):
                selected.append(list(range(*key.indices(size))))
            elif isinstance(key, (int, np.integer)) and not isinstance(key, bool):
                key = int(key)+size if key < 0 else int(key)
                if not 0 <= key < size: raise IndexError('Panel index is outside the grid.')
                selected.append([key])
            else:
                raise TypeError('Panel indices must be integers or slices.')
        indices = selected[0] if len(index) == 1 else [row*self.ncols+col for row in selected[0] for col in selected[1]]
        return indices if any(isinstance(key, slice) for key in index) else indices[0]

    def add_axis(self, row, col=None, **kwargs):
        """Create or reuse one panel; slices must describe a contiguous rectangle."""
        index = self.get_axis_index(row) if col is None else self.get_axis_index(row, col)
        indices = index if isinstance(index, list) else [index]
        if not indices: raise ValueError('Cannot create a panel from an empty selection.')

        # preserve existing panels and reject overlaps instead of deleting plotted data
        rows, cols = [i//self.ncols for i in indices], [i%self.ncols for i in indices]
        cells = {r*self.ncols+c for r in range(min(rows), max(rows)+1) for c in range(min(cols), max(cols)+1)}
        if cells != set(indices): raise ValueError('A spanning panel must occupy a contiguous rectangle.')
        occupied = {self.axes[i] for i in indices if self.axes[i] is not None}
        if occupied:
            if len(occupied) == 1:
                ax = next(iter(occupied))
                if cells == {i for i, existing in enumerate(self.axes) if existing is ax}:
                    if kwargs: raise ValueError('Cannot change creation options of an existing panel.')
                    return ax
            raise ValueError('The requested panel overlaps existing axes.')
        with matplotlib.rc_context(self._rc_settings(self.settings)):
            ax = self.fig.add_subplot(self.spec[min(rows):max(rows)+1, min(cols):max(cols)+1], **kwargs)
        for idx in indices: self.axes[idx] = ax
        return ax

    def _select_axes(self, selection):

        # accept owned matplotlib axes as well as grid selections and create missing panels lazily
        if isinstance(selection, _FigureSelection):
            if selection.owner is not self: raise ValueError('Panel selection belongs to another figure.')
            if any(ax not in self.fig.axes for ax in selection.axes): raise ValueError('Panel selection was cleared from the figure.')
            return selection.axes
        if any(selection is ax for ax in self.axes if ax is not None): return [selection]
        if selection is None:
            selection = next((i for i, ax in enumerate(self.axes) if ax is None or not ax.has_data()), None)
            if selection is None: raise ValueError('All panels contain data; select a panel explicitly to overlay another plot.')
        indices = self.get_axis_index(selection)
        indices = indices if isinstance(indices, list) else [indices]
        if not indices: raise ValueError('The panel selection is empty.')
        axes = []
        for idx in indices:
            ax = self.axes[idx] if self.axes[idx] is not None else self.add_axis(idx)
            if ax not in axes: axes.append(ax)
        return axes

    def __getitem__(self, key):
        return _FigureSelection(self, self._select_axes(key))

    def _plot(self, method, *args, ax=None, **settings):

        # apply helper settings to each panel without ending or resizing the entire figure
        axes = self._select_axes(ax)
        ending = {**self.settings, **settings}
        if ending.get('clf_before', False): raise ValueError('Use Figure.clear() before plotting instead of clf_before.')
        results = []
        for panel in axes:
            options = {**self.settings, **self._panel_settings.get(panel, {}), **settings}
            if method == 'loglog': options['loglog'] = True
            self._panel_settings[panel] = {key: value for key, value in options.items()
                if key in ALL_POSSIBLE_SETTINGS and key not in self._end_settings | {'show_colorbar'}}
            rc = self._rc_settings(options)
            options.update({format_rcparams_to_settings(key): value for key, value in rc.items()})
            options.update(fig_style=None, out=None, show=False, clf_before=False, clf_after=False, close_after=False)
            results.append(globals()[method](*args, ax=panel, **options))
        if ending.get('out') is not None or ending.get('show', False):
            self.draw(**{key: value for key, value in ending.items() if key in self._end_settings})
        return results[0] if len(results) == 1 else results

    def plot(self, *args, ax=None, **settings):
        return self._plot('plot', *args, ax=ax, **settings)

    def imshow(self, *args, ax=None, **settings):
        return self._plot('imshow', *args, ax=ax, **settings)

    def scatter(self, *args, ax=None, **settings):
        return self._plot('scatter', *args, ax=ax, **settings)

    def hist(self, *args, ax=None, **settings):
        return self._plot('hist', *args, ax=ax, **settings)

    def fill_between(self, *args, ax=None, **settings):
        return self._plot('fill_between', *args, ax=ax, **settings)

    def loglog(self, *args, ax=None, **settings):
        return self._plot('loglog', *args, ax=ax, **settings)

    def draw(self, **settings):
        """Apply settings, render, optionally save/display, and retain panels by default."""
        if settings.keys()-ALL_POSSIBLE_SETTINGS: raise ValueError(f'Invalid draw settings: {sorted(settings.keys()-ALL_POSSIBLE_SETTINGS)}')
        options = {**self.settings, **settings}
        if options.get('clf_before', False): raise ValueError('Use Figure.clear() explicitly before drawing.')

        # update selected axes properties without reapplying defaults or adding duplicate colorbars
        panel_options = {key: value for key, value in settings.items() if key not in self._end_settings}
        for ax in dict.fromkeys(ax for ax in self.axes if ax is not None):
            if panel_options:
                saved = {**self.settings, **self._panel_settings.get(ax, {}), **panel_options}
                saved.update({format_rcparams_to_settings(key): value for key, value in self._rc_settings(saved).items()})
                saved.update(fig_style=None, show_colorbar=False, out=None, show=False, clf_before=False, clf_after=False, close_after=False)
                blank_plot(ax=ax, **saved)
                self._panel_settings.setdefault(ax, {}).update(panel_options)
            if options.get('show_legend', False) or options.get('show_legend_on_ax_num') == -1:
                handles, labels = ax.get_legend_handles_labels()
                if handles: ax.legend()

        # save and finish this figure without clearing or closing another active figure
        with matplotlib.rc_context(self._rc_settings(options)):
            self.fig.canvas.draw()
            if options.get('out') is not None:
                fixed = options.get('fixed_width', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fixed_width'])
                if fixed:
                    iterate_height_to_get_fixed_width(self.fig,
                        max_iter=options.get('fixed_width_max_iter', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fixed_width_max_iter']),
                        fig_aspect=options.get('fig_aspect', DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_aspect']))
                self.fig.savefig(options['out'], dpi=matplotlib.rcParams['savefig.dpi'], **({'bbox_inches': 'tight'} if fixed else {}))
            if options.get('show', False):
                if matplotlib.get_backend() == 'module://matplotlib_inline.backend_inline': plt.show(close=False)
                else: plt.show()
        if options.get('clf_after', False): self.clear()
        if options.get('close_after', False): self.close()
        return self

    def clear(self):
        """Clear this figure and restore an empty grid with the same dimensions."""
        self.fig.clear()
        self.spec = self.fig.add_gridspec(self.nrows, self.ncols, **self._grid_kwargs)
        self.axes = [None]*(self.nrows*self.ncols)
        self._panel_settings.clear()
        return self

    def close(self):
        """Close this figure explicitly; retaining a panel reference does not close it."""
        plt.close(self.fig)
        return self


class _FigureSelection:
    """Route plotting through the parent defaults and expose single-panel axes attributes."""
    def __init__(self, owner, axes):
        self.owner, self.axes = owner, axes

    @property
    def ax(self):
        if len(self.axes) != 1: raise ValueError('Use .axes for a selection containing multiple panels.')
        return self.axes[0]

    def __getattr__(self, name):
        if name in self.owner._plot_methods: return functools.partial(self.owner._plot, name, ax=self)
        return getattr(self.ax, name)
