import io, functools, warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.ticker import AutoMinorLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageChops

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


def construct_paper_figsize(paper=None, aspect=None, subtype=None):
    """Construct a figure size (width, height) in inches based on the paper type, aspect ratio, and column type."""
    if paper not in CUSTOM_PAPER_SETTINGS:
        raise ValueError('Invalid paper type: {}'.format(paper))
    if aspect is None:
        aspect = 'auto'
    if subtype is None:
        subtype = CUSTOM_PAPER_SETTINGS[paper]['default_subtype']

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

def set_paper_defaults(settings={}, paper=None, aspect=None, subtype=None):
    """Set the default figure size and font size based on the paper type, aspect, and column type."""
    # settings are rc params only
    if paper not in CUSTOM_PAPER_SETTINGS and paper is not None:
        raise ValueError('Invalid paper type: {}'.format(paper))
    
    if paper is None:
        # set to default mpl settings
        figsize = MATPLOTLIB_DEFAULT_FIGSIZE
        fontsize = 10
    else:
        figsize = construct_paper_figsize(paper=paper, aspect=aspect, subtype=subtype)
        fontsize = CUSTOM_PAPER_SETTINGS[paper]['font_size']

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
    'fig_paper': 'ApJ',                          # if set, will override figsize and fontsize settings with that paper defaults
    'fig_aspect': 'auto',                        # if set, will override figure aspect ratio (width/height) with this value using paper default width 
    'fig_subtype': None,                         # if set, will override figure subtype (singlecol/doublecol) with this value using paper default width

    'no_ticks': False,                           # if true, will set all tick sizes to 0 (overrides any other tick size settings)
    'no_minor_ticks': False,                     # if true, will set all minor tick sizes to 0 (overrides any other minor tick size settings)
    'no_tick_labels': False,                     # if true, will set all tick label sizes to 0 (overrides any other tick label size settings)
    'force_minor_ticks': False,                  # if true, will try to force minor ticks on x and y axes
    'force_regular_log_major_ticks': False,      # if true, will force regular log major ticks on x and y axes (i.e. 1, 10, 100, etc. not 1, 100, etc.)
    'default_ticks_on_colorbar': True,           # if true, will force default ticks on colorbar
    
    'xlog': None,
    'ylog': None,
    'xminor_nsubs': None,         # if set, will use AutoMinorLocator with this number of subdivisions for the x-axis minor ticks (overrides any other x-axis minor tick locator settings)
    'yminor_nsubs': None,

    # add optional components
    'show_legend': False,
    'show_colorbar': False,
    
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
EXCLUDED_RC_PARAMS = {'path.effects'} # excluded due to e.g. overlap with other settings
ALLRCPARAMKEYS = set(k for k in plt.rcParams.keys() if k not in EXCLUDED_RC_PARAMS)
ALLRCPARAMKEYS_REFORMATTED = set(format_rcparams_to_settings(k) for k in ALLRCPARAMKEYS)
ALL_SET_SETTINGS = set(plt.gca().properties().keys()); plt.close()
ALL_POSSIBLE_SETTINGS = set(DEFAULT_CUSTOM_SUBPLOT_SETTINGS.keys()).union(ALL_SET_SETTINGS).union(ALLRCPARAMKEYS_REFORMATTED)
assert len(ALL_POSSIBLE_SETTINGS) == len(DEFAULT_CUSTOM_SUBPLOT_SETTINGS) + len(ALL_SET_SETTINGS) + len(ALLRCPARAMKEYS_REFORMATTED), 'There is overlap in the possible settings, please check ALL_POSSIBLE_SETTINGS.'


def use_paper(paper, subtype=None, aspect=None):
    """Switch the default figure size used for all future jv plots."""
    if paper not in CUSTOM_PAPER_SETTINGS and paper is not None:
        raise ValueError('Invalid paper type: {}'.format(paper))
    global DEFAULT_CUSTOM_SUBPLOT_SETTINGS
    DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_paper'] = paper
    if aspect is not None:
        DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_aspect'] = aspect
    if subtype is not None:
        DEFAULT_CUSTOM_SUBPLOT_SETTINGS['fig_subtype'] = subtype

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





def iterate_height_to_get_fixed_width(fig, max_iter=100, fig_aspect=1.0, fallback_aspect=1.0, _debug=False):
    """Iteratively adjust the figure height to achieve a fixed width with tight bbox, but variable height/figure aspect ratio."""
    dpi = matplotlib.rcParams['savefig.dpi']
    fig.set_dpi(dpi) # make sure dpi is right
    goal_width_inches = float(fig.get_figwidth())         # approximate width in inches (within pixel round-off error)
    goal_width_pixels = int(round(goal_width_inches*dpi)) # target exact width in pixels

    real_width_pixels = 0  # real width in pixels after tight bbox cut (will be updated in loop)
    i = max_iter           # max iterations to adjust figure size
    while goal_width_pixels != real_width_pixels:
        ## get the real width in pixels using the tight bbox
        fig.canvas.draw()    # expensive step!
        tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        real_width_pixels = int(round((tight_bbox.width + 2 * matplotlib.rcParams['savefig.pad_inches']) * dpi))
        
        ## adjust mpl figure size to get the desired width in pixels
        full_width_inches = fig.get_figwidth() # before bbox cut
        adjustment_factor = goal_width_pixels / real_width_pixels
        adjustment_factor2 = goal_width_inches / full_width_inches
        if _debug:
            print('want {} pix ({:.3f} in) - have {} pix ({:.3f} in) -> adjust by factor {} (~ {})'.format(goal_width_pixels, goal_width_inches, real_width_pixels, fig.get_figwidth(), adjustment_factor2, adjustment_factor))
            print(' > fig height: {:.3f} in'.format(fig.get_figheight())) 
        if adjustment_factor != 1.0:
            fig.set_figwidth(fig.get_figwidth()*adjustment_factor)                     
        
        ## check if the adjustment is too large
        check_ratio = full_width_inches/goal_width_inches
        if check_ratio > 100 or check_ratio < 0.01:
            warnings.warn('[PLOT ERROR] Large discrepancy in figure width ({} pixels vs {} pixels), stopping adjustment and defaulting to original size.'.format(goal_width_pixels, real_width_pixels)) # this can sometimes to occur if the figure height is not large enough, check that first
            aspect = 1.0 if fig_aspect == 'equal' else fig_aspect if fig_aspect != 'auto' else fallback_aspect # default to aspect ratio of 1 if none defined
            fig.set_size_inches((goal_width_inches, goal_width_inches/aspect)) # reset to original size
            fig.canvas.draw()
            break

        i-=1
        if i <= 0:
            warnings.warn('Failed to achieve fixed width after {} iterations, saving with current width ({} pixels, but should be {} pixels).'.format(custom_settings['fixed_width_max_iter'], goal_width_pixels, real_width_pixels))
            break


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
            set_settings = {k: v for k, v in kwargs.items() if k in DEFAULT_SET_SETTINGS and k not in custom_settings and k not in ALLRCPARAMKEYS_REFORMATTED}
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
            if custom_settings['fig_paper'] is not None:
                rc_params = set_paper_defaults(
                    settings=rc_params,
                    paper=custom_settings['fig_paper'], 
                    aspect=custom_settings['fig_aspect'], 
                    subtype=custom_settings['fig_subtype'],
                )
            else: # if "default" scheme
                custom_settings['fixed_width'] = False # no fixed width bbox adjustment
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
            if kwargs.get('xlog', None) is not None:
                set_settings['xscale'] = 'log' if kwargs['xlog'] else 'linear'
                custom_settings.pop("xlog", None)
            if kwargs.get('ylog', None) is not None:
                set_settings['yscale'] = 'log' if kwargs['ylog'] else 'linear'
                custom_settings.pop("ylog", None)
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
                
                ## custom settings
                if custom_settings['show_legend']:
                    ax.legend()

                if custom_settings['show_colorbar']:
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
                is_done = custom_settings['out'] is not None or custom_settings['show']

                if custom_settings['out'] is not None:
                    if custom_settings['fixed_width']:  # custom iteration to fix the resulting width of saved figure and have tight bbox
                        #import time #DEBUG
                        #t= time.time() #DEBUG
                        #print('new output: {}'.format(custom_settings['out'])) # DEBUG
                        iterate_height_to_get_fixed_width(fig, max_iter=custom_settings['fixed_width_max_iter'], fig_aspect=custom_settings['fig_aspect'], fallback_aspect=1.0)
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


@basic_figure_wrapper(show_legend=True, fig_aspect='equal')
def plot(x, y, ax=None, label=None, color=None, ls='-', lw=None, **kwargs):
    """General 1D plotting function for lines. Use a tuples of parameters to plot multiple lines."""
    if not isinstance(y, tuple):  # for now base it on y, but could also be based on x or other parameters, just need to make sure they are all the same length TODO
        y = (y,)
    if not isinstance(x, tuple):
        x = (x,) * len(y)
    if not isinstance(label, tuple):
        label = (label,) * len(y)
    if not isinstance(color, tuple):
        color = (color,) * len(y)
    if not isinstance(ls, tuple):
        ls = (ls,) * len(y)
    if not isinstance(lw, tuple):
        lw = (lw,) * len(y)
    rets = []
    for i in range(len(y)):
        rets.append(ax.plot(x[i], y[i], label=label[i], color=color[i], ls=ls[i], lw=lw[i], **kwargs))
    if len(y) == 1:
        return rets[0]
    return rets

@basic_figure_wrapper(fig_aspect='equal')
def scatter(x, y, ax=None, **kwargs): 
    """General 2D plotting function for images."""
    return ax.scatter(x, y, **kwargs)

@basic_figure_wrapper(show_colorbar=True, fig_aspect='equal', no_minor_ticks=True)
def imshow(X, ax=None, colorbar=True, **kwargs): 
    """General 2D plotting function for images."""
    return ax.imshow(X, **kwargs)







#############################################################################
### ... MULTIPANEL FIGURES ... NOT YET COMPLETE

class Figure():
    """Class for creating and managing plots in a convenient way that avoids repetitive code."""
    # DEFAULT SETTINGS

    def __init__(self, nrows=1, ncols=1, fig=None, **settings):
        # check for invalid input
        if nrows < 1 or ncols < 1 or not isinstance(nrows, int) or not isinstance(ncols, int):
            raise ValueError('nrows and ncols must be positive integers.')
        if fig is not None and (nrows != 1 or ncols != 1):
            raise ValueError('If fig is inputted, you cannot specify the nrows and ncols, they will be inferred.')
        if fig is not None and any(setting in settings for setting in self._default_figure_creation_settings_list):
            raise ValueError('If fig is inputted, you cannot specify any figure creation settings.')
        if any(setting not in self.default_figure_settings or setting not in self.default_subplot_settings for setting in settings):
            raise ValueError('Invalid figure or subplot setting: {}'.format(settings))

        # fill in other figure settings that were not passed in
        for figure_setting in self.default_figure_settings:
            settings.setdefault(figure_setting, self.default_figure_settings[figure_setting])
        for subplot_setting in self.default_subplot_settings:
            settings.setdefault(subplot_setting, self.default_subplot_settings[subplot_setting])

        # has figure already?
        if fig is not None:
            # find more info about this figure and its axes
            axes = fig.get_axes()
            spec = axes[0].get_subplotspec()
            for ax in axes:
                if ax.get_subplotspec() != spec:
                    raise NotImplementedError('Currently, if fig is inputted, it should only have one spec, i.e. all axes should be in the same grid layout.')
            axes = sorted(axes, key=lambda ax: (ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)) # sort axes by their position in the grid
            nrows, ncols, *_ = spec.get_geometry()
        else:
            # make a figure and axes with the specified number of rows and columns
            fig = MatplotlibFigure(**{k: v for k, v in settings.items() if k in self._default_figure_creation_settings_list})
            spec = fig.add_gridspec(nrows, ncols)
            axes = [None for _ in range(nrows * ncols)]

        self.fig = fig
        self.axes = axes
        self.spec = spec
        self.nrows = nrows
        self.ncols = ncols
        self.settings = settings
        return self
    
    def get_axis_index(self, *input):
        """
        Three ways to input:
           1) idx       - index of the subplot in reading order (e.g. idx=0 for the top left subplot in a 2x2 grid)
           2) row, col  - with row, col as integers (e.g. row=0, col=0 for the top left subplot in a 2x2 grid)
           2) row, col  - with row, col as slices (e.g. row=0, col=slice(0, 2) for the top row of a 2x2 grid) 
        """
        if len(input) == 1 and isinstance(input[0], int):
            idx = input[0]
            if idx < 0 or idx >= self.nrows * self.ncols:
                raise ValueError('Index should be between 0 (inclusive) and nrows*ncols (exclusive).')
        elif len(input) == 2 and all(isinstance(i, int) for i in input):
            row, col = input
            if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
                raise ValueError('Row should be between 0 (inclusive) and nrows (exclusive), and col should be between 0 (inclusive) and ncols (exclusive).')
            if len(self.axes) == self.nrows * self.ncols:
                idx = row * self.ncols + col
            else:
                idx = 0
                visited_axes = set()
                for r in range(self.nrows):
                    for c in range(self.ncols):
                        if self.axes[r * self.ncols + c] not in visited_axes:
                            visited_axes.add((r, c))
                        if r == row and c == col:
                            break
                        idx += 1
                idx = row * self.ncols + col
        elif len(input) == 2 and all(isinstance(i, slice) for i in input):
            row, col = input
            if row.start < 0 or row.stop > self.nrows or col.start < 0 or col.stop > self.ncols:
                raise ValueError('Row slice should be between 0 (inclusive) and nrows (inclusive), and col slice should be between 0 (inclusive) and ncols (inclusive).')
            idx = []
            for r in range(row.start, row.stop):
                for c in range(col.start, col.stop):
                    idx.append(r * self.ncols + c)
        else:
            raise ValueError('Invalid axis index input. Should be either an integer (0 to nsubplots in reading order), or row and col as integers, or row and col as slices.')
        return idx

    def add_axis(self, row, col=None):
        pass

    def __getitem__(self, key):
        """Get the axis at the specified index, row and column, or row and column slice.""" # TODO: return object that inherits from the axis object but defines default settings for ploting functions (e.g. plot, scatter, imshow, hist) 
        idx = self.get_axis_index(key)
        if isinstance(idx, list):
            return [self.axes[i] for i in idx]
        return self.axes[idx]

    def __del__(self):
        plt.close(self.fig)
        del self.specs
        del self.axes
        del self.fig

    def draw(self, **settings):
        """Call this method to apply the figure settings and draw the figure."""
        with matplotlib.rc_context(self.RC_CONTEXT_SETTINGS):
            
            for ax in self.axes:
                # subplot specific stuff: 
                pass

            
            if settings['out'] is not None:
                self.fig.savefig(settings['out'], dpi=settings['dpi'])
                if settings['clf_after'] is None:
                    settings['clf_after'] = True
            if settings['show']:
                self.fig.show()
                if settings['clf_after'] is None:
                    settings['clf_after'] = True
            if settings['clf_after'] is None:
                settings['clf_after'] = False
            if settings['clf_after']:
                plt.clf()
        return self
    

    def plot(self, x, y, ax=None, **plot_settings):
        raise NotImplementedError()
    
    def imshow(self, X, ax=None, **imshow_settings):
        raise NotImplementedError()
    
    def scatter(self, x, y, ax=None, **scatter_settings):
        raise NotImplementedError()
    
    def hist(self, x, ax=None, **hist_settings):
        raise NotImplementedError()


# Example Usage:
# fig = Figure(figsize=(10, 8), grid=(2, 2))
# fig.plot(x, y, label='line 1') # defaults to first axis (ax=(0, 0) here since its a 2x2 grid)
# fig.plot(x, y2, label='line 2')
# fig.scatter(x, y2, ax=(0, 1), label='scatter plot 1')
# fig.hist(z, ax=(1, 0), label='histogram 1')
# fig.imshow(X, ax=(1, 1), cmap='viridis')


# each have a method outside that creates figure and applies any extra default settings for each regime:
# plot 
# scatter
# imshow
# hist

# extra convience functions that apply some common settings for specific scenarios and possibly some extra functionality:
# spectrum
# image
# phase_diagram
# mollweide_map
# ...






# start with default settings including all possible settings
# fig = Figure(3, 3, **settings) # self.settings.update(settings), make figure object
# fig.plot(x, y, **settings) # plot to all axes, apply settings passed
# fig.imshow(x, y, **settings) # imshow to all axes, apply settings passed
# ...
# fig[0, 1].plot(x, y, **settings) # apply to (0, 1) row, col settings passed
# fig[1].plot(x, y, **settings) # apply to second subplot in reading order, apply settings passed 
# fig[0:2, 0:2].plot(x, y, **settings) # apply to first 2 rows and 2 cols settings passed

# jv.plot(x, y, ls='-', show=True) # works alone

# def plot(x, y, ncols=1, nrows=1, fig=None, **settings):
#     fig = Figure(ncols=ncols, nrows=nrows, fig=fig, **settings)
#     return fig.plot(x, y, **settings)