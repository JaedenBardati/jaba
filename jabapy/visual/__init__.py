import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')                    # non-interactive mode default


def basic_figure_wrapper(plotting_function):
    """General wrapper that encapulates most of the repeated parts when plotting."""
    def wrapper_function(*args, 
                         fig=None, ax=None, figsize=(6.4, 4.8), 
                         xlog=False, ylog=False, xlabel=None, ylabel=None, title=None,
                         xmin=None, xmax=None, ymin=None, ymax=None,
                         show_legend=False, legend_frameon=False, legend_loc=0, 
                         tight_layout=True, clf_before=False, clf_after=None, out=None, show=False, dpi=300, 
                         fontsize=10, axes_linewidth=1.25, legend_fontsize=None,
                         major_ticks_on=None, minor_ticks_on=None, tick_direction='in',
                         major_ticksize=8, minor_ticksize=4, major_tickwidth=1.5, minor_tickwidth=1.5,
                         xtick_bottom=True, xtick_top=True, ytick_left=True, ytick_right=True, 
                         **kwargs):
        if (fig is None) != (ax is None):
            raise ValueError('If fig or ax is inputted, the other should be also.')
        if clf_before:
            plt.clf()
        
        with matplotlib.rc_context({
            'xtick.direction': tick_direction,
            'ytick.direction': tick_direction,
            'font.size': fontsize,
            'axes.linewidth': axes_linewidth,
            'xtick.major.size': major_ticksize,
            'ytick.major.size': major_ticksize,
            'xtick.minor.size': minor_ticksize,
            'ytick.minor.size': minor_ticksize,
            'xtick.major.width': major_tickwidth,
            'ytick.major.width': major_tickwidth,
            'xtick.minor.width': minor_tickwidth,
            'ytick.minor.width': minor_tickwidth,
            'xtick.bottom': xtick_bottom,
            'xtick.top': xtick_top,
            'ytick.left': ytick_left,
            'ytick.right': ytick_right,
        }):
            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            
            plotting_function(*args, ax=ax, **kwargs)
            
            if title is not None:
                ax.set_title(title)
            if minor_ticks_on is not None:
                if minor_ticks_on:
                    ax.minorticks_on()
                else:
                    ax.minorticks_off()
            if major_ticks_on is not None:
                if major_ticks_on:
                    ax.majorticks_on()
                else:
                    ax.majorticks_off()
            if xmin is not None or xmax is not None:
                ax.set_xlim([xmin, xmax])
            if ymin is not None or ymax is not None:
                ax.set_xlim([ymin, ymax])
            if xlog:
                ax.set_xscale('log')
            if ylog:
                ax.set_yscale('log')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if tight_layout:
                fig.tight_layout()
            if show_legend:
                ax.legend(frameon=legend_frameon, loc=legend_loc, fontsize=legend_fontsize if legend_fontsize is not None else fontsize)
            if out is not None:
                fig.savefig(out, dpi=dpi)
                if clf_after is None:
                    clf_after = True
            if show:
                plt.show()
                if clf_after is None:
                    clf_after = True
            if clf_after is None:
                clf_after = False
            if clf_after:
                plt.clf()
            return fig, ax
        
    return wrapper_function

@basic_figure_wrapper
def plot1Dline(x, y, label=None, color=None, ls='-', lw=None, ax=None):
    """General 1D plotting function for lines. Use a tuple for y to plot multiple lines."""
    if not isinstance(y, tuple):
        y = (y,)
    if not isinstance(label, tuple):
        label = (label,)
    if not isinstance(color, tuple):
        color = (color,)
    if not isinstance(ls, tuple):
        ls = (ls,)
    if not isinstance(lw, tuple):
        lw = (lw,)
    for i in range(len(y)):
        ax.plot(x, y[i], label=label[i], color=color[i], ls=ls[i], lw=lw[i])

