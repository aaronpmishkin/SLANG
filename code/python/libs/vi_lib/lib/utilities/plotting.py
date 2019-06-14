import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

'''
Assorted helper functions for nice plotting
'''

def forceAspect(ax, aspect=1):
    r"""Forces an axis to an aspect ratio (width/height)."""
    scale_str = ax.get_yaxis().get_scale()
    
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()

    if scale_str=='linear':
        asp = abs((xmax-xmin)/(ymax-ymin))/aspect
    elif scale_str=='log':
        asp = abs((np.log(xmax)-np.log(xmin))/(np.log(ymax)-np.log(ymin)))/aspect

    ax.set_aspect(asp)

def remove(axes, elements):
    for elem in elements:
        assert elem in ["xlabel", "ylabel", "xticks", "yticks"]
        
    for elem in elements:
        for ax in axes:
            if elem is "xlabel":
                ax.set_xlabel("")
            if elem is "ylabel":
                ax.set_ylabel("")
            if elem is "xticks":
                ax.set_xticks([])
            if elem is "yticks":
                ax.set_yticks([])
    
def sync_lims(axes, lim_name):
    r"""
    Make all axes have the same min/max in the desired limits.
    :code:`lim_name` can be `xlims`, `ylims`, `clims`.
    """
    assert lim_name in ['x', 'y', 'c']
    
    def get_lims(ax):
        if lim_name is 'x':
            return ax.get_xlim()
        if lim_name is 'y':
            return ax.get_ylim()
        if lim_name is 'c':
            return ax.images[0].get_clim()

    def set_lim(ax, low, hig):
        if lim_name is 'x':
            return ax.set_xlim(low, hig)
        if lim_name is 'y':
            return ax.set_ylim(low, hig)
        if lim_name is 'c':
            return ax.images[0].set_clim(low, hig)

    lims = np.zeros((len(axes), 2))
    for i in range(len(axes)):
        current_lims = np.array(get_lims(axes[i]))
        lims[i, :] = current_lims
    low, hig = np.min(lims), np.max(lims)
    for ax in axes:
        set_lim(ax, low, hig)

    return low, hig

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Source: https://stackoverflow.com/a/20528097/1983613
    
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap

def centeredSymmetricColorMap(cmap, low, high, name="centeredsymmetricsmap2"):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    
    assert(high > 0)
    assert(low < 0)

    if abs(low) > high:
        start=0
        stop=0.5 + high/(2*abs(low))
    else:
        start=0.5 + low/(2*abs(high))
        stop=1

    reg_index = np.linspace(start, stop, 257)
    shift_index = np.linspace(0.0, 1.0, 129+128, endpoint=True)
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap

    
def centerColorMap(axes, cmap):
    low, hig = sync_lims(axes, 'c')
    shrunk_cmap = centeredSymmetricColorMap(cmap, low, hig, name='shrunk')
    for ax in axes:
        ax.get_images()[0].set_cmap(shrunk_cmap)
    return low, hig

def plotPerfectCorrelation(ax, linespec='k--'):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    newlims = [min(min(xlims), min(ylims)), max(max(xlims), max(ylims))]
    ax.plot(newlims, newlims, linespec)
    ax.set_xlim(newlims)
    ax.set_ylim(newlims)

def hide_label_and_tickmark(axis):
    axis.set_ticklabels([])
    axis.set_tick_params(length=0)

def shaded_cov(ax, x, mean, std, **kwargs):
    low = mean - std/np.sqrt(10)
    high = mean + std/np.sqrt(10)
    ax.fill_between(x, low, high, **kwargs)

