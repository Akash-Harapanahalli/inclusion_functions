import numpy as np
import matplotlib.pyplot as plt

f1 = lambda x : x**2/2
f2 = lambda x : x**3/3

def cone_f1 (ax, x, xx, color) :
    ax.plot(xx, (xx - x)+f1(x), lw=2, color=color, linestyle='dashed')
    ax.plot(xx, (x - xx)+f1(x), lw=2, color=color, linestyle='dashed')

def cone_f2 (ax, x, xx, color) :
    ax.plot(xx, (xx - x)+f2(x), lw=2, color=color, linestyle='dashed')
    ax.plot(xx, ones*f2(x), lw=2, color=color, linestyle='dashed')

def draw_interval (ax, _y, y_) :
    ax.plot([-1.6,-1.6], [_y, y_], lw=2, color='black')
    ax.plot([-1.7,-1.5], [_y, _y], lw=2, color='black')
    ax.plot([-1.7,-1.5], [y_, y_], lw=2, color='black')

owindow_xx = np.linspace(-2,2,10000)
owindow_yy = np.linspace(-2,2,10000)
zeros = np.zeros_like(owindow_xx)
ones = np.ones_like(owindow_xx)

xx = np.linspace(-1.2,1.2,10000)

fig, axs = plt.subplots(2,3,figsize=[12,7],dpi=100)
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.94,
                    wspace=0.01,hspace=0.01)

titles = ['Minimal', 'Centered', 'Cornered']

for axi, ax in enumerate(axs[0,:].flatten()) :
    ff = f1(xx); ax.plot(xx, ff, lw=4, color='black')
    window_xx = owindow_xx
    window_yy = np.linspace(-2.2,3.2,10000)
    # Axes
    ax.plot(window_xx, zeros, lw=1, color='black')
    ax.plot(zeros, window_yy, lw=1, color='black')
    ax.plot((-1)*ones, window_yy, lw=2, color='black', linestyle='dashed')
    ax.plot(ones, window_yy, lw=2, color='black', linestyle='dashed')
    ax.axis('off')
    ax.set_title(titles[axi], fontsize=20)
    if axi == 0 :
        ax.plot(xx, ones*f1(-1), lw=2, color='tab:blue', linestyle='dashed')
        ax.plot(xx, zeros, lw=2, color='tab:blue', linestyle='dashed')
        draw_interval(ax, 0, 1/2)
    elif axi == 1 :
        cone_f1(ax, 0, xx, 'tab:blue')
        draw_interval(ax, -1, 1)
    elif axi == 2 :
        cone_f1(ax, -1, xx, 'tab:blue')
        cone_f1(ax, 1, xx, 'tab:red')
        draw_interval(ax, -1.5, 2.5)
        
for axi, ax in enumerate(axs[1,:].flatten()) :
    ff = f2(xx); ax.plot(xx, ff, lw=4, color='black')
    window_xx = owindow_xx
    window_yy = owindow_yy
    # Axes
    ax.plot(window_xx, zeros, lw=1, color='black')
    ax.plot(zeros, window_yy, lw=1, color='black')
    ax.plot((-1)*ones, window_yy, lw=2, color='black', linestyle='dashed')
    ax.plot(ones, window_yy, lw=2, color='black', linestyle='dashed')
    ax.axis('off')
    if axi == 0 :
        ax.plot(xx, ones*f2(-1), lw=2, color='tab:blue', linestyle='dashed')
        ax.plot(xx, ones*f2(1), lw=2, color='tab:blue', linestyle='dashed')
        draw_interval(ax, -1/3, 1/3)
    elif axi == 1 :
        cone_f2(ax, 0, xx, 'tab:blue')
        draw_interval(ax, -1, 1)
    elif axi == 2 :
        cone_f2(ax, -1, xx, 'tab:blue')
        cone_f2(ax, 1, xx, 'tab:red')
        draw_interval(ax, -1/3, 1/3)


fig.savefig('incl_compared.pdf')
plt.show()
