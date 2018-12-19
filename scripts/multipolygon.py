from matplotlib import pyplot
from shapely.geometry import MultiPolygon
from descartes.patch import PolygonPatch


COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
    
fig = pyplot.figure(1, dpi=90)

# 1: valid multi-polygon
ax = fig.add_subplot(121)

a = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
b = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]

multi1 = MultiPolygon([[a, []], [b, []]])

for polygon in multi1:
    plot_coords(ax, polygon.exterior)
    patch = PolygonPatch(polygon, facecolor=v_color(multi1), edgecolor=v_color(multi1), alpha=0.5, zorder=2)
    ax.add_patch(patch)

ax.set_title('a) valid')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_aspect(1)



pyplot.show()

