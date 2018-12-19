from matplotlib import pyplot
from descartes.patch import PolygonPatch


def multipolygons(multipolygon):
    fig = pyplot.figure(1, dpi=90)
    ax = fig.add_subplot(121)

    for polygon in multipolygon:
        patch = PolygonPatch(polygon, facecolor='#6699cc', edgecolor='#ffffff', alpha=0.5, zorder=2)
        ax.add_patch(patch)

    pyplot.show()
