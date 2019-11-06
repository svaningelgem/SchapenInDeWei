import random
from collections import namedtuple
import os
import pickle


if os.path.exists("use_seed.dat"):
    with open("use_seed.dat", "rb") as fp:
        random.setstate(pickle.load(fp))

Point = namedtuple("Point", ('x', 'y'))
picture_counter = 0

# ################ CONFIG
min_x = 0
max_x = 100
min_y = 0
max_y = 100
# #######################


def generate_coordinates(n: int = 100):
    return list((
        Point(x=random.randint(min_x, max_x), y=random.randint(min_y, max_y))
        for _ in range(n)
    ))


# https://github.com/samfoy/GrahamScan/blob/master/graham_scan.py
def sort_points(point_array):
    """Return point_array sorted by leftmost first, then by slope, ascending."""

    def slope(y):
        """returns the slope of the 2 points."""
        x = point_array[0]
        return (x.y - y.y) / \
               (x.x - y.x)

    point_array.sort()  # put leftmost first
    point_array = point_array[:1] + sorted(point_array[1:], key=slope)
    return point_array


def graham_scan(point_array):
    """Takes an array of points to be scanned.
    Returns an array of points that make up the convex hull surrounding the points passed in in point_array.
    """

    def cross_product_orientation(a, b, c):
        """Returns the orientation of the set of points.
        >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
        """

        return (b.y - a.y) * \
                (c.x - a.x) - \
                (b.x - a.x) * \
                (c.y - a.y)

    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    sorted_points = sort_points(point_array)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and cross_product_orientation(convex_hull[-2], convex_hull[-1], p) >= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return convex_hull


def draw_result(coords, fence, filename=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()
    # Draw the coordinates
    ax.scatter(
        list(map(lambda pt: pt.x, coords)),
        list(map(lambda pt: pt.y, coords)),
        color='blue'
    )
    # Draw the fence
    line = Line2D(
        list(map(lambda pt: pt.x, fence)),
        list(map(lambda pt: pt.y, fence)),
        color='green'
    )
    ax.add_line(line)
    # Draw the closing of the fence
    line = Line2D(
        [fence[0].x, fence[-1].x],
        [fence[0].y, fence[-1].y],
        color='green'
    )
    ax.add_line(line)
    # plt.show()

    axes = plt.gca()
    axes.set_xlim([min_x, max_x])
    axes.set_ylim([min_y, max_y])

    if filename:
        plt.savefig(filename)
    else:
        global picture_counter
        picture_counter += 1
        plt.savefig(os.path.dirname(__file__) + "/step_" + str(picture_counter) + ".png")

    plt.close()


if __name__ == "__main__":
    with open("last_seed.dat", "wb") as fp:
        pickle.dump(random.getstate(), fp)

    coords = generate_coordinates()
    fence = graham_scan(coords)
    draw_result(coords, fence, "final_result.png")
