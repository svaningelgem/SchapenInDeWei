import random
from collections import namedtuple, defaultdict
from math import sqrt
from functools import lru_cache
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


@lru_cache(maxsize=None)
def distance(pt1, pt2):
    return sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2)


@lru_cache(maxsize=None)
def avg(a, b):
    return (a + b) / 2.0


def is_in_area(pt, fence):
    if pt in fence:
        return True

    import matplotlib.path as mplPath
    bbPath = mplPath.Path(fence)
    return bbPath.contains_point(pt)


def add_largest_distance_to_fence(fence, reverse_distances):
    distances = sorted(reverse_distances.keys())
    if len(distances) == 0:
        raise ValueError("No values anymore??")

    longest_distance = distances.pop()
    new_point_in_fence = reverse_distances[longest_distance][0]
    del reverse_distances[longest_distance][0]

    if len(reverse_distances[longest_distance]) == 0:
        del reverse_distances[longest_distance]

    # Sort the fence:
    add_point_to_fence(fence, new_point_in_fence)
    

@lru_cache(maxsize=None)
def orientation(p: Point, q: Point, r: Point) -> int:
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1  # clock
    else:
        return 2  # counterclock


# Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
@lru_cache(maxsize=None)
def on_segment(p: Point, q: Point, r: Point) -> bool:
    return q.x <= max(p.x, r.x) and \
        q.x >= min(p.x, r.x) and \
        q.y <= max(p.y, r.y) and \
        q.y >= min(p.y, r.y)


# Does line segment 'p1q1' and 'p2q2' intersect?
@lru_cache(maxsize=None)
def do_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False  # Doesn't fall in any of the above cases


@lru_cache(maxsize=None)
def is_intersecting(*args) -> bool:
    # print("is_intersecting:", args, end='')

    s = set(args)

    retVal = False
    if len(s) == 4:
        retVal = do_intersect(*args)

    # print(" -->", "YES" if retVal else "NO")

    return retVal


# This is needed otherwise our fence might get all mangled up!
def add_point_to_fence(fence, P):
    if len(fence) < 3:
        fence.append(P)
        return fence

    a = 1
    # Look where to place it in final
    # This position is i+1 for the first i verifying that neither [Ai-P] nor [Ai+1-P] intersects any other segments [Ak-Ak+1].
    for idx in range(1, len(fence)):
        # print(" --> Starting with index", idx)
        # Create a copy
        tmp = fence.copy()
        tmp.insert(idx, P)
        tmp.append(tmp[0])  # Add the first element in there again to have it completely check every segment

        t1 = list(is_intersecting(tmp[ix], tmp[ix + 1], tmp[x], tmp[x + 1]) for x in range(len(tmp) - 1) for ix in range(0, len(tmp) - 1))

#        if not any(is_intersecting(fence[idx], P, tmp[x], tmp[x + 1]) for x in range(len(tmp) - 1)):
        if not any(t1):
            # Do the change for real!
            fence.insert(idx, P)
            break
    else:
        fence.append(P)

    # print("-------------------------------")
    return fence


def enlarge_fence(fence, reverse_distances, coords, center):
    add_largest_distance_to_fence(fence, reverse_distances)

    # draw_result(coords, fence)
    # Now remove all coordinates that are WITHIN the fence
    to_remove = list()
    for pt in coords:
        if pt in fence or is_in_area(pt, fence):
            to_remove.append(pt)  # Can't modify the list during the loop

    for pt in to_remove:
        # Remove the coordinate
        coords.remove(pt)
        # calculate the distance
        d = distance(pt, center)
        # Remove it from the reverse_distances
        if pt in reverse_distances[d]:
            reverse_distances[d].remove(pt)

        # and if this is empty -> Clean out the entry
        if len(reverse_distances[d]) == 0:
            del reverse_distances[d]

    # draw_result(coords, fence, center)


def draw_result(coords, fence, center, filename=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()
    # Draw the center
    ax.scatter(
        [center.x],
        [center.y],
        color='red'
    )

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


# Try to remove coordinates and see if the result is still ok
def optimize_fence(fence, coords, center):
    while len(fence) > 3:
        # draw_result(coords, fence, center)

        start_len = len(fence)
        idx = 0
        while idx < len(fence):
            test = fence.copy()
            del test[idx]

            # draw_result(coords, test, center)

            tmp = list(is_in_area(x, test) for x in coords)
            if all(tmp):
                del fence[idx]
            else:
                idx += 1

        stop_len = len(fence)
        if stop_len == start_len:
            break

    # draw_result(coords, fence, center)


def solve_issue(coords):
    # Find min X/Y --> This is a possibility, but unlikely to be the smallest!
    min_x = min(map(lambda pt: pt.x, coords))
    min_y = min(map(lambda pt: pt.y, coords))
    max_x = max(map(lambda pt: pt.x, coords))
    max_y = max(map(lambda pt: pt.y, coords))

    # Now... Can we eliminate stuff??
    # -> Find the center point.
    center = Point(x=avg(min_x, max_x), y=avg(min_y, max_y))
    # -> Calculate for each point the distance to the center.
    # -> And reverse this so we have a dict(distance -> point-list)
    reverse_distances = defaultdict(list)
    for pt, dist in ((pt, distance(pt, center)) for pt in coords):
        reverse_distances[dist].append(pt)

    fence = list()
    while len(fence) < 2:  # the enlarge_fence function will add yet another point there!
        add_largest_distance_to_fence(fence, reverse_distances)

    while len(coords):
        enlarge_fence(fence, reverse_distances, coords, center)

    return fence, center


if __name__ == "__main__":
    with open("last_seed.dat", "wb") as fp:
        pickle.dump(random.getstate(), fp)
    
    coords = generate_coordinates()
    fence, center = solve_issue(coords.copy())
    draw_result(coords, fence, center, "before_optimization.png")
    optimize_fence(fence, coords, center)
    draw_result(coords, fence, center, "final_result.png")
