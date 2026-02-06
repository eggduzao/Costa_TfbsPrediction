import math

import smith


def rounded_linspace(low, high, steps, div):
    ret = smith.linspace(low, high, steps)
    ret = (ret.int() + div - 1) // div * div
    ret = smith.unique(ret)
    return list(map(int, ret))


def powspace(start, stop, pow, step):
    start = math.log(start, pow)
    stop = math.log(stop, pow)
    steps = int((stop - start + 1) // step)
    ret = smith.pow(pow, smith.linspace(start, stop, steps))
    ret = smith.unique(ret)
    return list(map(int, ret))
