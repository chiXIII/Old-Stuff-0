import math
import pandas as pd
import numpy as np
import itertools

class Counter:
    def __init__(self):
        self.n = 0
    def plus(self):
        self.n += 1
        return self.n
    def reset(self):
        self.n = 0
    def number(self):
        return self.n
    def __str__(self):
        return 'Count: {}'.format(str(self.number()))

def p(x):
    print(x)
    return x

def angle(point0, point1):
    """
    return the angle (counterclockwise) from
    point0 to point1, both tuples or lists with two coordinates.
    """
    angle = math.atan((point0[1] - point1[1])/(point0[0] - point1[0]))
    if point1[0] < point0[0]:
        angle += math.pi
    return angle

def distance(point0, point1):
    """
    Returns the Euclidean distance between point0 and point1,
    lists or tuples with two coordinates.
    """
    if type(point0) == type(point1) == list or type(point0) == type(point1) == tuple:
        x0, y0, x1, y1 = point0[0], point0[1], point1[0], point1[1]
    if type(point0) == type(point1) == pd.core.series.Series:
        x0, y0, x1, y1 = point0.iloc[0]['x'], point0.iloc[0]['y'], point1.iloc[0]['x'], point1.iloc[0]['y']
    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def lb(grams):
    return 0.00220462*grams

def ft_lb(jouls):
    return 0.737562*jouls
