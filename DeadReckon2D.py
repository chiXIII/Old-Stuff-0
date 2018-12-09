import utility
from utility import *
import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt, tan
import math
import pandas as pd
import numpy as np


class Curve:

    def __init__(self, r = 1, value = 0):
        self.state = pd.DataFrame({'x':[0], 'y':[0], 'theta':[0], 'v':[1], 'r':[r], 't':[0], 'value':[value]})

    def plus(self, dt, value = 0):
        state = self.state.iloc[::-1].iloc[0]
        t = state['t'] + dt
        theta = state['theta'] + dt*state['v']/state['r']
        x = state['x'] + dt * state['v'] * cos(state['theta'])
        y = state['y'] + dt * state['v'] * sin(state['theta'])
        v = state['v']
        r = state['r']
        self.state = pd.concat([self.state, pd.DataFrame({'x':[x], 'y':[y], 'theta':[theta], 'v':[v], 'r':[r], 't':[t], 'value':[value]})])

    def follow(self, c, n, getva = lambda x: x['v'], overwrite = []):
        ret = []
        copystate = c.state.copy()
        x0 , y0, r0, v0, theta0, t0 = 1,1,1,1,1,1
        for i in range(len(c.state)):
            sstate = self.state.iloc[::-1].iloc[0]
            state = copystate.iloc[i]
            x = state['x'] + n(state)*cos(state['theta'] + pi/2)
            y = state['y'] + n(state)*sin(state['theta'] + pi/2)
            if not i%10:
                ret.append([[state['x'].mean(), x], [state['y'].mean(), y]])
            value = getva(state)
            t = state['t']
            theta = angle((x0, y0), (x,y))
            v = sqrt((x - x0)**2 + (y - y0)**2)/(t - t0)
            r = None
            x0, y0, v0, r0, t0, theta0 = x, y, v, r, t, theta
            self.state = pd.concat([self.state, pd.DataFrame({'x':[x], 'y':[y], 'theta':[theta], 'v':[v], 'r':[r], 't':[t], 'value':[value]})])
        self.state = self.state.sort_values('t')
        self.state = self.state.iloc[1:] # delete starting point, all the parameters of which are 1

        for l in overwrite:
            old = self.state.loc[self.state['t'] > l[0]]
            old = old.loc[old['t'] < l[1]]
            v0 = old.iloc[0][l[2]]
            dv = (old.iloc[::-1].iloc[0][l[2]] - v0)/len(old)
            for i in range(len(old)):
                self.state.loc[self.state['t'] == old.iloc[i]['t'], l[2]] = v0 + dv*i

        for i in range(1,len(self.state)-1):
            row0 = self.state.iloc[i]
            row1 = self.state.iloc[i+1]
            r = row0['v']*(row1['t'] - row0['t'])/(2*math.pi*(row1['theta'] - row0['theta']))
            self.state.iloc[i ]['r'] = r

        """
        self.smooth(width = 1, column = 'theta')
        self.smooth(width = 1, column = 'theta', lopsided = True)
        self.smooth(width = 1, column = 'theta')
        self.smooth(width = 1, column = 'theta', lopsided = True)
        self.smooth(width = 1, column = 'theta')
        self.smooth(width = 1, column = 'theta', lopsided = True)
        self.smooth(width = 1, column = 'theta')
        self.smooth(width = 1, column = 'theta', lopsided = True)


        self.smooth(width = 1, column = 'r', asym = True)
        self.smooth(width = 1, column = 'r', lopsided = True, asym = True)
        self.smooth(width = 1, column = 'r', asym = True)
        self.smooth(width = 1, column = 'r', lopsided = True, asym = True)

        self.smooth(width = 1)
        self.smooth(width = 1, lopsided = True)
        self.smooth(width = 1)
        self.smooth(width = 1, lopsided = True)
        """

        self.state = self.state.iloc[1:] # \/ delete ends the r values of which are not set
        self.state = self.state.iloc[::-1].iloc[1:].iloc[::-1]
        return ret

    def smooth(self, width = 20, column = 'value', repeat = (True, True), asym = False, lopsided = False):
        self.state = self.state.dropna()
        sbt = int(lopsided)
        smooth_counter = utility.Counter()
        self.state = self.state.iloc[::-1]
        state = self.state.copy()
        for i in range(len(self.state)):
            points = [None,None]
            if math.copysign(1, state.iloc[i - width - sbt][column]) != math.copysign(1, state.iloc[i][column]) and asym: 
                points[0] = None
            else:
                if i-width-sbt >= 0:
                    points[0] = [state.iloc[i - width - sbt][column]]
                else:
                    points[0] = [2*state.iloc[0][column]-state.iloc[-i+width+sbt][column]]
            if i + width > len(state)-1:
                points[1] = [2*state.iloc[::-1].iloc[0][column]-state.iloc[-(i + width) + 2*(len(state) - 1)][column]]
            elif math.copysign(1, state.iloc[i + width][column]) != math.copysign(1, state.iloc[i][column]) and asym: 
                points[1] = None
            else:
                points[1] = [state.iloc[i + width][column]]

            smooth_counter.plus()
            try:
                if points[0] == None:
                    1+1
                elif points[1] == None:
                    1+1
                else:
                    self.state.iloc[i].loc[column] = np.mean(points)
            except:
                print(smooth_counter)
        if repeat[0]:
            if repeat[1]:
                self.smooth(width = width, column = column, repeat = (True,False), asym = asym)
            else:
                self.smooth(width = width, column = column, repeat = (False, False), asym = asym)

    def plot(self, axs, color = 'Black'):
        if type(axs) is list:
            axs[0].plot(self.state['x'], self.state['y'], color = color)
            axs[0].set_title('streamlines')
            axs[0].axis('equal')
            axs[1].plot(self.state['t'], self.state['r'])
            axs[1].set_title('radius')
            axs[2].plot(self.state['t'], self.state['value'])
            axs[2].set_title('velocity')
            axs[3].set_title('coordinates')
            axs[3].plot(self.state['t'], self.state['x'])
            axs[3].plot(self.state['t'], self.state['y'])
            axs[4].set_title('theta')
            axs[4].plot(self.state['t'], self.state['theta'])
        else:
            axs.plot(self.state['x'], self.state['y'], color = color)

