import numpy as np
import math
import utility
import matplotlib.pyplot as plt
import pandas as pd


            ####
       ####       ####
     ####  []       #####
    #####           ######
    #####           ######
     #####         ######
       ####       ####
            ####


class Bubble:
    def __init__(self, members_x = None, members_y = None, file = None, mesh = 0.1, kaze = [0,0]):
        self.mesh = mesh
        self.x = members_x
        self.y = members_y
        self.kaze = kaze

        if file:
            df = pd.read_csv(file)
            self.x = list(df.iloc[:,0])
            self.y = list(df.iloc[:,1])

        self.that = np.array([[0.,0.] for n in range(self.members)])
        self.tangents()
        self.v0rtices = np.array([[0., 0., 0.,] for n in range(self.members)])
        self.set_v0rtices()

    @property
    def members(self):
        return len(self.x)

    @property
    def centroid(self):
        x = sum(self.x)/self.members
        y = sum(self.y)/self.members
        return x,y

    def tangents(self):
        dx = self.x[0] - self.x[::-1][0]
        dy = self.y[0] - self.y[::-1][0]
        d = math.sqrt(dx**2+dy**2)
        ux, uy = dx/d, dy/d
        u0x, u0y = ux, uy
        for n in range(self.members-1):
            dx = self.x[n+1] - self.x[n]
            dy = self.y[n+1] - self.y[n]
            d = math.sqrt(dx**2+dy**2)
            vx, vy = dx/d, dy/d
            mag = math.sqrt((ux+vx)**2 + (uy+vy)**2)
            self.that[n,0], self.that[n,1] = (vx+ux)/mag, (vy +uy)/mag
            ux, uy = vx, vy
        mag = math.sqrt((ux+u0x)**2 + (uy+u0y)**2)
        self.that[self.members-1,0], self.that[self.members-1,1] = (ux + u0x)/mag, (uy + u0y)/mag

    def set_v0rtices(self):
        cx, cy = self.centroid
        for n in range(self.members-1):
           self.v0rtices[n] = [self.x[n]*0.5 + self.x[n+1]*0.5, self.y[n]*0.5 + self.y[n+1]*0.5, 0]
        self.v0rtices[n+1] = [self.x[n+1]*0.5 + self.x[0]*0.5, self.y[n+1]*0.5 + self.y[0]*0.5, 0]

    def set_kaze(self, x,y):
        self.kaze = [x,y]

    def tachinu(self):
        x, y = self.kaze[0], self.kaze[1]
        equations, totals = np.array([[0. for m in range(self.members)] for m in range(self.members)]), np.array([[0.] for m in range(self.members)])
        for n in range(self.members):
            for i in range(self.members):
                dx, dy = self.x[n] - self.v0rtices[i,0], self.y[n] - self.v0rtices[i,1]
                d2 = dx**2 + dy**2
                equations[n,i] = (dx*self.that[n,0]+dy*self.that[n,1])/d2
                totals[n,0] = x*self.that[n,1] - y*self.that[n,0]
        soln = np.linalg.solve(equations, totals)
        circ = 0
        print('###############################\nvalues\n###############################\n[')
        for n in range(self.members):
            self.v0rtices[n,2] = soln[n,0]
            circ += soln[n,0]
        print(']\n\n\ncirculation:\n')
        print(circ)
        return soln

    def plot_kaze(self):
        x,y = self.kaze[0], self.kaze[1]
        def kaze(X,Y):
            kaze_x, kaze_y = x,y
            for n in range(self.members):
                dx, dy = X-self.v0rtices[n,0], Y- self.v0rtices[n,1]
                d2 = dx**2 + dy**2
                kaze_x -= dy/d2*self.v0rtices[n,2]
                kaze_y += dx/d2*self.v0rtices[n,2]
            plt.plot([X,X+kaze_x/10], [Y,Y+kaze_y/10], color = 'green')
        for n in range(self.members):
            kaze(self.x[n],self.y[n])
        for m in range(-10,20):
            for n in range(-10,10):
                kaze(m/10,n/10)
        plt.scatter(self.x, self.y, color = 'black')
        plt.axis('square')

    def plot(self):
        plt.scatter(self.x, self.y, color = 'black')
        for n in range(self.members):
            plt.plot([self.x[n], self.x[n]+self.that[n,0]], [self.y[n], self.y[n]+self.that[n,1]], color = 'red')
            plt.scatter([self.v0rtices[n,0]],[self.v0rtices[n,1]], color = 'orange')
        plt.axis('square')

    def split(self, number):
        x,y = [],[]
        for n in range(self.members):
            print('foobar')

    def linear_transform(self, a11, a12, a21, a22):
        x,y = [self.x[n]*a11 + self.y[n]*a12 for n in range(self.members)], [self.x[n]*a21 + self.y[n]*a22 for n in range(self.members)]
        return Bubble(x,y)

    def rotate(self, theta):
        c, s = math.cos(theta), math.sin(theta)
        return self.linear_transform(c,-s, s,c)

    def multiply(self, integ):
        x,y = [],[]
        for n in range(self.members-1):
            for i in range(integ):
                x.append(self.x[n]*(integ-i)/integ + self.x[n+1]*(i)/integ)
                y.append(self.y[n]*(integ-i)/integ + self.y[n+1]*(i)/integ)
        n += 1
        for i in range(integ):
            x.append(self.x[n]*(integ-i)/integ + self.x[0]*(i)/integ)
            y.append(self.y[n]*(integ-i)/integ + self.y[0]*(i)/integ)
        return Bubble(x,y)
