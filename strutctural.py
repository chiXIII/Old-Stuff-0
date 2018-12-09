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
    def __init__(self, members_x = None, members_y = None, file = None, mesh = 0.1, boundary_flow = 'tangential'):
        self.mesh = mesh
        self.x = members_x
        self.y = members_y
        self.boundary_flow = boundary_flow

        if file:
            df = pd.read_csv(file)
            self.x = list(df.iloc[:,0])
            self.y = list(df.iloc[:,1])

        self.that = np.array([[0.,0.] for n in range(self.members)])
        self.tangents()
        l = [[0., 0., 0.,] for n in range(self.members)]
        if type(self.boundary_flow) == np.ndarray:
            self.v0rtices = np.array(l*2+2*[[0.,0.,0.]])
        else:
            self.v0rtices = np.array(l+[[0.,0.,0.]])
        self.set_v0rtices()
        """
        self.plot()
        plt.show()
        """

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
            try:
                vx, vy = dx/d, dy/d
            except:
                plt.scatter([self.x[n]], [self.y[n]])
                plt.show()
            mag = math.sqrt((ux+vx)**2 + (uy+vy)**2)
            self.that[n,0], self.that[n,1] = (vx+ux)/mag, (vy +uy)/mag
            ux, uy = vx, vy
        mag = math.sqrt((ux+u0x)**2 + (uy+u0y)**2)
        self.that[self.members-1,0], self.that[self.members-1,1] = (ux + u0x)/mag, (uy + u0y)/mag

    def set_v0rtices(self):
        cx, cy = self.centroid
        for n in range(self.members):
           self.v0rtices[n] = [self.x[n]-self.that[n,1]*self.mesh/2 + 0.5*self.that[n,0]*self.mesh/2, self.y[n]+self.that[n,0]*self.mesh/2 + 0.5*self.that[n,1]*self.mesh/2, 0.]
        self.v0rtices[n+1] = [cx, cy, 0.]
        if type(self.boundary_flow) == np.ndarray:
            for i in range(self.members+1, 2*self.members+1):
                n = i-1-self.members
                self.v0rtices[i] = [self.x[n]-self.that[n,1]*self.mesh + 0.5*self.that[n,0]*self.mesh, self.y[n]+self.that[n,0]*self.mesh + 0.5*self.that[n,1]*self.mesh, 0.]
            self.v0rtices[2*self.members+1] = [cx,cy,0.]

    def plot(self):
        plt.scatter(self.x, self.y, color = 'black')
        for n in range(self.members):
            plt.plot([self.x[n], self.x[n]+self.that[n,0]], [self.y[n], self.y[n]+self.that[n,1]], color = 'red')
            if type(self.boundary_flow) == np.ndarray:
                """
                plt.plot([self.x[n], self.x[n] + self.boundary_flow[0,n,0]], [self.y[n], self.y[n] + self.boundary_flow[0,n,1]], color = 'green')
                plt.plot([self.x[n], self.x[n] + self.boundary_flow[1,n,0]], [self.y[n], self.y[n] + self.boundary_flow[1,n,1]], color = 'blue')
                """
        for n in range(len(self.v0rtices)):
            plt.scatter([self.v0rtices[n,0]],[self.v0rtices[n,1]], color = 'orange')
        plt.axis('square')

###############################################################################
#################   S T R U T C T U R A L  ####################################
#################   Is not the same as:    ####################################
###############################################################################


class Structure2:
    def __init__(self, b = None):
        self.kaze = np.array([[0.,0.],[0.,0.]])
        self.bubbles = []
        if b:
            for bu in b:
                self.add(bu)

    def set_kaze(self,m,n,value, four = None):
        if four is not None:
            self.kaze = np.array([[m,n],[value,four]], dtype = 'float')
        else:
            self.kaze[m,n] = value

    def add(self, b):
        self.bubbles.append(b)
        b.v0rtices = np.array([[b.v0rtices[n,0],b.v0rtices[n,1],0.,0.] for n in range(len(b.v0rtices))])

    def stress(self, x,y):
        s = self.kaze.copy()
        for b in self.bubbles:
            for n in range(len(b.v0rtices)):
                dx,dy = x-b.v0rtices[n,0], y-b.v0rtices[n,1]
                d2 = dx**2+dy**2
                sx, sy = b.v0rtices[n,2]/d2, b.v0rtices[n,3]/d2
                if type(b.boundary_flow) == np.ndarray and n == len(b.v0rtices)-1:
                    s += np.array([[dx*sx, dy*sx], [dx*sy, dy*sy]])
                else:
                    s += np.array([[-dy*sx, dx*sx],[-dy*sy,dx*sy]])
        return s

    def tachinu(self):
        counter = 0
        l = []
        for b in self.bubbles:
            l += [0. for n in range(len(b.v0rtices))]
        equations, totals = np.array([[[0. for ll in l] for ll in l] for q in range(2)]), np.array([[[0.] for ll in l] for q in range(2)])
        i = 0
        rowed = False
        for b in self.bubbles:
            for n in range(b.members):
                rowed = False
                x,y = b.x[n], b.y[n]
                u,v = b.that[n,0], b.that[n,1]
                j = 0
                for c in self.bubbles:
                    for m in range(len(c.v0rtices)):
                        dx, dy = x-c.v0rtices[m,0], y-c.v0rtices[m,1]
                        d2 = dx**2+dy**2
                        for q in range(2):
                            if type(b.boundary_flow) == np.ndarray:
                                rowed = True
                                equations[q,i,j] = -dy/d2
                                equations[q,i+b.members+1,j] = dx/d2
                            else:
                                equations[q,i,j] = dx*u/d2 + dy*v/d2
                        j+= 1
                    if rowed:
                        for q in range(2):
                            equations[q,i,j-1] = dx/d2
                            equations[q,i+b.members+1,j-1] = dy/d2
                if rowed:
                    counter += 1
                for q in range(2):
                    if type(b.boundary_flow) == np.ndarray:
                        totals[q,i,0] = b.boundary_flow[q,n,0]
                        totals[q,i+b.members+1,0] = b.boundary_flow[q,n,1]
                    else:
                        totals[q,i,0] = self.kaze[q,0]*v - self.kaze[q,1]*u
                i += 1
            for n in range(i-b.members, i+1):
                for q in range(2):
                    equations[q,i,n] = 1.
            totals[0,i], totals[1,i] = 0.,0.
            i += 1
            if rowed:
                i += b.members
                print(b)
                for q in range(2):
                    equations[q,i,i] = 2*math.pi
                    total = (b.boundary_flow[q,b.members-1,0]*b.that[b.members-1,1]-b.boundary_flow[q,b.members-1,1]*b.that[b.members-1,0]) * math.sqrt((b.x[0]-b.x[b.members-1])**2 + (b.y[0]-b.y[b.members-1])**2)
                    for o in range(b.members-1):
                        total += (b.boundary_flow[q,o,0]*b.that[o,1] - b.boundary_flow[q,o,1]*b.that[o,0]) * math.sqrt((b.x[o]-b.x[o+1])**2 + (b.y[o]-b.y[o+1])**2)
                    totals[q,i,0] = (total)
                i += 1
        nulls = [0 for r in equations[0] if (r == np.array([0. for rr in r])).all()]
        print(len(nulls))
        print(list(equations[0,::-1][0]))
        print(totals[0,::-1][0])
        soln = np.linalg.solve(equations, totals)
        i = 0
        for b in self.bubbles:
            for n in range(len(b.v0rtices)):
                b.v0rtices[n,2], b.v0rtices[n,3] = soln[0,i], soln[1,i]
                i += 1

    def plot(self, kaze = True, window = 10, pixels= 20):
        window = abs(window)
        for b in self.bubbles:
            b.plot()
            print(b.v0rtices)
        if kaze:
            for mm in range(-pixels//2,pixels//2):
                for nn in range(-pixels//2,pixels//2):
                    m,n = mm*window/pixels, nn*window/pixels
                    s = self.stress(m/2,n/2)
                    plt.plot([m/2, (m+s[0,0])/2], [n/2, (n+s[0,1])/2], color = 'green')
                    plt.plot([m/2, (m+s[1,0])/2], [n/2, (n+s[1,1])/2], color = 'blue')
            for b in self.bubbles:
                for n in range(b.members):
                    x,y = b.x[n], b.y[n]
                    s = self.stress(x,y)
                    plt.plot([x, x+s[0,0]/2], [y, y+s[0,1]/2], color = 'green')
                    plt.plot([x, x+s[1,0]/2], [y, y+s[1,1]/2], color = 'blue')
                    #print(s[0,0]*b.that[n,1]-s[0,1]*b.that[n,0])
        plt.axis('square')
