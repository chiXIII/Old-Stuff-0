from utility import *
import matplotlib.pyplot as plt
import numpy as np

class Structure:
    def __init__(self, x, y, young, shear, mesh):
        self.young = young
        self.shear = shear
        self.mesh = mesh
        self.width = x
        self.length = y
        self.point_array = [[True for n in range(y)] for n in range(x)]
        self.force_array = [[[0,0] for n in range(y)] for n in range(x)]
        self.force_assigned= [[[False,False] for n in range(y)] for n in range(x)] # overriden in conect()
        self.free_array = [[(0,0) for n in range(y)] for n in range(x)]
        self.connect()

    def point_list(self):
        l = []
        for m in range(self.width):
            for n in range(self.length):
                if self.point_array[m][n]:
                    l.append((m,n))
        return l

    def eat(self, x, y):
        self.point_array[x][y] = False

    """
    def connect(self):
        self.connectors = []
        for m in range(self.width):
            for n in range(self.length):
                if self.point_array[m][n]:
                    if m < self.width-1:
                        if self.point_array[m+1][n]:
                            self.connectors.append( ((m,n),(m+1,n)) )
                        elif n<self.length-1 and self.point_array[m+1][n+1]:
                            self.connectors.append( ((m,n),(m+1,n+1)) )
                    if n < self.length-1:
                        if self.point_array[m][n+1]:
                            self.connectors.append( ((m,n),(m,n+1)) )
                        else:
                            if m>0 and self.point_array[m-1][n+1]:
                                self.connectors.append( ((m,n),(m-1,n+1)) )
                            if m<self.width-1 and self.point_array[m+1][n] and self.point_array[m+1][n+1]:
                                self.connectors.append( ((m,n),(m+1,n+1)) )

                    if n<self.length-1 and m > 0 and not self.point_array[m-1][n] and self.point_array[m-1][n+1]:
                        self.connectors.append( ((m,n),(m-1,n+1)) )

    """
    def connect(self):
        self.connectors = []
        m = -1
        for n in range(self.length):
            self.connectors.append(((m,n),(m+1,n)))
        for m in range(self.width-1):
            n = -1
            self.connectors.append(((m,n),(m,n+1)))
            for n in range(self.length-1):
                if self.point_array[m][n]:
                    self.connectors.append( ((m,n),(m+1,n)) )
                    self.connectors.append( ((m,n),(m,n+1)) )
            n = self.length-1
            self.connectors.append( ((m,n),(m+1,n)) )
            self.connectors.append( ((m,n),(m,n+1)) )
        m = self.width-1
        self.connectors.append(((m,-1),(m,0)))
        for n in range(self.length):
            self.connectors.append( ((m,n),(m,n+1)) )
            self.connectors.append( ((m,n),(m+1,n)) )
        self.force_list = [[0,0] for c in self.connectors]
        self.force_assigned = [[False,False] for c in self.connectors]
        self.plot()

    def plot(self, show = True, connectors = True, divide = 20):
        plist = self.point_list()
        if connectors:
            for n in range(len(self.connectors)):
                c = self.connectors[n]
                normal, shear = self.force_list[n][0], self.force_list[n][1]
                plt.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], color = 'Black')
                x,y = (c[0][0]+c[1][0])/2, (c[0][1] + c[1][1])/2
                if c[0][0] == c[1][0]:
                    plt.plot([x,x],[y, y+normal*self.mesh/divide], color = 'Red')
                    plt.plot([x,x+shear*self.mesh/divide],[y,y], color = 'Green')
                else:
                    plt.plot([x,x+normal*self.mesh/divide],[y,y], color = 'Red')
                    plt.plot([x,x],[y,y+shear*self.mesh/divide], color = 'Green')
        for m in range(self.width):
            for n in range(self.length):
                fx, fy = 0,0
                if self.point_array[m][n]:
                    if ((m,n), (m+1,n)) in self.connectors:
                        j = self.connectors.index(((m,n), (m+1,n)))
                        fx -= self.force_list[j][0]
                        fy -= self.force_list[j][1]
                    if ((m-1,n), (m,n)) in self.connectors:
                        j = self.connectors.index(((m-1,n), (m,n)))
                        fx += self.force_list[j][0]
                        fy += self.force_list[j][1]
                    if ((m,n), (m,n+1)) in self.connectors:
                        j = self.connectors.index(((m,n), (m,n+1)))
                        fy -= self.force_list[j][0]
                        fx += self.force_list[j][1]
                    if ((m,n-1), (m,n)) in self.connectors:
                        j = self.connectors.index(((m,n-1), (m,n)))
                        fy += self.force_list[j][0]
                        fx -= self.force_list[j][1]
                plt.plot([m,m+fx/divide], [n,n+fy/divide], color = 'Purple')
        plt.scatter([p[0] for p in plist],[p[1] for p in plist], color = 'Blue')
        plt.axis('square')
        if show:
            plt.show()

    def assign_force(self, x,y, direction, normal = None, shear = None):
        if self.point_array[x][y]:
            if direction == 0:
                j = self.connectors.index(((x,y),(x+1,y)))
            elif direction == 1:
                j = self.connectors.index(((x,y),(x,y+1)))
            elif direction == 2:
                j = self.connectors.index(((x-1,y),(x,y)))
            elif direction == 3:
                j = self.connectors.index(((x,y-1),(x,y)))
            if normal != None:
                self.force_list[j][0] = normal
                self.force_assigned[j][0] = True
            if shear != None:
                self.force_list[j][1] = shear
                self.force_assigned[j][1] = True

                """
                self.force_array[x][y][0] = Fx
                self.force_assigned[x][y][0] = True
                print('x')
            if Fy!=None:
                self.force_array[x][y][1] = Fy
                self.force_assigned[x][y][1] = True
                print('y')
                """

    def assign_free(self, x,y, is_horizontal = True):
        if horizontal:
            self.free_array[x][y][0] = 1
        else:
            self.free_array[x][y][1] = 1

    def compute_force(self):
        """
        force indexing convention:
            mn-m+1n_compress, mn-m+1n_shear, mn-mn+1_compress, mn-mn+1_shear, m+1n-m+2n_compress, ...
        """
        print(self.force_assigned)
        equations, b = [],[]
        pl = self.point_list()
        cl = self.connectors
        pa = self.point_array
        l = 2*len(cl)
        for i in range(len(pl)):
            point = pl[i]
            x,y = point[0],point[1]
            Fx, Fy, shear, modulix, moduliy = [0 for j in range(l)],[0 for j in range(l)],[0 for j in range(l)],[0 for j in range(l)],[0 for j in range(l)]
            Eright, Eleft, Eup, Edown = False, False, False, False

            if 0 < y and pa[x][y-1]:
                Edown = True
            j = cl.index(((x,y-1),(x,y)))
            Fx[2*j+1] = -1 #shear
            Fy[2*j] = 1 #normal
            shear[2*j+1] = 1
            if 0 < x and pa[x-1][y]:
                Eleft = True
            j = cl.index(((x-1,y),(x,y)))
            Fx[2*j] = 1 # normal
            Fy[2*j+1] = 1 #shear
            shear[2*j+1] = 1
            if x < self.width-1 and pa[x+1][y]:
                Eright = True
            j = cl.index(((x,y),(x+1,y)))
            Fx[2*j] = -1 #normal
            Fy[2*j+1] = -1 #shear
            shear[2*j+1] = 1
            if y < self.length-1 and pa[x][y+1]:
                Eup = True
            j = cl.index(((x,y),(x,y+1)))
            Fx[2*j+1] = 1 # shear
            Fy[2*j] = -1 #normal
            shear[2*j+1] = 1

            if Eup and Eright and pa[x+1][y+1]:
                moduliy[2*cl.index(((x,y),(x,y+1)))] = self.shear
                moduliy[2*cl.index(((x+1,y),(x+1,y+1)))] = -self.shear
                moduliy[2*cl.index(((x,y+1),(x+1,y+1)))+1] = self.young*self.mesh
                moduliy[2*cl.index(((x,y),(x+1,y)))+1] = -self.young*self.mesh
                equations.append(moduliy)
                b.append([0])

                modulix[2*cl.index(((x,y),(x+1,y)))] = self.shear
                modulix[2*cl.index(((x,y+1),(x+1,y+1)))] = -self.shear
                modulix[2*cl.index(((x,y),(x,y+1)))+1] = self.young*self.mesh
                modulix[2*cl.index(((x+1,y),(x+1,y+1)))+1] = -self.young*self.mesh
                equations.append(modulix)
                b.append([0])

            """
            if ( (Eleft and Eright and Eup and Edown) or self.force_assigned[x][y][0]) and not self.free_array[x][y][0]:
                equations.append(Fx)
                b.append([self.force_array[x][y][0]])
            if ( (Eleft and Eright and Eup and Edown) or self.force_assigned[x][y][1]) and not self.free_array[x][y][0]:
                equations.append(Fy)
                b.append([self.force_array[x][y][1]])
            if not(Eleft and Eright and Eup and Edown) or self.free_array[x][y][0]:
                equations.append(shear)
                moment = 0
                if not Eright:
                    moment += self.force_array[x][y][1]
                if not Eleft:
                    moment -= self.force_array[x][y][1]
                if not Eup:
                    moment -= self.force_array[x][y][0]
                if not Edown:
                    moment += self.force_array[x][y][0]
                b.append([moment])
            """
            equations.append(Fx)
            equations.append(Fy)
            b += [[0],[0]]
        for i in range(len(self.force_assigned)):
            for eo in range(2):
                if self.force_assigned[i][eo]:
                    row = [0 for p in range(l)]
                    row[2*i+eo] = 1
                    equations.append(row)
                    b.append([self.force_list[i][eo]])

            #self.plot(connectors = False)
        print(len(Fy))
        print(len(equations))

        for n in range(len(equations)):
            A = equations.copy()
            A.pop(n)
            A = np.array(A, dtype='float')
            print(np.linalg.matrix_rank(A))
        e = np.array(equations, dtype='float')
        b = np.array(b, dtype='float')
        solution = np.linalg.solve(e,b)
        print(solution)
        solution = list(solution)

        for n in range(len(solution)):
            self.force_list[n//2][n%2] = solution[n]
