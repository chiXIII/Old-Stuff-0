from medusa import *

"""
l = Lattice(1000,1000)
li = [(3,1),(4,2),(4,3),(3,4),(2,4),(1,3),(2,2),(2,3),(3,3),(3,2),(2,5),(3,5),(4,4),(2,6),(5,3),(1,2),(5,2),(2,1),(4,1),(3,0)]
for p in li:
    l.transmute(p[0],p[1])
print(l.obs)
for p in li:
    l.set_that(p[0],p[1])

for m in range(1000):
    for n in range(1000):
        if (m-500)**2/4 + (n-500)**2*4 <= 1000:
            l.transmute(m,n)
l.set_thats()
l.plot()

l.voertices()
stren = l.tachinu(-200,100)
l.plot_kaze(stren, -200,100)

l.plot()
x, y = [],[]
for t in range(100):
    x.append(math.cos(t*2*math.pi/100))
    y.append(0.5*math.sin(t/100*2*math.pi))
af = Bubble(file = 'airfoil_data', mesh = 0.0001)
#af = Bubble(x,y)
soln = af.tachinu(1,1)
af.plot_kaze(1,1,soln)
af.plot()

class tester(Bubble):
    def __init__(self):
        self.x = [0,1]
        self.y = [0,0]
        self.that = np.array([[1,0],[1,0]])
        self.v0rtices = np.array([[-1,1,0.],[0,1,0.]])

test = tester()
soln = test.tachinu(1,1)
print(soln)
test.plot_kaze(1,1, soln)
test.plot()
"""

import pandas as pd

x, y = [],[]
n = 50
for t in range(n):
    x.append(math.cos(2*math.pi*t/n))
    y.append(math.sin(2*math.pi*t/n)/4)
airfoil = Bubble(x,y).multiply(10)
airfoil.set_kaze(1,0)
airfoil.tachinu()
airfoil.plot_kaze()
airfoil.plot()
plt.show()
