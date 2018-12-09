from strutctural import *

bubbles = []

X,Y = [],[]
for t in range(400):
    X.append(5*math.cos(math.pi*t/200))
    Y.append(-5*math.sin(math.pi*t/200))
bubbles.append(Bubble(X,Y, mesh = 0.5))

x, y = [], []
for t in range(100):
    x.append(math.cos(2*math.pi*t/100) + 1), y.append(math.sin(2*math.pi*t/100) -2)
bubbles.append(Bubble(x,y))
xx, yy = [], []
for t in range(100):
    xx.append(math.cos(2*math.pi*t/100)), yy.append(math.sin(2*math.pi*t/100) + 2.0749834)
bubbles.append(Bubble(xx,yy))
xxx, yyy = [], []
for t in range(100):
    xxx.append(math.cos(2*math.pi*t/100)/2+3.05893), yyy.append(math.sin(2*math.pi*t/100) + 1)
bubbles.append(Bubble(xxx,yyy))

x4,y4= [], []
for t in range(100):
    x4.append(math.cos(2*math.pi*t/100)/2 - 3.09875), y4.append(0.5*math.sin(2*math.pi*t/100) - 1.3421)

vectors = np.array([[[q,1.] for n in x4] for q in range(2)])
bubbles.append(Bubble(x4,y4, boundary_flow = vectors) )

x5,y5= [], []
for t in range(100):
    x5.append(math.cos(2*math.pi*t/100)/2 - 3.2546), y5.append(math.sin(2*math.pi*t/100) + 1.01984)

vectors1 = np.array([[[-q,-1.] for n in x5] for q in range(2)])
bubbles.append(Bubble(x5,y5, boundary_flow = vectors1) )

"""
b = Bubble([0,1,1.], [0, 0., 1])
b.x, b.y = [-0.5], [0]
b.thats = np.array([[0,1]])
b.v0rtices = utility.p(np.array([[-0.75,0.1,0,0],[-1,-2.,0,0],[-1.25,1.5,0,0]]))
b.boundary_flow = np.array([ [[1,1.]], [[0.,0]] ])
bubbles.append(b)
"""

strut = Structure2()
for b in bubbles:
    strut.add(b)
strut.set_kaze(1,0.,0,0)
strut.tachinu()
strut.plot()
plt.show()

