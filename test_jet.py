from jet import *

s = Structure(4,7, 1,1,1)
s.connect()
"""
for c in s.connectors:
    s.plot(show = False, connectors = False)
    plt.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]])
    plt.show()
"""
for n in range(7):
    s.assign_force(0,n,2,normal = 0, shear = 0)
for n in range(6):
    s.assign_force(2,n,0,normal = 0, shear = 0)
for m in range(4):
    s.assign_force(m,0,3, normal = 0, shear = 1)
    s.assign_force(m,3,1, normal = 0, shear = 1)
s.compute_force()
s.plot()
