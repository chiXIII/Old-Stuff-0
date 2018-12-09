from strutctural import *

strut = Structure2(100,100)
for m in range(100):
    for n in range(100):
        if (m-30)**2 + (n-40)**2 <= 100 or (m-60)**2 + (n-40)**2 <= 100:
            strut.transmute(m,n, target = 2)
strut.set_thats()

strut.voertices(extra = False)
vort = strut.tachinu(1,0, 60,60)
strut.plot_kaze(vort,1,0)
strut.plot()
"""

windtunnel = Lattice(100,100)

def cond(windtunnel, x, y):
    if 40 <= y <= 60 and y<=x<=1.5*y-20:
        return True
    else:
        return False

windtunnel.shape(cond)
windtunnel.set_thats()
windtunnel.voertices()
vort = windtunnel.tachinu(1,0.5)
windtunnel.plot_kaze(vort, 1,0.5)
windtunnel.plot()
"""
