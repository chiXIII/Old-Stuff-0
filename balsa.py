from strutctural import *


def predict_break(rtest, rstandard):
    rbase = rstandard/100
    rpoint = rtest/100
    X, Y = [], []
    x,y = -rbase,0
    def update():
        X.append(x)
        Y.append(y)
    for t in range(99):
        y += 0.01
        update()
    x0, y0 = -rbase,1
    for t in range(math.ceil(rbase*100*math.pi/2)):
        x, y = x0+rbase*(1-math.cos(t/(rbase*100))), y0+rbase*(math.sin(t/(rbase*100)))
        update()
    x,y = 0,1+rbase
    for t in range(69):
        x += 0.01
        update()
    x0, y0 = 0.7, 1+rbase
    for t in range(math.ceil(rbase*100*math.pi/2)):
        x,y = x0+rbase*(math.sin(t/(rbase*100))), y0 + rbase*(math.cos(t/(rbase*100))-1)
        update()
    x,y = 0.7+rbase,1
    for t in range(69):
        y -= 0.01
        update()

    x0, y0 = 0.7+rbase, 0.3
    phi = math.pi/2 + math.atan(4/3) - utility.p(math.acos((rtest/100+rbase)/.5))
    for t in range(math.ceil(rstandard*phi)):
        x,y = x0 + rbase*(math.cos(t/(rbase*100))-1), y0 - rbase*math.sin(t/(rbase*100))
        update()
    x,y = 0.7+rbase*(math.cos(phi)), 0.3-rbase*math.sin(phi)
    l = math.ceil(math.sqrt(2500-(rtest+rstandard)**2))
    for t in range(l):
        x = t/l*(0.3-rpoint*math.cos(phi)) + (1-t/l)*(0.7+rbase*math.cos(phi))
        y = t/l*rpoint*math.sin(phi) + (1-t/l)*(0.3-rbase*math.sin(phi))
        update()
    for t in range(math.ceil(rtest*phi)):
        x,y = 0.3-rpoint*math.cos(phi-t/rtest), rpoint*math.sin(phi-t/rtest)
        update()

    complimentx, complimenty = X[::-1][1:], Y[::-1][1:]
    complimenty = [-c for c in complimenty]
    complimentx.append(-rbase)
    complimenty.append(0)
    X += complimentx
    Y += complimenty

    bound = Bubble(X,Y)

    X,Y = [],[]
    V = np.array([ [[0.,0.] for n in range(100)] for q in range(2)])
    for t in range(100):
        X.append(0.35+0.1*math.cos(math.pi*t/50))
        Y.append(0.5 + 0.1*math.sin(math.pi*t/50))
        V[0,t] = [math.cos(math.pi*t/50), math.sin(math.pi*t/50)]
        """
        V[0,t] = [0,1.]
        """
    upper = Bubble(X,Y)

    XX, YY = [],[]
    VV = np.array([ [[0.,0.] for n in range(100)] for q in range(2)])
    for t in range(100):
        XX.append(0.35+0.1*math.cos(math.pi*t/50))
        YY.append(-0.5+0.1*math.sin(math.pi*t/50))
        VV[0,t] = [-math.cos(math.pi*t/50), -math.sin(math.pi*t/50)]
        """
        VV[0,t] = [0,1.]
        """
    lower = Bubble(XX,YY)

    bubbles = [upper, lower]
    sample = Structure2(bubbles)
    sample.set_kaze(1,1,1,1)
    sample.tachinu()
    sample.plot(window = 3)
    plt.show()

predict_break(10, 10)
