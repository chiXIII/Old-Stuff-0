

############################################## math and physics #########################################################


from math import pi, sqrt, copysign # copysign for optimizer functions
def deg(alpha):
    return alpha/(2*pi)*360

dens = 1.225
visc = 1.983 * 10**-5


def reyn(velocity, length):
    return dens*velocity*length/visc

############################################### aircraft #####################################################################



from Airfoils import*

class Airfoil:
    def __init__(self, title, data):
        self.data = data
        self.title = title
        self.Cd = self.data['Cd'] # a function of reynolds and alpha

    def Cl(self, Re, aoa):
        """returns the lift coefficient for geometric AOA in degrees"""
        return self.data['slope'] * (aoa - self.data['a0'])

    def optimum_alpha(self, Re, a0 = 0, dy0 = None):
        """returns the angle of attack with highest lift/drag"""
        ratio0 = self.Cl(Re, a0)/self.Cd(Re, a0)
        dx = 0.01
        ratio1 = self.Cl(Re, a0+dx)/self.Cd(Re, a0+dx)
        dy = ratio1 - ratio0
        if dy0 is not None and (dy < 10**-5 or copysign(1, dy) != copysign(1, dy0)):
            return a0 + dx / 2
        else:
            dy0 = dy
            a0 += dy
            return self.optimum_alpha(Re, a0, dy0)


class Wing:
    def __init__(self, name, aif, length, width, alpha, stabilizer = False):
        """LENGTH and WIDTH in meters"""
        self.airfoil = aif
        self.length, self.width, self.alpha = length, width, alpha
        self.stabilizer = stabilizer
        self.name = name

    @property
    def area(self):
        return self.length * self.width

    @property
    def AR(self):
        return self.length**2 / self.area

    @property
    def e(self):
        return 0.7

    def Re(self, v):
        return reyn(v, self.width)

    def lift(self, v, pitch = 0):
        """Takes V in m/s and geometric AOA in degrees and returns lift in newtons
        """
        return self.length * self.width * self.airfoil.Cl(self.Re(v), self.alpha + pitch) * dens * 0.5 * v**2

    def drag(self, v, pitch = 0):
        """See lift.  Returns newtons
        """
        Cdeff = self.airfoil.Cd(self.Re(v), self.alpha + pitch) + abs(self.airfoil.Cl(self.Re(v), self.alpha + pitch))**2/(pi*self.e*self.AR)
        return self.length * self.width * Cdeff * dens * 0.5 * v**2

    


class Aircraft:

    ## Essential ##
    def __init__(self, mass, surfaces, pitch = 0, velocity = 2, slope = 0):
        """an aircraft with a MASS in kg and a list of SURFACES and their relative geometric angles of attack which are wings.
        """
        self.mass = mass
        self.weight = 9.8*mass # weight in newtons
        self.surfaces = surfaces
        self.slope = slope
        self.pitch = pitch
        self.velocity = velocity

    def reyn(self):
        return reyn(self.velocity, self.surfaces[0].width) #will need to replace this often

    def lift(self):
        return sum([self.surfaces[n].lift(self.velocity, self.pitch) for n in range(len(self.surfaces))])

    def drag(self):
         return sum([self.surfaces[n].drag(self.velocity, self.pitch) for n in range(len(self.surfaces))])

    ## Utility ##

    def calc_slope(self): # CALL THIS TO SET GLIDE SPEED & SLOPE
        self.slope = self.drag()/self.lift()
        m = self.slope
        self.calc_speed()
        M = self.drag()/self.lift()
        self.slope = M
        if abs(m - M) < 0.001:
            return
        else:
            self.calc_slope()

    def calc_speed(self):  # DO NOT CALL
        if self.slope == 0:
            self.calc_slope()
        m = self.slope
        v = self.velocity
        self.velocity = sqrt(self.weight*sum([self.surfaces[n].area * 0.5 * dens * v**2 for n in range(len(self.surfaces))]) /(self.drag() * (1/m + m**2) * sqrt(1 + m)))

    def optimizeMEA(self, wing, a0 = -5, a1 = 10, tolerance = 0.001):
        assert wing in self.surfaces
        dx = (a1 - a0)/30
        x = a0
        xmax, max = x, 0
        while x <= a1:
            wing.alpha = x
            ratio = wing.lift(self.velocity)/wing.drag(self.velocity)
            if ratio > max and wing.lift(self.velocity) > 0:
                max = ratio
                xmax = x
            x += dx
        wing.alpha = xmax
        if abs(3*dx) <= tolerance:
            print('alpha', xmax)
            return xmax
        else:
            return self.optimizeMEA(wing, xmax - dx, xmax + dx, tolerance)

            
    def fly(self):
        self.calc_slope()
        print({'m' : self.slope , 'v' : self.velocity, 'Re' : self.reyn()})
        return {'m' : self.slope , 'v' : self.velocity, 'Re' : self.reyn()}

############################################# constants #########################################################

S8064 = Airfoil('Symmetrical S8064', S8064dat)
AG12 = Airfoil('Sailplane AG12', AG12dat)

def fix_re_floor(aircraft, re = 45000):
    if aircraft.reyn() < re:
        print(aircraft.reyn())
        for wing in aircraft.surfaces:
            wing.width += 0.01
        return False
    return True


##############################################  primary mechanic ###################################################

def Deepen(aircraft, constraints):
    finished_constraints = []
    met_constraints = []
    impossible_constraints = []

    for constraint in constraints:
        while True:
            aircraft.fly()
            for constr in finished_constraints:
                constr.update()
            try:
                aircraft.fly()
                constraint.establish()
            except ImpossibleConstraint as e:
                impossible_constraints.append(e.constraint)
                finished_constraints.append(e.constraint)
                break
            except MetConstraint as e:
                met_constraints.append(e.constraint)
                finished_constraints.append(e.constraint)
                break

    flight = aircraft.fly()
    print(flight)
    print('met', met_constraints)
    print('failed', impossible_constraints)
        

class ImpossibleConstraint(Exception):

    def __init__(self, constraint):
        self.constraint = constraint

class MetConstraint(Exception):

    def __init__(self, constrant):
        self.constraint = constraint

class Constraint:

    def __init__(self, aircraft,  test_fn, establish_fn, update_fn, fuse):
        self.aircraft = aircraft
        self.test_fn = test_fn
        self.establish_fn = establish_fn
        self.update_fn = update_fn
        self.fuse = fuse

    def establish(self):
        if self.fuse == 0:
            raise ImpossibleConstraint(self)
        self.establish_fn(self.aircraft)
        self.fuse -= 1
        if self.test_fn(self.aircraft):
            raise MetConstraint(self)

    def update(self):
        self.update_fn(self.aircraft)

    

##################################################### tests #########################################################


w1, w2, w3 = Wing('port', AG12, 0.5, 0.1, 2), Wing('starboard', AG12, 0.1, 0.3, 2), Wing('tail', S8064, 0.1, 0.3, 0, True)
plane = Aircraft(0.2, [w1, w2, w3])

def set_angles(plane):
    plane.optimizeMEA(w1)
    plane.optimizeMEA(w2)

def fix_angles(plane):
    plane.optimizeMEA(w1, w1.alpha - 1, w1.alpha + 1)
    plane.optimizeMEA(w2, w2.alpha - 1, w2.alpha + 1)


def angle_establish(self): # 'self' for copy/paste convenience.  This is an aircraft.
    def optimizeMEA(wing, a0 = -5, a1 = 10, tolerance = 0.001):
        efficient_angles.optimized['first time'] = False
        dx = (a1 - a0)/30
        x = a0
        xmax, max = x, 0
        while x <= a1:
            wing.alpha = x
            ratio = wing.lift(self.velocity)/wing.drag(self.velocity)
            if ratio > max and wing.lift(self.velocity) > 0:
                max = ratio
                xmax = x
            x += dx
        wing.alpha = xmax
        if abs(2*dx) <= tolerance:
            if min(abs(xmax - a1), abs(xmax - a0)) <= dx:
                efficient_angles.optimized[wing.name] = -1
            else:
                efficient_angles.optimized[wing.name] = 1
        print(a0, a1, 'alpha', xmax)
        efficient_angles.optimized['dx'] = dx
        print(efficient_angles.optimized)
        
    for wing in self.surfaces[:2]:
        efficient_angles.optimized[wing.name] = False
        if efficient_angles.optimized['first time']:
            optimizeMEA(wing)
        else:
            optimizeMEA(wing, wing.alpha - 5*efficient_angles.optimized['dx'], wing.alpha + 5*efficient_angles.optimized['dx'])

def angle_test(acft):
    if [bool(efficient_angles.optimized[w]) for w in efficient_angles.optimized] == [True for w in acft.surfaces[:2] ]:
        if [bool(efficient_angles.optimized[w]) for w in efficient_angles.optimized] != [1 for w in acft.surfaces[:2] ]:
            print('Warning:')
            print('Wing angles of attack may not be properly optimized.')
        return True
    else:
        return False
    

def angle_update(self):  # 'self' for copy/paste convenience.  This is an aircraft.
    def optimizeMEA(wing, a0, a1, tolerance = 0.001):
        dx = (a1 - a0)/30
        x = a0
        xmax, max = x, 0
        while x <= a1:
            wing.alpha = x
            ratio = wing.lift(self.velocity)/wing.drag(self.velocity)
            if ratio > max and wing.lift(self.velocity) > 0:
                max = ratio
                xmax = x
            x += dx
        wing.alpha = xmax
        if abs(3*dx) <= tolerance:
            return
        else:
            optimizeMEA(wing, wing.alpha - dx, wing.alpha + dx)

    for wing in self.surfaces[:2]:
        optimizeMEA(wing, wing.alpha - 1, wing.alpha + 1)


efficient_angles = Constraint(plane, angle_test, angle_establish, angle_update, 100) 
efficient_angles.optimized = {'first time':True}

print(w1.width, w1.alpha, w3.width, w3.alpha)


Deepen(plane, [efficient_angles])
