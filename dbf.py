from DeepSky import *
"""
conventions:
*All wight in lb
*all distance in ft
*all mass in slug
*all speed in ft/sec
*all energy in ft-lb
"""

sqin = [613.8, 224, 215.948, 749.52, 661.25, 270, 422.625]
sqft = [s/144 for s in sqin]
pounds = [1.6, 0.437, 0.45, 1.49, 1.72, 0.82, 1.01]

w_coefs = np.polyfit(sqft, pounds, 1)
def ewnb(S):
    return w_coefs[1] + w_coefs[0]*S
plt.scatter(sqft, pounds)
plt.plot([0, 8],[ewnb(0), ewnb(8)])
plt.close()

cd = [0.0493, 0.053, 0.0509, 0.0380, 0.0413, 0.0358, 0.0899, 0.1138]
sqin = [540.454, 547.17, 215.948, 613.8, 479.52, 661.25, 270, 422.625]
sqft = [s/144 for s in sqin]
d_coefs = np.polyfit(sqft, cd, 1)
def CD0(S):
    return d_coefs[1] + d_coefs[0]*S
plt.scatter(sqft, cd)
plt.plot([0, 8],[CD0(0), CD0(8)])
plt.close()

class Micro(Product):
    """
    Parameters: ncells, payload, S, v
    """

    def __init__(self, ncells, payload, S, v):
        Product.__init__(self, plist = [ncells, payload, S, v])

    def set_constants(self, lap = 800, CLmax = 1.5, battery_energy = 6840.6/3,\
battery_weight = 0.414469/3, AR = 2.35443, e = 0.7, launch_speed = 22):
        self.lap, self.CLmax, self.battery_energy, self.battery_weight, self.AR, self.e, self.launch_speed =\
lap, CLmax, battery_energy, battery_weight, AR, e, launch_speed

    def weight(self):
        return self.battery_weight*self.parameters[0] + ewnb(self.parameters[2]) + self.parameters[1]

    def hand_launch_constraint(self):
        return math.sqrt(2*self.weight()/(rho*self.CLmax*self.parameters[2])) <= self.launch_speed
    def energy_constraint(self, print = False):
        #return self.parameters[0] >= 4-self.parameters[3]/10
        if print:
            return p(self.parameters[0]*self.battery_energy) >= p(0.5*self.lap*rho*(self.parameters[3]**2)*self.parameters[2]*CD0(self.parameters[2]) + 2*self.lap*(self.weight()**2)/(rho*(self.parameters[3]**2)*math.pi*self.AR*self.e*self.parameters[2]))
        else:
            return self.parameters[0]*self.battery_energy >= 0.5*self.lap*rho*(self.parameters[3]**2)*self.parameters[2]*CD0(self.parameters[2]) + 2*self.lap*(self.weight()**2)/(rho*(self.parameters[3]**2)*math.pi*self.AR*self.e*self.parameters[2])
    def stall_speed_constraint(self):
        return self.parameters[3] >= math.sqrt(2*self.weight()/(rho*self.CLmax*self.parameters[2]))

######################## Micro class design for competition 2018 #####################################


"""
Parameters: ncells, payload, S, v
p(11.875*13.625/144)
"""

micro = Micro(1,1,5,10)
micro.set_constants()

def score(self):
    return self.parameters[1]/math.sqrt(self.weight()-self.parameters[1])

def box_size_constraint():
    return micro.parameters[2] <= (11.875*13.625/144)*(3-2*micro.parameters[1]/3.93)
# 
intervals = [(0,5), (0, 3.93), (0.1,4), (1, 70)]
resolution = [5, 20, 20, 20]
constraints = [micro.hand_launch_constraint, micro.energy_constraint, box_size_constraint, micro.stall_speed_constraint]

#print( micro.kid(score, intervals, resolution, [1, 0.01, 0.05, 1], constraints, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], [('ncells', 'payload'), ('ncells', 'S'), ('ncells', 'v'), ('payload', 'S'), ('payload', 'v'), ('S', 'v')]) )

#micro.feasible_space(intervals, resolution, constraints)
micro.general_optimize(score, intervals, resolution, constraints, integers = [1, 0,0,0])
print('parameters:')
print(micro.parameters)
print('score')
print(score(micro))
print('ewnb:')
print(ewnb(micro.parameters[2]))
print('empty weight')
print(micro.weight()-micro.parameters[1])

cmap = matplotlib.cm.get_cmap('inferno')

"""
s, o = micro.slice(0,3)
x = [p[0] for p in s]
y = [p[1] for p in s]
"""

micro.display_slice(0,1,'ncells', 'payload')
micro.display_slice(0,2,'ncells', 'S')
micro.display_slice(1,2,'payload', 'S')
micro.display_slice(1,3,'payload', 'v')
micro.display_slice(0,3,'ncells', 'v')
micro.display_slice(2,3,'S', 'v')

