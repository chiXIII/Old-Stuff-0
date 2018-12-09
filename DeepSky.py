from utility import *
import matplotlib.pyplot as plt
import matplotlib

prnt = p

rho = 0.0023769 #slug/ft**3
cmap = matplotlib.cm.get_cmap('inferno')


def optimize(assign, update, objective, feasible, low, high, step, save, revert):
    assert (high>low),'Upper bound must be greater than lower bound or else loop will not halt.'
    assign(low)
    update()
    param_list, obj_list = [],[]
    unfeas_param, unfeas_obj = [],[]

    x, max, xmax = low, objective(), low
    while x<=high:
        x+=step
        assign(x)
        update()
        param_list.append(x)
        ob = objective()
        obj_list.append(ob)

        if ob>max and feasible():
            max=ob
            xmax = x
            save()
        if not feasible():
            unfeas_param.append(x)
            unfeas_obj.append(ob)
            
    revert()

    plt.scatter(param_list, obj_list, color = 'Blue')
    plt.scatter(unfeas_param, unfeas_obj, color = 'Red')
    plt.scatter(xmax, max, color = 'Purple')
    plt.show()

class Product:

    def __init__(self, plist = [], objective = lambda self: 0):
        self.parameters = plist
        self.objective = objective

    def set_parameters(self, plist):
        self.parameters = plist

    def set_objective(self, objective):
        self.objective = objective

    def feasible_space(self, intervals, resolution, constraint_list=[]):
        assert len(intervals) == len(self.parameters), "INTERVALS must contain a range of acceptable values (in the form of a tuple or list) for each member of self.parameters"
        assert [len(x) for x in intervals] == [2 for x in intervals], "members of INTERVALS must consist of an upper and lower bound"
        if type(resolution) == float or type(resolution) == int:
            resolution = [resolution for i in intervals]
        else:
            assert len(resolution) == len(intervals), "if RESOLUTION is not a scalar, it must be an iterable containing a resoulution for each parameter"

        self.feasible_points = []

        def iterate(n):
            x = intervals[n][0]
            step = (intervals[n][1]-intervals[n][0])/resolution[n]

            while x < intervals[n][1]:
                self.parameters[n] = x
                if n>=len(intervals)-1:
                    if [c() for c in constraint_list] == [True for c in constraint_list]:
                        self.feasible_points.append(self.parameters.copy())
                else:
                    iterate(n+1)
                x += step

        iterate(0)
        return self.feasible_points

    def optimize(self, objective):
        self.objectives = []
        max, maxp = 0, [0 for p in self.parameters]
        for p in self.feasible_points:
            self.parameters = list(p)
            o = objective(self)
            self.objectives.append(o)
            if o > max:
                maxp = p.copy()
                max = o
        self.parameters = maxp
        return o

    def slice(self, parameter0, parameter1):
        sliced = self.feasible_points.copy()
        obs = self.objectives.copy()
        def slicer(point_list0, point_list1, parameteri):
            return [p for p in point_list0 if p[parameteri] == self.parameters[parameteri] ], [point_list1[n] for n in range(len(point_list1)) if point_list0[n][parameteri] == self.parameters[parameteri] ]
        for parameteri in range(len(self.parameters)):
            if parameteri != parameter0 and parameteri != parameter1:
                sliced, obs = slicer(sliced, obs, parameteri)

        return sliced, obs

    def display_slice(self, parameter0, parameter1, param0label = 'x', param1label = 'y'):
        sliced, obs = self.slice(parameter0, parameter1)
        maxcolor = max(obs)
        mincolor = min(obs)
        for n in range(len(sliced)):
            try:
                plt.scatter(sliced[n][parameter0], sliced[n][parameter1], color = cmap((obs[n]-mincolor)/(maxcolor-mincolor)) )
            except:
                plt.scatter(sliced[n][parameter0], sliced[n][parameter1])
        #plt.scatter([p[0] for p in ncells_v_v], [p[3] for p in ncells_v_v])   
        plt.xlabel(param0label)
        plt.ylabel(param1label)
        plt.show()

    def general_optimize(self, objective, intervals, resolution, constraint_list=[], integers = None):
        self.feasible_space(intervals, resolution, constraint_list)
        self.optimize(objective)

        n_of_p = [ [] for p in self.parameters]
        for p in self.feasible_points:
            for n in range(len(self.parameters)):
                if p[n] not in n_of_p[n]:
                    n_of_p[n].append(p[n])

        def get_res(n):
            if integers and integers[n]:
                return resolution[n]
            return resolution[n]**2/len(n_of_p[n])
        resolution = [get_res(n) for n in range(len(resolution))]
        self.feasible_space(intervals, resolution, constraint_list)
        self.optimize(objective)
            

    def kid(self, objective, intervals, resolutions0, resolutions1, constraint_list=[], parameters_to_display = [], axis_labels = []):
        self.feasible_space(intervals, resolutions0, constraint_list)
        self.optimize(objective)
        for n in range(len(parameters_to_display)):
            p = parameters_to_display[n]
            if axis_labels:
                self.display_slice(p[0], p[1], axis_labels[n][0], axis_labels[n][1])
            else:
                self.display_slice(p[0], p[1])

        def set_resolution(rlist, n):
            if (intervals1[n][1]-intervals1[n][0])/rlist[n] > resolutions1[n]:
                return 4
            else:
                return 4

        print('intervals1')
        intervals1 = prnt([( max(self.parameters[n]-(intervals[n][1] - intervals[n][0])/resolutions0[n], intervals[n][0]), min(self.parameters[n]+(intervals[n][1] - intervals[n][0])/resolutions0[n], intervals[n][1]) ) for n in range(len(resolutions0))])
        while [(intervals1[n][1]-intervals1[n][0])/resolutions0[n]<=resolutions1[n] for n in range(len(resolutions0))] != [True for n in range(len(resolutions0))]:
            print(intervals1)
            print('iteration')
            print('parameters')
            print(self.parameters)
            resolutions0 = prnt([set_resolution(resolutions0, n) for n in range(len(resolutions0)) ])
            intervals1 = prnt([( max(self.parameters[n]-(intervals1[n][1] - intervals1[n][0])/resolutions0[n], intervals[n][0]), min(self.parameters[n]+(intervals1[n][1] - intervals1[n][0])/resolutions0[n], intervals[n][1]) ) for n in range(len(resolutions0))])
            self.feasible_space(intervals1, resolutions0, constraint_list)
            self.optimize(objective)

        for n in range(len(parameters_to_display)):
            p = parameters_to_display[n]
            if axis_labels:
                self.display_slice(p[0], p[1], axis_labels[n][0], axis_labels[n][1])
            else:
                self.display_slice(p[0], p[1])
        return self.parameters, objective(self)

"""
thingy = Aircraft(lambda:9.8, lambda:1)
thingy.fly(10, [1,1])

def c1():
    return thingy.parameters[1]>thingy.parameters[0]
def c2():
    return thingy.parameters[0]<thingy.v
def c3():
    return thingy.parameters[1]<2*thingy.parameters[0]

print(len(thingy.feasible_space([(0,20),(-1, 15)], 30.0, [c1,c2,c3])))
plt.scatter(thingy.feasible_points[0], thingy.feasible_points[1])
plt.show()

class Tester:
    def __init__(self, parameter):
        self.parameter = parameter
        self.dependent = -parameter**2
        self.backup_param = parameter
        self.backup_dep = self.dependent
    def set_param(self, x):
        self.parameter = x
    def update(self):
        self.dependent = -self.parameter**2
        if self.parameter == -6:
            self.dependent = 1
    def objective(self):
        return self.dependent
    def feasible(self):
        if self.parameter>0:
            return True
        return False
    def save(self):
        self.backup_param = self.parameter
        self.backup_dep = self.dependent
    def revert(self):
        self.parameter = self.backup_param
        self.update()

testy = Tester(3)
optimize(testy.set_param, testy.update, testy.objective, testy.feasible, -8, 4, 1, testy.save, testy.revert)
print(testy.parameter, testy.dependent)
"""
