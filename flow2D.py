import utility
from DeadReckon2D import *
from math import copysign, e
import math


dt = 0.025
res = 100


class Surface(Curve):
    def __init__(self, r = 1, value = 0, freestream = (0,0)):
        self.streams = []
        Curve.__init__(self, r, value)
        self.freestream = freestream
        self.vortex_centers, self.test_points, self.vortex_coefficients = [], np.array([]), []

    def split(self, fraction):
        assert 0<=fraction<=1
        n = int(len(self.state)*fraction)
        new_state = self.state.iloc[n:]
        self.state = self.state.iloc[:n]
        new = Surface(freestream = self.freestream)
        new.state = new_state
        return new

    def add_curvature(self, radians):
        dtheta = radians/len(self.state)
        t1, n = self.state.iloc[0]['t'], 0
        for t in self.state.iloc[1:]['t']:
            n += 1
            row = self.state.loc[self.state['t'] == t1]
            self.state.loc[self.state['t'] == t, 'x'] = row['x'] + dt*row['v']*cos(row['theta'])
            self.state.loc[self.state['t'] == t, 'y'] = row['y'] + dt*row['v']*sin(row['theta'])
            self.state.loc[self.state['t'] == t, 'theta'] += n*dtheta
            self.state.loc[self.state['t'] == t, 'r'] = (self.state.loc[self.state['t'] == t, 'theta'] - row['theta']) / row['v']
            t1 = t

    def copy(self):
        s = Surface()
        s.state = self.state.copy()
        assert type(self.freestream) == tuple, 'This attribute must be immutable.  I have required it be a tuple.'
        s.streams, s.streams = self.streams.copy(), self.freestream
        s.vortex_centers, s.test_points = self.vortex_centers.copy(), self.test_points.copy()
        self.vortex_coefficients = np.array([])
        return s

    def set_freestream(self, freestream):
        self.freestream = freestream

    def rotate(self, delta_angle):
        for t in self.state['t']:
            row = self.state.loc[self.state['t'] == t]
            self.state.loc[self.state['t'] == t, 'x'] = row['x']*cos(delta_angle) - row['y']*sin(delta_angle)
            self.state.loc[self.state['t'] == t, 'y'] = row['x']*sin(delta_angle) + row['y']*cos(delta_angle)
            self.state.loc[self.state['t'] == t, 'theta'] += delta_angle

    def translate(self, vector):
        for t in self.state['t']:
            self.state.loc[self.state['t'] == t, 'x'] += vector[0]
            self.state.loc[self.state['t'] == t, 'y'] += vector[1]

    def vortex_points(self, spacing = 10):
        """
        Set points to be the centers of hypothetical irrotational vortices to
        superpose, and set test points on the surface at which to assert that 
        the flow velocity is parallel to the surface and thus generate a linear
        equation.  SPACING indicates how frequently to place centers/test
        points, measured in surface.state rows.
        """
        print('Generating vortex centers and test points...')

        vc = self.vortex_centers

        backwards = self.state.iloc[::-1]

        self.test_points = []
        for i in range(1, len(self.state)):
            if not (i - spacing//2)%spacing:
                row = self.state.iloc[i]
                opp_row = backwards.iloc[i]
                to_mul = min(1, distance((row['x'], row['y']), (opp_row['x'], opp_row['y'])))
                vc.append((row['x'] + to_mul*sin(row['theta'])/5, row['y'] - to_mul*cos(row['theta'])/5))

        for i in range(len(self.state)):
            if not (i)%spacing:
                row = self.state.iloc[i]
                self.test_points.append(row)

        if len(vc) > len(self.test_points):
            vc.pop()
        elif len(vc) < len(self.test_points):
            self.test_points.pop()
        assert len(vc) == len(self.test_points)

        print('Done')
        print('Generating plot...')

        self.plot(plt)
        plt.axis('equal')
        plt.scatter([c[0] for c in vc], [c[1] for c in vc])
        plt.scatter([p['x'] for p in self.test_points], [p['y'] for p in self.test_points])

        print('Done')
        plt.show()

        print('')

    def vortex_predict(self):
        """
        Computes strength coefficients for vortices with centers specified
        such that the flow is parallel to the surface at all specified test
        points.
        """

        print('Computing coefficients for {} vortex centers...'.format(str(len(self.vortex_centers))) )

        assert self.vortex_centers, 'No vortex centers to calculate coefficients for.  Try calling Surface.vortex_centers()'
        vc = self.vortex_centers

        rows = []
        b = []
        for p in self.test_points:
            row = []
            m = tan(p['theta'])
            com = (m*p['y'] + p['x'])
            for c in vc:
                row.append( (-c[0] - m*c[1] + com)/((p['y'] - c[1])**2 + (p['x'] - c[0])**2) )
            rows.append(row)
            b.append([m*self.freestream[0] - self.freestream[1]])
        A, b = np.array(rows), np.array(b)
        self.vortex_coefficients = np.linalg.solve(A, b)
        k = self.vortex_coefficients
        print('Done\nBelow are the coefficients:')
        print(k, '\n')

        print('Plotting.')
        self.plot(plt)
        plt.axis('square')

        for p in self.test_points:
            xx, yy = self.freestream[0]/10,self.freestream[1]/10
            for i in range(len(vc)):
                c = vc[i]
                xx += k[i]*(c[1] - p['y'])/((p['y'] - c[1])**2 + (p['x'] - c[0])**2)/10
                yy += k[i]*(p['x'] - c[0])/((p['y'] - c[1])**2 + (p['x'] - c[0])**2)/10
            plt.plot([p['x'], p['x'] + xx], [p['y'], p['y'] + yy])
        plt.show()

        print('computing velocity distribution')
        for t in self.state['t']:
            row = self.state.loc[self.state['t'] == t]
            x, y = row['x'], row['y']
            xx, yy = self.freestream[0], self.freestream[1]
            for i in range(len(vc)):
                c = vc[i]
                xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))
                yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))
            self.state.loc[self.state['t'] == t, 'value'] = sqrt( xx**2 + yy**2 )

        plt.plot(self.state['t'], self.state['value'])
        plt.show()

        print('\n')

    def vortex_plot(self, resolution = 20):
        vc, k = self.vortex_centers, self.vortex_coefficients
        assert vc, 'No vortex centers.  Try calling Surface.vortex_centers()'
        assert k.any(), 'No vortex coefficients to generate velocity vector field.  Try calling Surface.vortex_predict()'

        ymin, ymax = min(self.state['y']), max(self.state['y'])
        xmin, xmax = min(self.state['x']), max(self.state['x'])

        ymin_bound, ymax_bound = int(ymin - 1 * (ymax - ymin)) - 1, int(ymax + 2 * (ymax - ymin)) + 1
        xmin_bound, xmax_bound = int(xmin - 1 * (xmax - xmin)) - 1, int(xmax + 2 * (xmax - xmin)) + 1

        for y in range(ymin_bound, ymax_bound):

            print('Generating streamline {}'.format(str(y)) )
            x,y = xmin_bound,y
            X,Y = [],[]
            while x < xmax_bound:
                X.append(x)
                Y.append(y)
                xx, yy = self.freestream[0]/resolution, self.freestream[1]/resolution
                for i in range(len(vc)):
                    c = vc[i]
                    xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))/resolution
                    yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))/resolution
                x += xx
                y += yy
            plt.plot(X,Y, color = 'blue')

        self.plot(plt)
        plt.axis('equal')
        plt.show()


    def vortex_flow(self, spacing = 10, resolution = 20):
        self.vortex_points(spacing)
        self.vortex_predict()
        self.vortex_plot(resolution)

#########################################################################################
#the following methods are depreciated

    def follow(self, s, const = 0.2, overwrite = []):
        def n(state):
            return const/state['value']
        def getva(state):
            return state['r']/(state['r'] - n(state)) * state['value']
        return Curve.follow(self, s, n, getva = getva, overwrite = overwrite)

    def simulate(self, iterations = 5, width = 0.1, show = True):
        self.streams = []
        fig, axs = plt.subplots(1,5, figsize = (12, 5))
        s = self
        s.plot(axs)
        for i in range(iterations):
            s1 = Streamline()
            plts = s1.follow(s, width)
            for p in plts:
                axs[0].plot(p[0], p[1], color = 'Blue')
            s = s1
            s.plot(axs, color = 'Red')
            self.streams.append(s1)
        return fig, axs

    def freestream(self, lines, v):
        self.streams = [Streamline() for l in range(lines)]
        for stream in self.streams:
            for i in range(len(self.state)):
                stream.plus(dt)
        print(self.streams)
        i = 1
        j = i
        n = 0.2
        state = self.state.iloc[::-1]
        print(state)
        stream = 0
        toggle = True
        trip = False
        le = len(state)
        for s in self.streams:
            s.x, s.y = [], []
            s.r, s.theta = [], []
            s.t, s.v = [], []
            s.value = []
        while i < le:
            i = j
            toggle = not toggle
            stream = i + int(toggle) - 1
            stream = stream
            if stream >= lines - 1:
                trip = True
            if trip:
                stream = lines - 1
            if not i % 10:
                print(i)
            while stream >= 0:
                state1 = state.iloc[i]
                t = state1['t']
                theta = state1['theta']
                x = state1['x'] + (1+int(trip))*stream*n*cos(theta + pi/2)
                y = state1['y'] + (1 + int(trip))*stream*n*sin(theta + pi/2)
                v = state1['v']
                value = state1['value']
                r = state1['r']
                s = self.streams[stream]
                s.x.append(x)
                s.y.append(y)
                s.r.append(r)
                s.theta.append(theta)
                s.t.append(t)
                s.v.append(v)
                s.value.append(value)
                i += 1
                stream -= 1
            if toggle or trip:
                j += 1
        self.plot(plt)
        for s in self.streams:
            s.state = pd.DataFrame({'x':s.x, 'y':s.y, 'r':s.r, 'theta':s.theta, 't':s.t, 'v':s.v, 'value':s.value})
            print(s.state)
            s.plot(plt)
        plt.show()

#############################################################################################

def surface_from_points(point_list, angles = [], open_curve = False, solution = 1):
    assert (len(point_list)%2), 'Number of points must be odd, or there will not be a unique solution.'
    thetas = []
    for n in range(len(point_list) - 1):
        thetas.append(angle(point_list[n], point_list[n+1]))
    thetas.append(angle(point_list[::-1][0], point_list[0]))

    rows = []
    target = []
    for n in range(len(thetas) - 1):
        row = []
        for i in range(n):
            row.append(0)
        if n in angles:
            row += [1]
            target.append([-pi])
        else:
            row += [1,1]
            target.append([thetas[n]- thetas[n+1]])
        while len(row) < len(thetas):
            row.append(0)
        rows.append(row)

    rows.append([1] + [0]*(len(thetas)-2) + [1])
    target.append([thetas[::-1][0] - thetas[0]])

    coefs = np.array(rows)

    phis = np.linalg.solve(coefs, np.array(target))

    def acute(angle):
        while angle > pi/2:
            angle -= pi
        while angle < -pi/2:
            angle += pi
        return angle

    def correct(angle):
        if solution:
            angle = angle%(2*pi)
            if angle > pi:
                angle -= 2*pi
        return angle

    phis = [correct(a[0] + pi) for a in phis]

    switch = False
    connect = Surface()

    for n in range(len(point_list)):

        t = connect.state.iloc[::-1].iloc[0]['t'] + dt

        x, y = point_list[n-1]

        theta = thetas[n-1] + phis[n-1]
        if phis[n-1] == 0:
            r = 10000
        else:
            r = -distance(point_list[n-1], point_list[n])/(2*sin(phis[n-1]))
        connect.state = pd.concat([connect.state, pd.DataFrame({'x':[x], 'y':[y], 't':[t], 'v':1, 'r':[r], 'value':[1], 'theta':[theta]})])
        if phis[n-1] == 0:
            for i in range(int(distance(point_list[n-1], point_list[n])//dt)):
                connect.plus(dt)
        else:
            for i in range(int((abs(r*2*phis[n-1])//dt))):
                connect.plus(dt)
    connect.state = connect.state.iloc[1:]
    connect.plot(plt)
    plt.scatter([p[0] for p in point_list], [p[1] for p in point_list])
    plt.axis('square')
    plt.show()

    return connect



class Flow:
    def __init__(self, freestream = (0,0), surfaces = []):
        self.surfaces = surfaces
        self.freestream = freestream
        self.vortex_centers = []
        self.test_points = []
        for s in self.surfaces:
            for v in s.vortex_centers:
                self.vortex_centers.append(v)
            for tp in s.test_points:
                self.test_points.append(tp)

    def add_surface(self, surface):
        self.surfaces.append(surface)
        surface.set_freestream(self.freestream)
        self.vortex_centers += surface.vortex_centers
        self.test_points += surface.test_points

    def set_freestream(self, freestream):
        self.freestream = freestream
        for s in self.surfaces:
            s.set_freestream(freestream)

    def vortex_predict(self):
        """
        Computes strength coefficients for vortices with centers specified
        such that the flow is parallel to all surfaces at all specified test
        points.
        """

        print('Computing coefficients for {} vortex centers...'.format(str(len(self.vortex_centers))) )

        vc = self.vortex_centers
        tp = self.test_points
        print(vc)
        print(tp)

        rows = []
        b = []
        for p in tp:
            row = []
            m = tan(p['theta'])
            com = (m*p['y'] + p['x'])
            for c in vc:
                row.append( (-c[0] - m*c[1] + com)/((p['y'] - c[1])**2 + (p['x'] - c[0])**2) )
            rows.append(row)
            b.append([m*self.freestream[0] - self.freestream[1]])
        A, b = np.array(rows), np.array(b)
        print(A,b)
        self.vortex_coefficients = np.linalg.solve(A, b)
        k = self.vortex_coefficients
        print('Done\nBelow are the coefficients:')
        print(k, '\n')

        print('Plotting.')
        for s in self.surfaces:
            s.plot(plt)
        plt.axis('square')

        for p in self.test_points:
            xx, yy = self.freestream[0]/10,self.freestream[1]/10
            for i in range(len(vc)):
                c = vc[i]
                xx += k[i]*(c[1] - p['y'])/((p['y'] - c[1])**2 + (p['x'] - c[0])**2)/10
                yy += k[i]*(p['x'] - c[0])/((p['y'] - c[1])**2 + (p['x'] - c[0])**2)/10
            plt.plot([p['x'], p['x'] + xx], [p['y'], p['y'] + yy])
        plt.show()

        """
        print('computing velocity distribution')
        for t in self.state['t']:
            row = self.state.loc[self.state['t'] == t]
            x, y = row['x'], row['y']
            xx, yy = self.freestream[0], self.freestream[1]
            for i in range(len(vc)):
                c = vc[i]
                xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))
                yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))
            self.state.loc[self.state['t'] == t, 'value'] = sqrt( xx**2 + yy**2 )

        plt.plot(self.state['t'], self.state['value'])
        plt.show()
        """

        print('\n')

    def vortex_plot(self, resolution = res):
        vc, k = self.vortex_centers, self.vortex_coefficients
        assert vc, 'No vortex centers.  Try calling Surface.vortex_centers()'
        assert k.any(), 'No vortex coefficients to generate velocity vector field.  Try calling Surface.vortex_predict()'

        ymin, ymax = min([min(s.state['y']) for s in self.surfaces]), max([max(s.state['y']) for s in self.surfaces])
        xmin, xmax = min([min(s.state['x']) for s in self.surfaces]), max([max(s.state['x']) for s in self.surfaces])

        ymin_bound, ymax_bound = int(ymin - 2 * (ymax - ymin)) - 1, int(ymax + 2 * (ymax - ymin)) + 1
        xmin_bound, xmax_bound = int(xmin - 1 * (xmax - xmin)) - 1, int(xmax + 2 * (xmax - xmin)) + 1

        for y in range(ymin_bound, ymax_bound):

            print('Generating streamline {}'.format(str(y)) )
            x,y = xmin_bound,y
            X,Y = [],[]
            while x < xmax_bound:
                X.append(x)
                Y.append(y)
                xx, yy = self.freestream[0], self.freestream[1]
                for i in range(len(vc)):
                    c = vc[i]
                    xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))
                    yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))
                veloc = sqrt(xx**2 + yy**2)
                xx /= veloc*resolution
                yy /= veloc*resolution
                x += xx
                y += yy
            plt.plot(X,Y, color = 'blue')

        for s in self.surfaces:
            s.plot(plt)
        plt.axis('equal')
        plt.show()

    def plot_streams(self, start_points, resolution = res):
        vc, k = self.vortex_centers, self.vortex_coefficients
        assert vc, 'No vortex centers.  Try calling Surface.vortex_centers()'
        assert k.any(), 'No vortex coefficients to generate velocity vector field.  Try calling Surface.vortex_predict()'

        xmin, xmax = min([min(s.state['x']) for s in self.surfaces]), max([max(s.state['x']) for s in self.surfaces])
        xmin_bound, xmax_bound = int(xmin - 1 * (xmax - xmin)) - 1, int(xmax + 2 * (xmax - xmin)) + 1

        for point in start_points:
            x,y = point[0], point[1]
            X,Y = [],[]
            while x < xmax_bound:
                X.append(x)
                Y.append(y)
                xx, yy = self.freestream[0], self.freestream[1]
                for i in range(len(vc)):
                    c = vc[i]
                    xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))
                    yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))
                veloc = sqrt(xx**2 + yy**2)
                xx /= veloc*resolution
                yy /= veloc*resolution
                x += xx
                y += yy
            plt.plot(X,Y, color = 'blue')

        for s in self.surfaces:
            s.plot(plt)
        plt.axis('equal')
        plt.show()

    def vortex_get_velocity(self, coords):
        vc, k = self.vortex_centers, self.vortex_coefficients
        x, y = coords[0], coords[1]
        xx, yy = self.freestream[0], self.freestream[1]
        for i in range(len(vc)):
            c = vc[i]
            xx += float(k[i]*(c[1] - y)/((y - c[1])**2 + (x - c[0])**2))
            yy += float(k[i]*(x - c[0])/((y - c[1])**2 + (x - c[0])**2))
        veloc = sqrt(xx**2 + yy**2)
        return veloc

    def vortex_cl(self):
        lift = 0
        for s in self.surfaces:
            l = 0
            for i in range(len(s.state)):
                row = s.state.iloc[i]
                q = self.vortex_get_velocity((row['x'], row['y']))**2
                l += q*cos(row['theta'])
            l /= len(s.state)
            lift += l
        lift /= (len(self.surfaces)*(self.freestream[0]**2 + self.freestream[1]**2))
        return lift

    def vortex_cd(self):
        lift = 0
        for s in self.surfaces:
            l = 0
            for i in range(len(s.state)):
                row = s.state.iloc[i]
                q = self.vortex_get_velocity((row['x'], row['y']))**2
                l -= q*sin(row['theta'])
            l /= len(s.state)
            lift += l
        lift /= (len(self.surfaces)*(self.freestream[0]**2 + self.freestream[1]**2))
        return lift

    def show_surfaces(self):
        for s in self.surfaces:
            s.plot(plt)
        plt.axis('square')
        plt.show()

