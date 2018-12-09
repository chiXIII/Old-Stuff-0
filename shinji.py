from flow2D import *

point_list = [(14.04, 1.68), (13.18, 1.7), (12.55, 1.72), (12.00, 1.75), (11.60, 1.77), (10.62, 1.85), (9.8, 1.9), (8.9, 1.97), (7.79, 2.04), (7.07, 2.07), (6.63, 2.08), (5.83, 2.08), (4.70, 2.07), (4.23, 2.06), (3.62, 1.99), (3.43, 1.79), (3.50,1.46), (3.73, 1.20), (4.17, 0.93), (4.75, 0.71), (5.68, 0.52), (6.95, 0.45), (9.35, 0.61), (13.19, 1.31), (15.1,1.69)]
point_list = [(p[0], 1 - p[1]) for p in point_list]
airfoil = surface_from_points(point_list, angles = [len(point_list) - 2])
airfoil.set_freestream((1,0))
airfoil.rotate(-pi/12)
airfoil.add_curvature(pi/16)

print(len(airfoil.state))

deltax = airfoil.state.iloc[0]['x'] - airfoil.state.iloc[::-1].iloc[0]['x']
deltay = airfoil.state.iloc[0]['y'] - airfoil.state.iloc[::-1].iloc[0]['y']

for t in airfoil.state.iloc[700:800]['t']:
    airfoil.state.loc[airfoil.state['t'] == t, 'x'] += deltax/4
    airfoil.state.loc[airfoil.state['t'] == t, 'y'] += deltay/4
for t in airfoil.state.iloc[800:900]['t']:
    airfoil.state.loc[airfoil.state['t'] == t, 'x'] += 2*deltax/4
    airfoil.state.loc[airfoil.state['t'] == t, 'y'] += 2*deltay/4
for t in airfoil.state.iloc[900:]['t']:
    airfoil.state.loc[airfoil.state['t'] == t, 'x'] += 3*deltax/4
    airfoil.state.loc[airfoil.state['t'] == t, 'y'] += 3*deltay/4

airfoil.vortex_points()

"""
for n in range(3):
    afoil = [airfoil1, airfoil2, airfoil3][n]
    afoil.translate(((n+1)*deltax/3,(n+1)*deltay/3))

airfoil1 = airfoil.copy()
airfoil1.translate((9, -3))
airfoil.rotate(pi/16)

for afoil in [airfoil, airfoil1, airfoil2, airfoil3]:
    afoil.vortex_points()
"""

#air = Flow(freestream = (1,0), surfaces = [airfoil, airfoil1, airfoil2, airfoil3])
air = Flow(freestream = (1,0), surfaces = [airfoil])
air.show_surfaces()

air.vortex_predict()
print(air.vortex_cl())
#air.vortex_plot()
air.plot_streams([(7.83, -1.68), (10.25, -2.65), (12.63, -3.63), (14.50, -4.45)])
