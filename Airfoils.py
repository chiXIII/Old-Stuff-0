import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd






############# S8064 ############




alpha = np.array([-6.15,-5.11,-4.07,-3.10,-2.06,-1.03,-0.05,1.04,2.04,3.09,4.09,5.12,6.18,7.20,8.20,9.18,10.21,11.24])

Cd = np.array([0.0264,0.0205,0.0170,0.0187,0.0182, 0.0194, 0.0206, 0.0205, 0.0233, 0.0261, 0.0229, 0.0189, 0.0161, 0.0176, 0.0248, 0.0317, 0.0392, 0.0527])

data100k = np.array([alpha, Cd]).transpose()
table100k = pd.DataFrame(data = data100k, columns = ['a', 'Cd'])

p100k = np.polyfit(alpha, Cd, 6)

df100k = np.poly1d(p100k)
xp = np.linspace(-7,12,100)
plt.plot(alpha,Cd, '.', xp, df100k(xp), '-')


alpha = np.array([-7.14,-6.10, -5.12, -4.05, -3.10, -2.04, -0.98, -0.04, 0.98, 2.06, 3.13, 4.12, 5.15, 6.19, 7.13, 8.17, 9.23, 10.25, 11.19])

Cd = np.array([0.0238, 0.0187, 0.0151, 0.0123,0.0130, 0.0133, 0.0142, 0.0145, 0.0151, 0.0143, 0.0128, 0.0116, 0.0100, 0.0126, 0.0161, 0.0214, 0.0264, 0.0333, 0.0434])

data200k = np.array([alpha, Cd]).transpose()
table200k = pd.DataFrame(data = data200k, columns = ['a', 'Cd'])

p200k = np.polyfit(alpha, Cd, 6)
df200k = np.poly1d(p200k)
plt.plot(alpha,Cd, '.', xp, df200k(xp), '-')

alpha = np.array([-7.16, -6.12, -5.02, -4.07, -3.01, -1.98, -1.03, 0.06, 1.05, 2.05, 3.08, 4.09, 5.15, 6.21, 7.15, 8.22, 9.25, 10.24, 11.24])

Cd = np.array([0.0195, 0.0163, 0.0140, 0.0114, 0.0108, 0.0109, 0.0110, 0.0109, 0.0104, 0.0099, 0.0087, 0.0087, 0.0097, 0.0123, 0.0152, 0.0199, 0.0249, 0.0315, 0.0415])

data300k = np.array([alpha, Cd]).transpose()
table300k = pd.DataFrame(data = data300k, columns = ['a', 'Cd'])

p300k = np.polyfit(alpha, Cd, 6)
df300k = np.poly1d(p300k)
plt.plot(alpha,Cd, '.', xp, df300k(xp), '-')

alpha = np.array([-7.61, -6.63, -5.53, -4.56, -3.47, -2.54, -1.51, -0.49, 0.53, 1.57, 2.58, 3.63, 4.67, 5.66, 6.71, 7.72, 8.77, 9.71, 10.78, 11.75])

Cd = np.array([0.0200, 0.0170, 0.0144, 0.0124, 0.0098, 0.0093, 0.0090, 0.0089, 0.0085, 0.0080, 0.0076, 0.0077, 0.0087, 0.0110, 0.0136, 0.0169, 0.0212, 0.0262, 0.0337, 0.0458])

p400k = np.polyfit(alpha, Cd, 6)
data400k = np.array([alpha, Cd]).transpose()
df400k = np.poly1d(p400k)
plt.plot(alpha,Cd, '.', xp, df400k(xp), '-')


alpha = np.array([-8.09, -7.07, -6.08, -5.12, -4.01, 03.01, -1.97, -1.07, 0.03, 1.09, 2.03, 3.09, 4.16, 5.18, 6.18, 7.19, 8.26, 9.27, 10.28, 11.28, 12.25])

Cd = np.array([0.0206, 0.0168, 0.0146, 0.0126, 0.0106, 0.0087, 0.0081, 0.0077, 0.0074, 0.0072, 0.0069, 0.0069, 0.0077, 0.0087, 0.0119, 0.0147, 0.0183, 0.0223, 0.0271, 0.0346, 0.0503])

p500k = np.polyfit(alpha, Cd, 6)
df500k = np.poly1d(p500k)
plt.plot(alpha,Cd, '.', xp, df500k(xp), '-')



Re = np.array([99850.6, 200214.1, 299862.0, 399890.6, 499444.3])
lst = []
for i in range(7):
    a = np.array([p100k[i], p200k[i], p300k[i], p400k[i], p500k[i]])
    lst += [a]

r = np.linspace(0, 500000, 100)
coeffunctions = []
for i in range(7):
    f = np.poly1d(np.polyfit(Re, lst[i], 4))
    coeffunctions.append(f)


def drag(Re, alpha):
    return coeffunctions[0](Re) * alpha**6 + coeffunctions[1](Re)* alpha**5 + coeffunctions[2](Re) * alpha**4 + coeffunctions[3](Re) * alpha**3 + coeffunctions[4](Re) * alpha**2 + coeffunctions[5](Re) * alpha + coeffunctions[6](Re)

for re in [100000, 200000, 300000, 400000, 500000]:
    plt.plot(xp, drag(re, xp), '--')

S8064dat = {'Cd' : drag, 'slope': 0.1, 'a0': 0}








############# AG12 ############




alpha = np.array([-3.05, -1.98, -1.49, -0.96, -0.45, 0.02, 1.07, 2.10, 3.15, 4.24, 5.25, 6.31, 7.24, 8.29, 9.28, 10.31, 11.26])

Cd = np.array([0.0200, 0.0123, 0.0131, 0.0136, 0.0128, 0.0132, 0.0158, 0.0116, 0.0175, 0.0248, 0.0271, 0.0395, 0.0606, 0.0642, 0.1083, 0.1193, 0.1743])

data40k = np.array([alpha, Cd]).transpose()
table40k = pd.DataFrame(data = data40k, columns = ['a', 'Cd'])

p40k = np.polyfit(alpha, Cd, 6)

df40k = np.poly1d(p40k)
xp = np.linspace(-7,12,100)
plt.plot(alpha,Cd, '.', xp, df40k(xp), '-')

alpha = np.array([-3.01, -1.97, -1.54, -1.03, -0.46, 0.04, 1.12, 2.11, 3.21, 4.16, 5.24, 6.23, 7.24, 8.23, 9.30, 10.27, 11.32])

Cd = np.array([0.0138, 0.0127, 0.0112, 0.0105, 0.0122, 0.0116, 0.0119, 0.0148, 0.0159, 0.0184, 0.0204, 0.0248, 0.0275, 0.0402, 0.0676, 0.1340, 0.1693])

data60k = np.array([alpha, Cd]).transpose()
table60k = pd.DataFrame(data = data60k, columns = ['a', 'Cd'])

p60k = np.polyfit(alpha, Cd, 6)
df60k = np.poly1d(p60k)
p60k = np.polyfit(alpha, Cd, 6)
plt.plot(alpha, Cd, '.', xp, df60k(xp), '-')

alpha = np.array([-3.05, -2.03, -1.48, -0.91, -0.46, 0.06, 1.09, 2.17, 3.16, 4.22, 5.29, 6.20, 7.25, 8.22, 9.23, 10.24, 11.27])

Cd = np.array([0.0139, 0.0116, 0.0096, 0.0095, 0.0101, 0.0103, 0.0110, 0.0125, 0.0143, 0.0160, 0.0171, 0.0203, 0.0266, 0.0325, 0.0496, 0.1344, 0.1670])

data80k = np.array([alpha, Cd]).transpose()
table80k = pd.DataFrame(data = data80k, columns = ['a', 'Cd'])
p80k = np.polyfit(alpha, Cd, 6)

df80k = np.poly1d(p80k)
p80k = np.polyfit(alpha, Cd, 6)
plt.plot(alpha, Cd, '.', xp, df80k(xp), '-')
alpha = np.array([-3.04, -2.00, -1.42, -0.92, -0.43, 0.06, 1.07, 2.21, 3.17, 4.11, 5.17, 6.20, 7.30, 8.25, 9.24, 10.25, 11.23])

Cd = np.array([0.0131, 0.0095, 0.0086, 0.0098, 0.0098, 0.0083, 0.0110, 0.0117, 0.0131, 0.0151, 0.0180, 0.0211, 0.0259, 0.0367, 0.0590, 0.1355, 0.1641])

p100k = np.polyfit(alpha, Cd, 6)
df100k = np.poly1d(p100k)
plt.plot(alpha,Cd, '.', xp, df100k(xp), '-')

alpha = np.array([-3.02, -1.99, -1.45, -0.90, -0.43, 0.06, 1.08, 2.11, 3.14, 4.12, 5.20, 6.21, 7.29, 8.27, 9.21, 10.19, 11.17])

Cd = np.array([0.0116, 0.0091, 0.0081, 0.0074, 0.0084, 0.0088, 0.0101, 0.0088, 0.0102, 0.0118, 0.0138, 0.0169, 0.0217, 0.0288, 0.0422, 0.1332, 0.1639])

data150k = np.array([alpha, Cd]).transpose()
table150k = pd.DataFrame(data = data150k, columns = ['a', 'Cd'])

p150k = np.polyfit(alpha, Cd, 6)
df150k = np.poly1d(p150k)
plt.plot(alpha,Cd, '.', xp, df150k(xp), '-')




Re = np.array([39617.3, 59971.8, 79922.2, 100021.3, 149943.1])
lst = []
for i in range(7):
    a = np.array([p40k[i], p60k[i], p80k[i], p100k[i], p150k[i]])
    lst += [a]

r = np.linspace(0, 200000, 100)
coeffunctions = []
for i in range(7):
    f = np.poly1d(np.polyfit(Re, lst[i], 4))
    coeffunctions.append(f)


def drag(Re, alpha):
    return coeffunctions[0](Re) * alpha**6 + coeffunctions[1](Re)* alpha**5 + coeffunctions[2](Re) * alpha**4 + coeffunctions[3](Re) * alpha**3 + coeffunctions[4](Re) * alpha**2 + coeffunctions[5](Re) * alpha + coeffunctions[6](Re)

for re in [40000, 60000, 80000, 100000, 150000]:
    plt.plot(xp, drag(re, xp), '--')


AG12dat = {'Cd' : drag, 'slope': 0.1, 'a0': -2}
