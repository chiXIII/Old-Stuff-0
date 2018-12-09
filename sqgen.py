l1, l2 = [-1,-1,-1,-1],[0,0.02,0.04,0.06]
x,y = -1,0.06
for n in range(50):
    x += 0.02
    l1.append(x)
    l2.append(y)
l1 += [0,0,0]
l2 += [0.04,0.02,0]
y = 0
for n in range(50):
    x -= 0.02
    l1.append(x)
    l2.append(y)
text = ''
for n in range(len(l1)):
    text += str(l1[n]) + ' ' + str(l2[n]) +'\n'
file = open('rect.txt', 'w+')
file.write(text)
file.close()
