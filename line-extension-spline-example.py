import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

# http://stackoverflow.com/questions/20643637/extend-line-to-smoothly-connect-with-another-line

# Red line data.
x1 = [0.01, 0.04, 0.08, 0.11, 0.15, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44, 0.46, 0.49, 0.51, 0.54, 0.56, 0.58]
y1 = [2.04, 2.14, 2.24, 2.34, 2.44, 2.54, 2.64, 2.74, 2.84, 2.94, 3.04, 3.14, 3.24, 3.34, 3.44, 3.54, 3.64, 3.74, 3.84, 3.94]

# Blue line data.
x2 = [0.4634, 0.4497, 0.4375, 0.4268, 0.4175, 0.4095, 0.4027, 0.3971, 0.3925, 0.389, 0.3865, 0.3848, 0.384, 0.3839, 0.3845, 0.3857, 0.3874, 0.3896, 0.3922, 0.3951, 0.3982, 0.4016, 0.405, 0.4085, 0.412, 0.4154, 0.4186, 0.4215, 0.4242, 0.4265, 0.4283, 0.4297, 0.4304, 0.4305, 0.4298, 0.4284, 0.4261, 0.4228, 0.4185, 0.4132, 0.4067, 0.399, 0.39, 0.3796, 0.3679, 0.3546, 0.3397, 0.3232, 0.305, 0.285]
y2 = [1.0252, 1.0593, 1.0934, 1.1275, 1.1616, 1.1957, 1.2298, 1.2639, 1.298, 1.3321, 1.3662, 1.4003, 1.4344, 1.4685, 1.5026, 1.5367, 1.5708, 1.6049, 1.639, 1.6731, 1.7072, 1.7413, 1.7754, 1.8095, 1.8436, 1.8776, 1.9117, 1.9458, 1.9799, 2.014, 2.0481, 2.0822, 2.1163, 2.1504, 2.1845, 2.2186, 2.2527, 2.2868, 2.3209, 2.355, 2.3891, 2.4232, 2.4573, 2.4914, 2.5255, 2.5596, 2.5937, 2.6278, 2.6619, 2.696]

x3, y3 = [], []

# Store a small section of the blue line in these new lists: only those points
# closer than 0.2 to the last point in this line.
for indx,y2_i in enumerate(y2):
    if (y2[-1]-y2_i)<=0.2:
        y3.append(y2_i)
        x3.append(x2[indx])

# The same as above but for the red line: store only those points between
# 0. and 0.4 in the y axis and with a larger x value than the last point in the
# blue line.
for indx,y1_i in enumerate(y1):
    if 0. <(y1_i-y2[-1])<=0.4 and x1[indx] > x2[-1]:
        y3.append(y1_i)
        x3.append(x1[indx])

sx = np.array(x2+x3)
sy = np.array(y2+y3)
t  = np.arange(sx.size,dtype=float)
t /= t[-1]
N  = np.linspace(0,1,2000)
SX = spline(t,sx,N,order=4)
SY = spline(t,sy,N,order=4)

plt.plot(x1,y1, 'r^')
plt.plot(x2,y2, 'b*')
plt.show(block=False)
plt.scatter(x3,y3,c='k')
plt.show(block=False)
plt.plot(SX, SY,'g',alpha=.7,lw=3)
plt.show()