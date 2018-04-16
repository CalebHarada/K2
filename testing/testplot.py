import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-50,50,1000)

def func1(xvals):
	yvals = np.sin(xvals)/xvals
	return yvals

y = func1(x)

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)
ax1.set_xlim((-10*np.pi,10*np.pi))
ax1.set_title('sinc(x)')
ax1.plot(x,y,'m-')

plt.show()

