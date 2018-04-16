import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t = []
f = []
ferr = []

with open('test_data//epic201367065_real.txt') as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		t.append(data_flt[0])
		f.append(data_flt[1])
		ferr.append(data_flt[2])

plt.errorbar(t, f, ferr, fmt='b.')


params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 1.5                     #orbital period
params.rp = 0.035                    #planet radius (in units of stellar radii)
params.a = 4.                        #semi-major axis (in units of stellar radii)
params.inc = 88.                     #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 90.                       #longitude of periastron (in degrees)
params.u = []                #limb darkening coefficients [u1, u2]
params.limb_dark = "uniform"       #limb darkening model

tl = np.linspace(-0.5, 0.5, 5000)

m = batman.TransitModel(params, tl)    #initializes model
f_bm = m.light_curve(params)           #calculates light curve



plt.plot(tl, f_bm, 'r-')

plt.title('EPIC 201367065')
plt.xlabel("$T-T_0$")
plt.ylabel("Relative Flux")
plt.legend(('model','data'), loc=3)
plt.show()

