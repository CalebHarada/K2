import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import batman
import emcee
import corner

# Read the data.
data = []
with open('testing//test_data//epic201367065_real.txt') as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		data.append(data_flt)
data = np.asarray(data)
data = data[data[:,0].argsort()]
t = []									# Idk why this step is necessary, but Batman makes a weird LC otherwise.
for i in data[:,0]:
	t.append(i)
t = np.asarray(t)
f = []
for i in data[:,1]:
	f.append(i)
f = np.asarray(f)
ferr = []
for i in data[:,2]:
	ferr.append(i)
ferr = np.asarray(ferr)

# Define some initial guesses.
t0_init = 0.                       #time of inferior conjunction
per_init = 1.5                     #orbital period
rp_init = 0.035                    #planet radius (in units of stellar radii)
a_init = 5.                        #semi-major axis (in units of stellar radii)
inc_init = 88.                     #orbital inclination (in degrees)
ecc_init = 0.                      #eccentricity
w_init = 90.                       #longitude of periastron (in degrees)
u1_init = 0.2                      #limb darkening coefficient u1
u2_init = 0.4               	   #limb darkening coefficient u2
f_init = 0.1					   #uncertainties underestimated by this fraction

# Define log of the likelihood function.
def lnlike(theta, x, y, yerr):
	t0, per, rp, a, inc, ecc, w, u1, u2, lnf = theta
	# Set up transit parameters.
	params = batman.TransitParams()
	params.t0 = t0
	params.per = per
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = ecc
	params.w = w
	params.u = [u1, u2]
	params.limb_dark = 'quadratic'
	# Initialize the transit model.
	m_init = batman.TransitModel(params, x)
	model = m_init.light_curve(params)
	inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Set up priors.
def lnprior(theta):
    t0, per, rp, a, inc, ecc, w, u1, u2, lnf = theta
    if -0.05 < t0 < 0.05 and 0.0 < per < 800.0 and 0.0 < rp < 1.0 \
    and 0.0 < a < 500.0 and 85.0 < inc < 95.0 and 0.0 < ecc < 1.0 \
    and 0.0 < w < 360.0 and 0.0 < u1 < 1.0 and 0.0 < u2 < 1.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Log probability function.
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Maximize the likelihood function.
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [t0_init, per_init, rp_init, a_init, inc_init, ecc_init, w_init, u1_init, u2_init, np.log(f_init)], args=(t, f, ferr))
t0_ml, per_ml, rp_ml, a_ml, inc_ml, ecc_ml, w_ml, u1_ml, u2_ml, lnf_ml = result["x"]

# Initialize walkers around maximum likelihood.
ndim, nwalkers = 10, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, f, ferr))

# Run the MCMC for 1000 steps, starting from initial conditions defined above.
sampler.run_mcmc(pos, 1100)

# Discard the first 100 steps. 
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))


# Plot some MCMC results.
tl = np.linspace(-.5,.5,1000)
plt.figure(figsize=(10,8))
plt.errorbar(t,f,yerr=ferr,fmt='k.',capsize=3)
for t0, per, rp, a, inc, ecc, w, u1, u2, lnf in samples[np.random.randint(len(samples), size=100)]:
	# Create a model with the maximum likelihood parameters.
	params = batman.TransitParams()
	params.t0 = t0
	params.per = per
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = ecc
	params.w = w
	params.u = [u1, u2]
	params.limb_dark = "quadratic"
	m_final = batman.TransitModel(params, tl)
	f_mcmc = m_final.light_curve(params)
	plt.plot(tl, f_mcmc, 'k-', alpha=0.1)
plt.xlim((-0.5,0.5))
plt.xlabel("$T-T_0$")
plt.ylabel("Relative Flux")
plt.title('Random MCMC Samples')

# Uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
samples[:, 9] = np.exp(samples[:, 9])
t0_mcmc, per_mcmc, rp_mcmc, a_mcmc, inc_mcmc, ecc_mcmc, w_mcmc, u1_mcmc, u2_mcmc, f_mcmc = map(
	lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = t0_mcmc[0]
params_final.per = per_mcmc[0]
params_final.rp = rp_mcmc[0]
params_final.a = a_mcmc[0]
params_final.inc = inc_mcmc[0]
params_final.ecc = ecc_mcmc[0]
params_final.w = w_mcmc[0]
params_final.u = [u1_mcmc[0], u2_mcmc[0]]
params_final.limb_dark = "quadratic"
m = batman.TransitModel(params_final, tl)
f_final = m.light_curve(params_final)
plt.figure(figsize=(10,8))
plt.errorbar(t,f,yerr=ferr,fmt='k.',capsize=3)
plt.plot(tl, f_final, 'r-')
plt.title('EPIC 201367065 light curve')
plt.xlabel("$T-T_0$")
plt.ylabel("Relative Flux")
plt.legend(('model','data'), loc=1)
plt.xlim((-0.5,0.5))
plt.show()

# Make a corner plot of posteriors.
# fig = corner.corner(samples, labels=["$t_0$", "$per$", "$rp$", "$a$", "$inc$", "$ecc$", "$w$", "$u1$", "$u2$", "$\ln\,f$"])

print "time of inferior conjunction: ", t0_mcmc
print "orbital period: ", per_mcmc
print "planet radius (in units of stellar radii): ", per_mcmc
print "semi-major axis (in units of stellar radii): ", a_mcmc
print "orbital inclination (in degrees): ", inc_mcmc
print "eccentricity: ", ecc_mcmc
print "longitude of periastron (in degrees): ", w_mcmc
print "limb darkening coefficients, u1 and u2: ", u1_mcmc, u2_mcmc
print "f: ", f_mcmc

plt.show()

