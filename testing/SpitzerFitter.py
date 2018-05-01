import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import batman
import emcee
import corner
import os
import sys



# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################
targ_name = 'K2-36'
epic = 201713348
save_plots = False	# Save plots?
show_plots = True	# Show plots?
annotate_plot = True

bjd_offset = 57460.		# Makes nicer plots
utc2bjd = 0.006465959	# TDB correction from http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html

# Define initial planet parameters from Crossfield et al. 2016
param_names = ["$t_0$", "$rp$", "$a$", "$inc$", "$offset$", "$\sigma_f$"]
t0_i = 57464.45		# BJD
rp_i = 0.0323		# Rp/R*
per_i = 5.34		# days \FIXED\ (will be used to set prior on a)
inc_i = 85.			# degrees
offset_i = 0.00005	# offset added to model
sigma_i = 0.001		# flux uncertainty

# Stellar parameters from Crossfield et al. 2016
Rstar = 0.75		# Rsun
Mstar = 0.81		# Msun

# MCMC parameters
nsteps = 5000
burn_in = 750
ndim = 6
nwalkers = 100
##########################################################

# eventually should change prior on 'a' to a Gaussian prior
# for now, just say 30% of calculated value is acceptable
a_i = (((per_i/365.)**2) / Mstar)**(1./3.) * Rstar * 215.035216		# semimajor axis from period and stellar params

# Priors.
def lnprior(theta, t0_i=t0_i, a_i=a_i):
	t0, rp, a, inc, offset, sigma = theta
	if t0_i-0.1 < t0 < t0_i+0.1 \
	and 0.01 < rp < 0.1 \
	and 0.7*a_i < a < 1.3*a_i \
	and 0. < inc < 90. \
	and 0.0 < offset < 0.001 \
	and 0.0001 < sigma < 0.1:
		return 0.0
	return -np.inf

# Import data from text file
data = []
with open('PhotData//%s_Spitzer.txt' % targ_name) as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		data.append(data_flt)
data = np.asarray(data)
data = data[data[:,0].argsort()]

t = []
f = []

for i in data[:,0]:
	t.append(i + utc2bjd)
for i in data[:,1]:
	f.append(i)

t = np.asarray(t)
f = np.asarray(f)

# Define log of the likelihood function.
def lnlike(theta, x, y, per=per_i):
	t0, rp, a, inc, offset, sigma = theta
	# Set up transit parameters.
	params = batman.TransitParams()
	params.t0 = t0
	params.per = per 	# fixed parameter
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = 0.
	params.w = 0.
	params.u = []
	params.limb_dark = 'uniform'
	# Initialize the transit model.
	m_init = batman.TransitModel(params, x)
	model = m_init.light_curve(params) + offset
	inv_sigma2 = 1.0/(sigma**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(2*np.pi*inv_sigma2)))

# Define log of probability function.
def lnprob(theta, x, y):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y)


initial_params = t0_i, rp_i, a_i, inc_i, offset_i, sigma_i

# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, f))

# Run MCMC for n steps and display progress bar.
width = 50
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
	n = int((width+1) * float(i) / nsteps)
	sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
sys.stdout.write("\n")
print 'Sampling complete!'

samples = sampler.chain

# Plot walkers as a function of step number.
walker_fig, ax = plt.subplots(ndim, sharex=True, figsize=(10, 8))
ax[-1].set_xlabel('Iteration')
for n in range(ndim):
    for k in range(nwalkers):
        ax[n].plot(samples[k, :, n], color='k', alpha=0.1, lw=1)
    ax[n].set_ylabel((param_names)[n], fontsize=9)
    ax[n].margins(0, None)
    ax[n].axvline(burn_in, color='b', alpha=0.5, lw=1, ls='--')

# Discard burn-in. 
samples = samples[:, burn_in:, :].reshape((-1, ndim))

# Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
samples[:, ndim-1] = np.exp(samples[:, ndim-1])
t0_mcmc, rp_mcmc, a_mcmc, inc_mcmc, offset_mcmc, sigma_mcmc = map(
	lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

# create array for flux uncert
ferr = []
for i in data[:,1]:
	ferr.append(sigma_mcmc[0] - 1.)
ferr = np.asarray(ferr)

# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = t0_mcmc[0]
params_final.per = per_i
params_final.rp = rp_mcmc[0]
params_final.a = a_mcmc[0]
params_final.inc = inc_mcmc[0]
params_final.ecc = 0.
params_final.w = 0.
params_final.u = []
params_final.limb_dark = "uniform"
tl = np.linspace(min(t),max(t),1000)
m = batman.TransitModel(params_final, tl)
m_res = batman.TransitModel(params_final, t)

f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title('%s (EPIC %s)' % (targ_name, str(epic)))
ax.errorbar(t-bjd_offset,f + offset_mcmc[0],yerr=ferr,fmt='k.',capsize=0,alpha=0.4)
ax.plot(tl-bjd_offset, f_final, 'r-', lw=3)
if annotate_plot == True:
	ant = AnchoredText('$T_0 = %s^{+%s}_{-%s}$ \n $R_p/R_* = %s^{+%s}_{-%s}$' % (2400000+round(t0_mcmc[0],4),round(t0_mcmc[1],4),
	round(t0_mcmc[2],4),round(rp_mcmc[0],4),round(rp_mcmc[1],4),round(rp_mcmc[2],4)), prop=dict(size=11), frameon=True, loc=3)
	ant.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
	ax.add_artist(ant)
ax.set_xlabel("$BJD_{TDB} - 24%s$" % str(bjd_offset))
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','Spitzer'), loc=2)

f_res = m_res.light_curve(params_final) + offset_mcmc[0]
resid_fig, axr = plt.subplots(figsize=(10,8))
axr.errorbar(t-bjd_offset,f-f_res,yerr=ferr,fmt='k.',capsize=0,alpha=0.4)
axr.plot(t-bjd_offset, np.zeros(len(t)), 'r-', lw=3)
axr.set_xlabel("$BJD_{TDB} - 24%s$" % str(bjd_offset))
axr.set_ylabel("Residual")

corn_fig = corner.corner(samples, labels=param_names)

if save_plots == True:
	save_to = 'fig//Spitzer_plots//%s' % targ_name
	if not os.path.exists(save_to):
		os.makedirs(save_to)
	final_fig.savefig('%s//%s_MCMCfit.png' % (save_to,targ_name), bbox_inches='tight')
	corn_fig.savefig('%s//%s_corner.png' % (save_to,targ_name), bbox_inches='tight')
	walker_fig.savefig('%s//%s_walkers.png' % (save_to,targ_name), bbox_inches='tight')
	resid_fig.savefig('%s//%s_residuals.png' % (save_to,targ_name), bbox_inches='tight')

print 't0: ',t0_mcmc
print 'rp: ',rp_mcmc
print 'a: ',a_mcmc
print 'inc: ',inc_mcmc
print 'offset: ',offset_mcmc
print 'sigma: ',sigma_mcmc


if show_plots == True:
	plt.show()
