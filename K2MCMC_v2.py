# v2 -- timing, save results, save phased LC, resolved issue of sigma > 1, naming scheme


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import batman
from corner import corner
from everest import Everest, TransitModel
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import sys, os
import time as systime



# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################
targ_name = 'K2-79'
epic = 210402237
save_results = True	# Save results?
save_phased = True	# Save phased LC data?
show_plots = False	# Show plots?
annotate_plot = True

BJD_offset = 2454833	# Get this by plotting the full Everest light curve for plotting

# Define initial parameters
pnames_alpha = ['$per$ [d]', '$T_0$ [BJD-%s]' % BJD_offset, 'b']
pnames_beta = ['$R_p/R_*$', '$a/R_*$', 'b', 'offset', '$\sigma_f$']
t0_i = 2237.25		# BJD (look at light curve)
b_i = 0.1			# 
offset_i = 0.005	# offset added to model
sigma_i = 0.001		# flux uncertainty

# From Crossfield et al. 2016
rp_i = 0.0261		# Rp/R*
s_rp = 0.0012
per_i = 10.99573	# days
s_per = 0.00070
Rstar = 1.28		# Rsun
s_Rstar = 0.12
Mstar = 1.06		# Msun
s_Mstar = 0.04

# From Claret et al. 2012/13
u1 = 0.4899		# LD coeff 1
u2 = 0.1209		# LD coeff 2

# MCMC parameters
nsteps = 10000
burn_in = 2500
nwalkers = 100
# WARNING: THIS IS SLOW
##########################################################
'''

# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################

# SOMETHING IS WRONG WITH THE EVEREST DATA FOR THIS TARGET...
# phased LC has quadratic-like artifact

targ_name = 'K2-36(c)'
epic = 201713348
save_results = True	# Save results?
save_phased = True	# Save phased LC data?
show_plots = False	# Show plots?
annotate_plot = True

BJD_offset = 2454833	# Get this by plotting the full Everest light curve for plotting

# Define initial parameters
pnames_alpha = ['$per$ [d]', '$T_0$ [BJD-%s]' % BJD_offset, 'b']
pnames_beta = ['$R_p/R_*$', '$a/R_*$', 'b', 'offset', '$\sigma_f$']
t0_i = 1979.85		# BJD (look at light curve)
b_i = 0.1			# 
offset_i = 0.005	# offset added to model
sigma_i = 0.001		# flux uncertainty

# From Crossfield et al. 2016
rp_i = 0.0323		# Rp/R*
s_rp = 0.0028
per_i = 5.34072		# days
s_per = 0.00011
Rstar = 0.75		# Rsun
s_Rstar = 0.02
Mstar = 0.81		# Msun
s_Mstar = 0.02

# From Claret et al. 2012/13
u1 = 0.6147		# LD coeff 1
u2 = 0.1061		# LD coeff 2

# MCMC parameters
nsteps = 10
burn_in = 2
nwalkers = 100
# WARNING: THIS IS SLOW
##########################################################
'''

############## THE CODE STARTS HERE ##############
##################################################

start = systime.time()
save_location = 'fig//K2_plots//%s//V2//%ssteps' % (targ_name,nsteps)

# First, find the period and T0 from the whole light curve.

# Set up priors.
def lnprior_alpha(x, t0_i=t0_i, mu_per=per_i, sigma_per=s_per):
	per, t0, b = x
	if (t0_i-0.1 < t0 < t0_i+0.1) \
	and (0. < per) \
	and (-1. < b < 1.):
		Pr_per = (1. / (np.sqrt(2.*np.pi)*sigma_per)) * np.exp((-1. / (2. * sigma_per**2.)) * (per - mu_per)**2.)
		if Pr_per > 0.:
			return np.log(Pr_per)
	return -np.inf

# Log likelihood function
def lnlike_alpha(x, star):
	ll = lnprior_alpha(x)
	if np.isinf(ll):
		return ll, (np.nan, np.nan)
	per, t0, b = x
	model = TransitModel('b', per=per, t0=t0, b=b)(star.time)
	like, d, vard = star.lnlike(model, full_output=True)
	ll += like
	return ll, (d,)

# Initialize the everest model
star = Everest(epic)

# Set up the MCMC sampler
blobs = ['Depth [%]']
ndim = len(pnames_alpha)
nblobs = len(blobs)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike_alpha, args=[star])
x0 = [[per_i + 0.01 * np.random.randn(), t0_i + 0.01 * np.random.randn(), b_i + 0.1 * np.random.randn()] for k in range(nwalkers)]
blobs0 = [[0.] for k in range(nwalkers)]

# Run MCMC for n steps and display progress bar.
barwidth = 50
for i, result in enumerate(sampler.sample(x0, iterations=nsteps, blobs0=blobs0)):
	n = int((barwidth+1) * float(i) / nsteps)
	sys.stdout.write("\r{}[{}{}]{}".format('sampling EVEREST... ', '#' * n, ' ' * (barwidth - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
sys.stdout.write("\n")
print 'sampling complete'

# Add the blobs to the chain for plotting
chain = np.concatenate((sampler.chain, np.array(sampler.blobs).swapaxes(0, 1)), axis=2)

# Take the absolute value of the impact parameter for plotting
chain[:, :, 2] = np.abs(chain[:, :, 2])

# Re-scale the transit depth as a percentage
chain[:, :, 3] *= 100.

# Plot the chains
fig_walkers_alpha, ax = plt.subplots(ndim + nblobs, figsize=(10, 8))
fig_walkers_alpha.suptitle('alpha walkers')
ax[-1].set_xlabel("Iteration")
for n in range(ndim + nblobs):
	for k in range(nwalkers):
		ax[n].plot(chain[k, :, n], color='k', alpha=0.3, lw=1)
	ax[n].set_ylabel((pnames_alpha + blobs)[n])
	ax[n].margins(0, None)
	ax[n].axvline(burn_in, color='b', alpha=0.5, lw=1, ls='--')

# Plot the posterior distributions
samples = chain[:, burn_in:, :].reshape(-1, ndim + nblobs)
fig_corner_alpha = corner(samples, labels=pnames_alpha + blobs)
fig_corner_alpha.suptitle('alpha posteriors')
for axis in fig_corner_alpha.axes:
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontsize(7)

# Extract most likely parameters from posterior distribution with uncertainties
per, t0, b, depth = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))

s_per = np.mean([per[1],per[2]])		# used to set Guassian prior for 2nd MC

##########################################################

# Now run another MC with to fit BATMAN to phased LC

# Fold data to period and t0 found above
star.mask_planet(t0[0],per[0])
star.compute()

time, flux = star.data_folded(t0[0],per[0])
t = time[time.argsort()] + t0[0]
f = flux[time.argsort()]

if save_phased == True:
	if not os.path.exists(save_location):
		os.makedirs(save_location)
	with open('%s//%s_%s_K2PhasedData.txt' % (save_location,targ_name,nsteps), 'w') as file_lc:
		for i in range(len(t)):
			file_lc.write(str(t[i]) + '\t' + str(f[i]) + '\n')


a_i = (((per[0]/365.)**2) / Mstar)**(1./3.) * Rstar * 215.035216		# semimajor axis from period and stellar params
s_a = a_i * np.sqrt((s_Rstar / Rstar)**2 + ((2*s_per) / (3*per[0]))**2 + (s_Mstar / (3*Mstar))**2)


# Set up priors.
def lnprior_beta(theta, mu_rp=rp_i, sigma_rp=s_rp, mu_a=a_i, sigma_a=s_a):
	rp, a, b, offset, sigma = theta
	if (0. < rp) \
	and (b < a) \
	and (0. <= b < 1.) \
	and (-0.05 < offset < 0.05) \
	and (0.0001 < sigma < 0.1):
		Pr_rp = (1. / (np.sqrt(2.*np.pi)*sigma_rp)) * np.exp((-1. / (2. * sigma_rp**2.)) * (rp - mu_rp)**2.)		# Gaussian priors on Rp and a
		Pr_a = (1. / (np.sqrt(2.*np.pi)*sigma_a)) * np.exp((-1. / (2. * sigma_a**2.)) * (a - mu_a)**2.)
		return np.log(Pr_rp + Pr_a)
	return -np.inf

# Log of the likelihood function.
def lnlike_beta(theta, x, y, t0=t0[0], period=per[0], u1=u1, u2=u2):
	rp, a, b, offset, sigma = theta
	# Set up transit parameters.
	params = batman.TransitParams()
	params.t0 = t0
	params.per = period
	params.rp = rp
	params.a = a
	params.inc = np.arccos(b / a) * (180. / np.pi)
	params.ecc = 0.
	params.w = 0.
	params.u = [u1,u2]
	params.limb_dark = 'quadratic'
	# Initialize the transit model.
	m_init = batman.TransitModel(params, x)
	model = m_init.light_curve(params) + offset
	inv_sigma2 = 1.0/(sigma**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(2*np.pi*inv_sigma2)))

# Define log of probability function.
def lnprob_beta(theta, x, y):
	lp = lnprior_beta(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_beta(theta, x, y)

ndim = len(pnames_beta)
initial_params = rp_i, a_i, b_i, offset_i, sigma_i

# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob_beta, args=(t, f))

# Run MCMC for n steps and display progress bar.
for i, result in enumerate(sampler2.sample(pos, iterations=nsteps)):
	n = int((barwidth+1) * float(i) / nsteps)
	sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (barwidth - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
sys.stdout.write("\n")
print 'sampling complete'

samples = sampler2.chain

# Plot walkers as a function of step number.
fig_walkers_beta, ax = plt.subplots(ndim, sharex=True, figsize=(10, 8))
fig_walkers_beta.suptitle('beta walkers')
ax[-1].set_xlabel('Iteration')
for n in range(ndim):
    for k in range(nwalkers):
        ax[n].plot(samples[k, :, n], color='k', alpha=0.3, lw=1)
    ax[n].set_ylabel((pnames_beta)[n], fontsize=9)
    ax[n].margins(0, None)
    ax[n].axvline(burn_in, color='b', alpha=0.5, lw=1, ls='--')

# Discard burn-in. 
samples = samples[:, burn_in:, :].reshape((-1, ndim))

# Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
rp_mcmc, a_mcmc, b_mcmc, offset_mcmc, sigma_mcmc = map(
	lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

# create array for flux uncert
ferr = []
for i in f:
	ferr.append(sigma_mcmc[0])
ferr = np.asarray(ferr)

# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = t0[0]
params_final.per = per[0]
params_final.rp = rp_mcmc[0]
params_final.a = a_mcmc[0]
params_final.inc = np.arccos(b_mcmc[0] / a_mcmc[0]) * (180. / np.pi)
params_final.ecc = 0.
params_final.w = 0.
params_final.u = [u1,u2]
params_final.limb_dark = "quadratic"
tl = np.linspace(min(t),max(t),5000)
m = batman.TransitModel(params_final, tl)
m_res = batman.TransitModel(params_final, t)

f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title('%s (EPIC %s)' % (targ_name, str(epic)))
ax.errorbar(t,f - offset_mcmc[0],yerr=ferr,fmt='k.',capsize=0,alpha=0.5,zorder=1)
ax.plot(tl, f_final, 'r-', lw=3, zorder=2)
if annotate_plot == True:
	ant = AnchoredText('$T_0 = %s^{+%s}_{-%s}$ \n $Per = %s^{+%s}_{-%s}$ \n $R_p/R_* = %s^{+%s}_{-%s}$' % (
		BJD_offset+round(t0[0],5),round(t0[1],5),round(t0[2],5),
		round(per[0],5),round(per[1],5),round(per[2],5),
		round(rp_mcmc[0],5),round(rp_mcmc[1],5),round(rp_mcmc[2],5)), prop=dict(size=11), frameon=True, loc=3)
	ant.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
	ax.add_artist(ant)
ax.set_xlabel("$BJD_{TDB} - %s$" % str(BJD_offset))
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','K2'), loc=2)

f_res = m_res.light_curve(params_final) + offset_mcmc[0]
resid_fig, axr = plt.subplots(figsize=(10,8))
axr.errorbar(t,f-f_res,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
axr.plot(t, np.zeros(len(t)), 'r-', lw=3,zorder=2)
axr.set_xlabel("$BJD_{TDB} - %s$" % str(BJD_offset))
axr.set_ylabel("Residual")

fig_corner_beta = corner(samples, labels=pnames_beta)
fig_corner_beta.suptitle('beta posteriors')

param_names = ['Per', 'T0', 'Rp/R*', 'a/R*', 'b', 'offset', 'sigma_f']
fit_params = [per, t0, rp_mcmc, a_mcmc, b_mcmc, offset_mcmc, sigma_mcmc]

exec_time = systime.time() - start

if save_results == True:
	fig_walkers_alpha.savefig('%s//%s_%s_walkersA_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	fig_corner_alpha.savefig('%s//%s_%s_cornerA_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	final_fig.savefig('%s//%s_%s_MCMCfit_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	fig_corner_beta.savefig('%s//%s_%s_cornerB_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	fig_walkers_beta.savefig('%s//%s_%s_walkersB_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	resid_fig.savefig('%s//%s_%s_residuals_K2.png' % (save_location,targ_name,nsteps), bbox_inches='tight')
	
	with open('%s//%s_%s_LOG_K2.txt' % (save_location,targ_name,nsteps), 'w') as newfile:
			newfile.write(
				'K2_ID' + '\t' + targ_name + '\n'
				'EPIC' + '\t' + str(epic) + '\n'
				'data' + '\t' + 'K2' + '\n'
				'\n'
				'nsteps' + '\t' + str(nsteps) + '\n'
				'nwalk' + '\t' + str(nwalkers) + '\n'
				'burnin' + '\t' + str(burn_in) + '\n'
				'\n'
				'Rstar' + '\t' + str(Rstar) + '\t' + str(s_Rstar) + '\t' + '[Rsun]' + '\t' + '(fixed)' + '\n'
				'Mstar' + '\t' + str(Mstar) + '\t' + str(s_Mstar) + '\t' + '[Msun]' + '\t' + '(fixed)' + '\n'
				'u1' + '\t' + str(u1) + '\t' + '(fixed)' + '\n'
				'u2' + '\t' + str(u2) + '\t' + '(fixed)' + '\n'
				'\n'
				)
			for i,param in enumerate(param_names):
				newfile.write(
					param + '\t' + 
					str(fit_params[i][0]) + '\t' + 
					str(fit_params[i][1]) + '\t' + 
					str(fit_params[i][2]) + '\n'
					)
			newfile.write('\n\n\n' + 'runtime' + '\t' + str(round(exec_time,2)) + '\t' + '[s]')


print 'period: ',per
print 't0: ',t0
print 'impact parameter: ',b_mcmc
print 'depth (percent): ',depth
print 'rp: ',rp_mcmc
print 'a: ',a_mcmc
print 'offset: ',offset_mcmc
print 'sigma: ',sigma_mcmc
print 'runtime: ',exec_time

if show_plots == True:
	plt.show()