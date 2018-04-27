# v2 -- Impact parameter; Gaussian priors on a, p, rp
# v3 -- binning, TDB correction deprecated


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy.stats import binned_statistic
import batman
import emcee
import corner
import os, sys, time


'''
# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################
targ_name = 'K2-79'
epic = 210402237
save_results = True	# Save plots?
show_plots = False	# Show plots?
annotate_plot = True

bjd_offset = 57710.		# Makes nicer plots

# Define initial parameters
param_names = ["$T_0$", "$R_p/R_*$", "$a/R_*$", "$b$", "$offset$", "$\sigma_f$"]
t0_i = 57718.93		# BJD (look at light curve)
b_i = 0.1			# degrees
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
u1 = 0.0764		# LD coeff 1
u2 = 0.1004		# LD coeff 2

# Number of Bins
nbins = 100

# MCMC parameters
nsteps = 10000
burn_in = 2500
ndim = 6
nwalkers = 100
##########################################################
'''

# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################
targ_name = 'K2-36'
epic = 201713348
save_results = True	# Save plots?
show_plots = False	# Show plots?
annotate_plot = True

bjd_offset = 57460.		# Makes nicer plots

# Define initial parameters
param_names = ["$T_0$", "$R_p/R_*$", "$a/R_*$", "$b$", "$offset$", "$\sigma_f$"]
t0_i = 57464.45		# BJD (look at light curve)
b_i = 0.5			# impact parameter
offset_i = 0.00005	# offset added to model
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
u1 = 0.0965		# LD coeff 1
u2 = 0.1144		# LD coeff 2

# Number of Bins
nbins = 500

# MCMC parameters
nsteps = 10000
burn_in = 2500
ndim = 6
nwalkers = 100
##########################################################



############## THE CODE STARTS HERE ##############
##################################################

start = time.time()

a_i = (((per_i/365.)**2) / Mstar)**(1./3.) * Rstar * 215.035216		# semimajor axis from period and stellar params
s_a = a_i * np.sqrt((s_Rstar / Rstar)**2 + ((2*s_per) / (3*per_i))**2 + (s_Mstar / (3*Mstar))**2)

# Priors.
def lnprior(theta, t0_i=t0_i, mu_rp=rp_i, sigma_rp=s_rp, mu_per=per_i, sigma_per=s_per, mu_a=a_i, sigma_a=s_a):
	t0, rp, a, b, offset, sigma = theta
	if (t0_i-0.05 < t0 < t0_i+0.05) \
	and (0. < rp) \
	and (b < a) \
	and (0. <= b < 1.) \
	and (-0.1 < offset < 0.1) \
	and (0.0001 < sigma < 0.1):
		Pr_rp = (1. / (np.sqrt(2.*np.pi)*sigma_rp)) * np.exp((-1. / (2. * sigma_rp**2.)) * (rp - mu_rp)**2.)		# Gaussian priors on Rp and a
		Pr_a = (1. / (np.sqrt(2.*np.pi)*sigma_a)) * np.exp((-1. / (2. * sigma_a**2.)) * (a - mu_a)**2.)
		return np.log(Pr_rp + Pr_a)
	return -np.inf

# Import data from text file
data = []
with open('PhotData//%s_Spitzer_unbin_TDB.txt' % targ_name) as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		data.append(data_flt)
data = np.asarray(data)
data = data[data[:,0].argsort()]

t_all = []
f_all = []

for i in data[:,0]:
	t_all.append(i)
for i in data[:,1]:
	f_all.append(i)

def bin_data(t,f,nbins):
	binnedTimes = []
	binnedData = binned_statistic(t,f,bins=nbins)
	bin_edges = binnedData[1]
	for n in range(len(bin_edges) - 1):
		time = (bin_edges[n] + bin_edges[n+1]) / 2.
		binnedTimes.append(time)
	t_binned = np.asarray(binnedTimes)
	f_binned = binnedData[0]
	return t_binned, f_binned

DataBinned = bin_data(t_all,f_all,nbins)

t = DataBinned[0]
f = DataBinned[1]

# Define log of the likelihood function.
def lnlike(theta, x, y, per=per_i, u1=u1, u2=u2):
	t0, rp, a, b, offset, sigma = theta
	# Set up transit parameters.
	params = batman.TransitParams()
	params.t0 = t0
	params.per = per
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
	inv_sigma2 = 1.0 / (sigma**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(2*np.pi*inv_sigma2)))

# Define log of probability function.
def lnprob(theta, x, y):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y)


initial_params = t0_i, rp_i, a_i, b_i, offset_i, sigma_i

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
t0_mcmc, rp_mcmc, a_mcmc, b_mcmc, offset_mcmc, sigma_mcmc = map(
	lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

# create array for flux uncert
ferr = []
for i in f:
	ferr.append(sigma_mcmc[0])
ferr = np.asarray(ferr)

# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = t0_mcmc[0]
params_final.per = per_i
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
ax.errorbar(t-bjd_offset,f - offset_mcmc[0],yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
ax.plot(tl-bjd_offset, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
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
axr.errorbar(t-bjd_offset,f-f_res,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
axr.plot(t-bjd_offset, np.zeros(len(t)), 'r-',alpha=0.8,lw=3,zorder=2)
axr.set_xlabel("$BJD_{TDB} - 24%s$" % str(bjd_offset))
axr.set_ylabel("Residual")

corn_fig = corner.corner(samples, labels=param_names)

fit_params = [t0_mcmc, rp_mcmc, a_mcmc, b_mcmc, offset_mcmc, sigma_mcmc]

exec_time = time.time() - start

if save_results == True:
	save_to = 'fig//Spitzer_plots//%s//V3//bin%s' % (targ_name,nbins)
	if not os.path.exists(save_to):
		os.makedirs(save_to)

	final_fig.savefig('%s//%s_%sb_%ss_MCMCfit_Spitzer.png' % (save_to,targ_name,nbins,nsteps), bbox_inches='tight')
	corn_fig.savefig('%s//%s_%sb_%ss_corner_Spitzer.png' % (save_to,targ_name,nbins,nsteps), bbox_inches='tight')
	walker_fig.savefig('%s//%s_%sb_%ss_walkers_Spitzer.png' % (save_to,targ_name,nbins,nsteps), bbox_inches='tight')
	resid_fig.savefig('%s//%s_%sb_%ss_residuals_Spitzer.png' % (save_to,targ_name,nbins,nsteps), bbox_inches='tight')

	with open('%s//%s_%sb_%ss_LOG_Spitzer.txt' % (save_to,targ_name,nbins,nsteps), 'w') as newfile:
		newfile.write(
			'K2_ID' + '\t' + targ_name + '\n'
			'EPIC' + '\t' + str(epic) + '\n'
			'data' + '\t' + 'Spitzer' + '\n'
			'\n'
			'nbins' + '\t' + str(nbins) + '\n'
			'nsteps' + '\t' + str(nsteps) + '\n'
			'nwalk' + '\t' + str(nwalkers) + '\n'
			'burnin' + '\t' + str(burn_in) + '\n'
			'\n'
			'per' + '\t' + str(per_i) + '\t' + str(s_per) + '\t' + '[d]' + '\t' + '(fixed)' + '\n'
			'Rstar' + '\t' + str(Rstar) + '\t' + str(s_Rstar) + '\t' + '[Rsun]' + '\t' + '(fixed)' + '\n'
			'Mstar' + '\t' + str(Mstar) + '\t' + str(s_Mstar) + '\t' + '[Msun]' + '\t' + '(fixed)' + '\n'
			'u1' + '\t' + str(u1) + '\t' + '(fixed)' + '\n'
			'u2' + '\t' + str(u2) + '\t' + '(fixed)' + '\n'
			'\n'
			)
		for i,param in enumerate(param_names):
			newfile.write(
				param[1:len(param)-1] + '\t' + 
				str(fit_params[i][0]) + '\t' + 
				str(fit_params[i][1]) + '\t' + 
				str(fit_params[i][2]) + '\n'
				)
		newfile.write('\n\n\n' + 'runtime' + '\t' + str(round(exec_time,2)) + '\t' + '[s]')

print 't0: ',t0_mcmc
print 'rp: ',rp_mcmc
print 'a: ',a_mcmc
print 'b: ',b_mcmc
print 'offset: ',offset_mcmc
print 'sigma: ',sigma_mcmc
print 'runtime: ',exec_time

if show_plots == True:
	plt.show()