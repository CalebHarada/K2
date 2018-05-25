
import numpy as np
import matplotlib.pyplot as plt 
import batman
import corner
import emcee
import os, sys
import time as systime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy.stats import binned_statistic




# ADJUST THINGS IN THIS SECTION. BE VERY CAREFUL!
##########################################################
targ_name = 'K2-79'
epic = 210402237
save_plots = True	# SAVE plots?
save_samples = True	# SAVE chains for future plotting?
disp_plots = False	# DISPLAY plots?
annotate_plot = True

kep_BJD = 2454833.
spit_BJD = 2400000.

# Define initial parameters
param_names = ['$T_0$','$Per$','$a/R*$','$b$','$R_p/R_*$ (K2)','$R_p/R_*$ (Spitzer)','$\sigma_{K2}$', '$\sigma_{Spitzer}$', 'offset (K2)', 'offset (Spitzer)']

#t0_in = 2237.2415		# from K2 fit
#s_t0 = 0.0011
#t0_in += kep_BJD

t0_in = 57718.9336			# from Spitzer fit
s_t0 = 0.0054
t0_in += spit_BJD

per_in = 10.99499		# from K2 fit
s_per = 0.00032

aKEP_in = 18.6
s_aKEP = 1.4
aSPIT_in = 15.7
s_aSPIT = 3.3
a_in = (aKEP_in/(s_aKEP**2) + aSPIT_in/(s_aSPIT**2)) / (1./(s_aKEP**2) + 1./(s_aSPIT**2))
s_a = 1. / np.sqrt(1./(s_aKEP**2) + 1./(s_aSPIT**2))

bKEP_in = 0.22
s_bKEP = 0.21
bSPIT_in = 0.38
s_bSPIT = 0.31
b_in = (bKEP_in/(s_bKEP**2) + bSPIT_in/(s_bSPIT**2)) / (1./(s_bKEP**2) + 1./(s_bSPIT**2))
s_b = 1. / np.sqrt(1./(s_bKEP**2) + 1./(s_bSPIT**2))

rpKEP_in = 0.02815
s_rpKEP = 0.00058

rpSPIT_in = 0.0262
s_rpSPIT = 0.0010

sigmaKEP_in = 0.0001625
s_sigmaKEP = 0.0000078

sigmaSPIT_in = 0.001151
s_sigmaSPIT = 0.000037

offsetKEP_in = -0.000025
s_offsetKEP = 0.000012

offsetSPIT_in = 0.000691
s_offsetSPIT = 0.000058


# From Claret et al. 2012/13
u1_KEP = 0.4899		# LD coeff 1
u2_KEP = 0.1209		# LD coeff 2
u1_SPIT = 0.0764		# LD coeff 1
u2_SPIT = 0.1004		# LD coeff 2

# From Crossfield et al. 2016
Rstar = 1.28		# Rsun
s_Rstar = 0.12
Mstar = 1.06		# Msun
s_Mstar = 0.04

# Number of bins for Spitzer
nbins = 200

# MCMC parameters
nsteps = 20000
burn_in = 12500
ndim = 10
nwalkers = 100
##########################################################


start = systime.time()
save_location = 'fig//JointAnalysis//%s//%ssteps' % (targ_name,nsteps)


# Priors (Guassian priors based on params from individual fits)
def lnprior(theta,
	mu_t0=t0_in, sig_t0=s_t0,
	mu_per=per_in, sig_per=s_per,
	mu_a=a_in, sig_a=s_a,
	mu_b=b_in, sig_b=s_b,
	mu_rpKEP=rpKEP_in, sig_rpKEP=s_rpKEP,
	mu_rpSPIT=rpSPIT_in, sig_rpSPIT=s_rpSPIT,
	mu_sigmaKEP=sigmaKEP_in, sig_sigmaKEP=s_sigmaKEP,
	mu_sigmaSPIT=sigmaSPIT_in, sig_sigmaSPIT=s_sigmaSPIT,
	mu_offsetKEP=offsetKEP_in, sig_offsetKEP=s_offsetKEP,
	mu_offsetSPIT=offsetSPIT_in, sig_offsetSPIT=s_offsetSPIT):
	t0, per, a, b, rpKEP, rpSPIT, sigmaKEP, sigmaSPIT, offsetKEP, offsetSPIT = theta
	"""
	if (mu_t0 - 5*sig_t0 < t0 < mu_t0 + 5*sig_t0) \
	and (mu_per - 5*sig_per < per < mu_per + 5*sig_per) \
	and (mu_a - 5*sig_a < a < mu_a + 5*sig_a) \
	and (mu_b - 5*sig_b < b < mu_b + 5*sig_b) \
	and (mu_rpKEP - 5*sig_rpKEP < rpKEP < mu_rpKEP + 5*sig_rpKEP) \
	and (mu_rpSPIT - 5*sig_rpSPIT < rpSPIT < mu_rpSPIT + 5*sig_rpSPIT) \
	and (mu_offsetKEP - 5*sig_offsetKEP < offsetKEP < mu_offsetKEP + 5*sig_offsetKEP) \
	and (mu_offsetSPIT - 5*sig_offsetSPIT < offsetSPIT < mu_offsetSPIT + 5*sig_offsetSPIT) \
	and (mu_sigmaKEP - 5*sig_sigmaKEP < sigmaKEP < mu_sigmaKEP + 5*sig_sigmaKEP) \
	and (mu_sigmaSPIT - 5*sig_sigmaSPIT < sigmaSPIT < mu_sigmaSPIT + 5*sig_sigmaSPIT) \
	"""
	if (0. < rpKEP) \
	and (0. < rpSPIT) \
	and (0. < sigmaKEP) \
	and (0. < sigmaSPIT) \
	and (b < a) \
	and (0. <= b < 1.):
		Pr_t0 = (1. / (np.sqrt(2.*np.pi)*sig_t0)) * np.exp((-1. / (2. * sig_t0**2.)) * (t0 - mu_t0)**2.)
		Pr_per = (1. / (np.sqrt(2.*np.pi)*sig_per)) * np.exp((-1. / (2. * sig_per**2.)) * (per - mu_per)**2.)
		Pr_a = (1. / (np.sqrt(2.*np.pi)*sig_a)) * np.exp((-1. / (2. * sig_a**2.)) * (a - mu_a)**2.)
		Pr_b = (1. / (np.sqrt(2.*np.pi)*sig_b)) * np.exp((-1. / (2. * sig_b**2.)) * (b - mu_b)**2.)
		Pr_rpKEP = (1. / (np.sqrt(2.*np.pi)*sig_rpKEP)) * np.exp((-1. / (2. * sig_rpKEP**2.)) * (rpKEP - mu_rpKEP)**2.)		# Gaussian priors on Rp and a
		Pr_rpSPIT = (1. / (np.sqrt(2.*np.pi)*sig_rpSPIT)) * np.exp((-1. / (2. * sig_rpSPIT**2.)) * (rpSPIT - mu_rpSPIT)**2.)
		Pr_sigmaKEP = (1. / (np.sqrt(2.*np.pi)*sig_sigmaKEP)) * np.exp((-1. / (2. * sig_sigmaKEP**2.)) * (sigmaKEP - mu_sigmaKEP)**2.)
		Pr_sigmaSPIT = (1. / (np.sqrt(2.*np.pi)*sig_sigmaSPIT)) * np.exp((-1. / (2. * sig_sigmaSPIT**2.)) * (sigmaSPIT - mu_sigmaSPIT)**2.)
		Pr_offsetKEP = (1. / (np.sqrt(2.*np.pi)*sig_offsetKEP)) * np.exp((-1. / (2. * sig_offsetKEP**2.)) * (offsetKEP - mu_offsetKEP)**2.)
		Pr_offsetSPIT = (1. / (np.sqrt(2.*np.pi)*sig_offsetSPIT)) * np.exp((-1. / (2. * sig_offsetSPIT**2.)) * (offsetSPIT - mu_offsetSPIT)**2.)
		return np.log(Pr_t0) + np.log(Pr_per) + np.log(Pr_a) + np.log(Pr_b) + np.log(Pr_rpKEP) + np.log(Pr_rpSPIT) + np.log(Pr_sigmaKEP) + np.log(Pr_sigmaSPIT) + np.log(Pr_offsetKEP) + np.log(Pr_offsetSPIT)
	return -np.inf


# Import Spitzer data from text file
spit_data = []
with open('PhotData//%s_Spitzer_unbin_TDB.txt' % targ_name) as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		spit_data.append(data_flt)
spit_data = np.asarray(spit_data)
spit_data = spit_data[spit_data[:,0].argsort()]
t_spit = []
f_spit = []
for i in spit_data[:,0]:
	t_spit.append(i)
for i in spit_data[:,1]:
	f_spit.append(i)

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

DataBinned = bin_data(t_spit,f_spit,nbins)
t_spit = DataBinned[0]
for i,n in enumerate(t_spit):
	t_spit[i] += spit_BJD
f_spit = DataBinned[1]


# Import K2 folded data from text file
kep_data = []
with open('PhotData//%s_10000_K2PhasedData.txt' % targ_name) as file:
	for line in file:
		data_str = line.split()
		data_flt = [float(i) for i in data_str]
		kep_data.append(data_flt)
kep_data = np.asarray(kep_data)
kep_data = kep_data[kep_data[:,0].argsort()]
t_kep = []
f_kep = []
for i in kep_data[:,0]:
	t_kep.append(i)
for i in kep_data[:,1]:
	f_kep.append(i)
for i,n in enumerate(t_kep):
	t_kep[i] += kep_BJD
t_kep = np.asarray(t_kep)
f_kep = np.asarray(f_kep)

# Define log of the likelihood function.
def lnlike(theta, t_kep, f_kep, t_spit, f_spit,
	u1_kep=u1_KEP, u2_kep=u2_KEP,
	u1_spit=u1_SPIT, u2_spit=u2_SPIT):
	t0, per, a, b, rpKEP, rpSPIT, sigmaKEP, sigmaSPIT, offsetKEP, offsetSPIT = theta

	# Set up K2 transit parameters.
	kep_params = batman.TransitParams()
	kep_params.t0 = t0
	kep_params.per = per
	kep_params.rp = rpKEP
	kep_params.a = a
	kep_params.inc = np.arccos(b / a) * (180. / np.pi)
	kep_params.ecc = 0.
	kep_params.w = 0.
	kep_params.u = [u1_kep,u2_kep]
	kep_params.limb_dark = 'quadratic'
	# Initialize the K2 transit model.
	kep_init = batman.TransitModel(kep_params, t_kep)
	kep_model = kep_init.light_curve(kep_params) + offsetKEP
	# Calculate likelihood of K2 model
	like_kep = -0.5*(np.sum((f_kep-kep_model)**2*(1/sigmaKEP**2) - np.log(2*np.pi*(1/sigmaKEP**2))))

	# Set up Spitzer transit parameters.
	spit_params = batman.TransitParams()
	spit_params.t0 = t0
	spit_params.per = per
	spit_params.rp = rpSPIT
	spit_params.a = a
	spit_params.inc = np.arccos(b / a) * (180. / np.pi)
	spit_params.ecc = 0.
	spit_params.w = 0.
	spit_params.u = [u1_spit,u2_spit]
	spit_params.limb_dark = 'quadratic'
	# Initialize the Spitzer transit model.
	spit_init = batman.TransitModel(spit_params, t_spit)
	spit_model = spit_init.light_curve(spit_params) + offsetSPIT
	# Calculate likelihood of Spitzer model
	like_spit = -0.5*(np.sum((f_spit-spit_model)**2*(1/sigmaSPIT**2) - np.log(2*np.pi*(1/sigmaSPIT**2))))

	return like_kep + like_spit

	
# Define log of probability function.
def lnprob(theta, t_kep, f_kep, t_spit, f_spit):
	prior = lnprior(theta)
	evidence = lnlike(theta, t_kep, f_kep, t_spit, f_spit)
	if not np.isfinite(prior):
		return -np.inf
	elif not np.isfinite(evidence):
		return -np.inf
	return prior + evidence


initial_params = t0_in, per_in, a_in, b_in, rpKEP_in, rpSPIT_in, sigmaKEP_in, sigmaSPIT_in, offsetKEP_in, offsetSPIT_in

# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t_kep, f_kep, t_spit, f_spit))

# Run MCMC for n steps and display progress bar.
width = 50
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
	n = int((width+1) * float(i) / nsteps)
	sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
sys.stdout.write("\n")
print 'Sampling complete!'

samples = sampler.chain

if save_samples == True:
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	np.save('%s//%s_%sb_%ss_samples_JOINT' % (save_location,targ_name,nbins,nsteps), samples)


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
t0_f, per_f, a_f, b_f, rpKEP_f, rpSPIT_f, sigmaKEP_f, sigmaSPIT_f, offsetKEP_f, offsetSPIT_f = map(
	lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

# create arrays for flux uncert
ferrKEP = []
for i in f_kep:
	ferrKEP.append(sigmaKEP_f[0])
ferrKEP = np.asarray(ferrKEP)
ferrSPIT = []
for i in f_spit:
	ferrSPIT.append(sigmaSPIT_f[0])
ferrSPIT = np.asarray(ferrSPIT)

# Plot the final transit model.
kep_params_final = batman.TransitParams()
kep_params_final.t0 = t0_f[0]
kep_params_final.per = per_f[0]
kep_params_final.rp = rpKEP_f[0]
kep_params_final.a = a_f[0]
kep_params_final.inc = np.arccos(b_f[0] / a_f[0]) * (180. / np.pi)
kep_params_final.ecc = 0.
kep_params_final.w = 0.
kep_params_final.u = [u1_KEP,u2_KEP]
kep_params_final.limb_dark = "quadratic"
tl_kep = np.linspace(min(t_kep),max(t_kep),5000)
kep_model_f = batman.TransitModel(kep_params_final, tl_kep)
kep_model_resid = batman.TransitModel(kep_params_final, t_kep)
kep_modelLC = kep_model_f.light_curve(kep_params_final)

spit_params_final = batman.TransitParams()
spit_params_final.t0 = t0_f[0]
spit_params_final.per = per_f[0]
spit_params_final.rp = rpSPIT_f[0]
spit_params_final.a = a_f[0]
spit_params_final.inc = np.arccos(b_f[0] / a_f[0]) * (180. / np.pi)
spit_params_final.ecc = 0.
spit_params_final.w = 0.
spit_params_final.u = [u1_SPIT,u2_SPIT]
spit_params_final.limb_dark = "quadratic"
tl_spit = np.linspace(min(t_spit),max(t_spit),5000)
spit_model_f = batman.TransitModel(spit_params_final, tl_spit)
spit_model_resid = batman.TransitModel(spit_params_final, t_spit)
spit_modelLC = spit_model_f.light_curve(spit_params_final)


# Make some figures
final_fig, (ax_kep, ax_spit) = plt.subplots(2, figsize=(10,10))
final_fig.suptitle('%s (EPIC %s)' % (targ_name, str(epic)))

kep_time_offset = float(str(t_kep[0])[0:6]+'0')
ax_kep.errorbar(t_kep-kep_time_offset, f_kep-offsetKEP_f[0],yerr=ferrKEP,fmt='k.',capsize=0,alpha=0.75,zorder=1)
ax_kep.plot(tl_kep-kep_time_offset, kep_modelLC, 'r-',alpha=0.5,lw=3,zorder=2)
if annotate_plot == True:
	ant_kep = AnchoredText('$R_p/R_* = %s^{+%s}_{-%s}$' % (round(rpKEP_f[0],5),round(rpKEP_f[1],5),round(rpKEP_f[2],5)), prop=dict(size=11), frameon=True, loc=3)
	ant_kep.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
	ax_kep.add_artist(ant_kep)
ax_kep.set_xlabel("$BJD_{TDB} - %s$" % str(kep_time_offset))
ax_kep.set_ylabel("Relative Flux")
ax_kep.set_title('K2')

spit_time_offset = float(str(t_spit[0])[0:6]+'0')
ax_spit.errorbar(t_spit-spit_time_offset, f_spit-offsetSPIT_f[0],yerr=ferrSPIT,fmt='k.',capsize=0,alpha=0.75,zorder=1)
ax_spit.plot(tl_spit-spit_time_offset, spit_modelLC, 'r-',alpha=0.5,lw=3,zorder=2)
if annotate_plot == True:
	ant_spit = AnchoredText('$T_0 = %s^{+%s}_{-%s}$ \n $R_p/R_* = %s^{+%s}_{-%s}$' % (round(t0_f[0],4),round(t0_f[1],4),
	round(t0_f[2],4),round(rpSPIT_f[0],5),round(rpSPIT_f[1],5),round(rpSPIT_f[2],5)), prop=dict(size=11), frameon=True, loc=3)
	ant_spit.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
	ax_spit.add_artist(ant_spit)
ax_spit.set_xlabel("$BJD_{TDB} - %s$" % str(spit_time_offset))
ax_spit.set_ylabel("Relative Flux")
ax_spit.set_title('Spitzer')

corn_fig = corner.corner(samples, labels=param_names)

fit_params = [t0_f, per_f, a_f, b_f, rpKEP_f, rpSPIT_f, sigmaKEP_f, sigmaSPIT_f, offsetKEP_f, offsetSPIT_f]

exec_time = systime.time() - start

if save_plots == True:
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	final_fig.savefig('%s//%s_%sb_%ss_MCMCfit_JOINT.png' % (save_location,targ_name,nbins,nsteps), bbox_inches='tight')
	corn_fig.savefig('%s//%s_%sb_%ss_corner_JOINT.png' % (save_location,targ_name,nbins,nsteps), bbox_inches='tight')
	walker_fig.savefig('%s//%s_%sb_%ss_walkers_JOINT.png' % (save_location,targ_name,nbins,nsteps), bbox_inches='tight')

	with open('%s//%s_%sb_%ss_log_JOINT.txt' % (save_location,targ_name,nbins,nsteps), 'w') as newfile:
		newfile.write(
			'K2_ID' + '\t' + targ_name + '\n'
			'EPIC' + '\t' + str(epic) + '\n'
			'data' + '\t' + 'K2 and Spitzer' + '\n'
			'\n'
			'nbins' + '\t' + str(nbins) + '\n'
			'nsteps' + '\t' + str(nsteps) + '\n'
			'nwalk' + '\t' + str(nwalkers) + '\n'
			'burnin' + '\t' + str(burn_in) + '\n'
			'\n'
			'Rstar' + '\t' + str(Rstar) + '\t' + str(s_Rstar) + '\t' + '[Rsun]' + '\t' + '(fixed)' + '\n'
			'Mstar' + '\t' + str(Mstar) + '\t' + str(s_Mstar) + '\t' + '[Msun]' + '\t' + '(fixed)' + '\n'
			'K2_u1' + '\t' + str(u1_KEP) + '\t' + '(fixed)' + '\n'
			'K2_u2' + '\t' + str(u2_KEP) + '\t' + '(fixed)' + '\n'
			'Spitzer_u1' + '\t' + str(u1_SPIT) + '\t' + '(fixed)' + '\n'
			'Spitzer_u2' + '\t' + str(u2_SPIT) + '\t' + '(fixed)' + '\n'
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


if disp_plots == True:
	plt.show()









