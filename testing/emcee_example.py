import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner

# Choose the "true" parameters.
m_true = -0.6036
b_true = 4.3846
f_true = 0.3865

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
xx = np.linspace(-5,15,10)
y_true = m_true*xx+b_true
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# Plot the data.
plt.figure(figsize=(10,8))
plt.errorbar(x,y,yerr=yerr,fmt='k.',capsize=3)
plt.plot(xx,y_true,'k-',alpha=0.3,lw=3,label='True values')
plt.xlim((-0.2,10.2))
plt.xlabel('$x$')
plt.ylabel('$y$')

# Linear least-squares fit.
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
y_ls = m_ls*xx + b_ls
plt.plot(xx,y_ls,'k--',alpha=0.5,label='Least-squares')

# Define the log of the likelihood function.
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Maximizing the likelihood function.
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]
y_ml = m_ml*xx + b_ml
plt.plot(xx,y_ml,'k-',label='Max likelihood')

# Setting up priors.
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Log probability function.
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Initialize walkers around maximum likelihood.
ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# Run the MCMC for 500 steps, starting from initial conditions defined above.
sampler.run_mcmc(pos, 500)

# Discard the first 50 steps. 
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
y_mcmc = m_mcmc[0]*xx + b_mcmc[0]
plt.plot(xx,y_mcmc,'r-',lw=3,alpha=0.5,label='MCMC')
plt.legend()

# Make a corner plot.
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
                      #truths=[m_true, b_true, np.log(f_true)])

# Plot some MCMC results.
plt.figure(figsize=(10,8))
plt.errorbar(x,y,yerr=yerr,fmt='k.',capsize=3)
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(xx, m*xx+b, color='k', alpha=0.1)
plt.xlim((-.2,10.2))

print m_true
print m_mcmc

print b_true
print b_mcmc

plt.show()
