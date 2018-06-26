# -*- coding: utf-8 -*-
"""

    TOPIC: Bayesian linear regression (with PyMC3)

"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

# Seaborn setup.
sns.set(style="darkgrid", palette="muted")


# %%

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    beta_0: Intercept
    beta_1: Slope of univariate predictor, X
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0.
    randomState = np.random.RandomState(1)
    x = randomState.choice(np.linspace(0, 1, N, endpoint=True), N, replace=False)
    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    epsilon = randomState.normal(loc=eps_sigma_sq[0], scale=eps_sigma_sq[1], size=N)
    y = beta_0 + beta_1*x + epsilon
    df = pd.DataFrame(data={'x': x, 'y': y},
                      columns=['x', 'y'])
    return df


# Simulate the "linear" data.
N = 500
beta_0 = 1.0
beta_1 = 2.0
eps_sigma_sq = (0.0, 0.5)
df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

# Plot the data, and a frequentist linear regression fit
# using the seaborn package
sns.lmplot(x="x", y="y", data=df, size=10)
plt.xlim(0.0, 1.0)
plt.show()


# ----------------------------------------------------------------------------

# %% Compute trace of the GLM Bayesian LR model on the supplied data.

def glm_mcmc_inference(df, iterations=5000):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression
    model on supplied data.

    df: DataFrame containing the data
    iterations: Number of iterations to carry out MCMC for
    """
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        pm.glm.GLM.from_formula("y ~ x", df, family=pm.glm.families.Normal())
        # Calculate the trace.
        #  * Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        #  * Use the No-U-Turn Sampler
        trace = pm.sample(draws=iterations, step=pm.step_methods.hmc.nuts.NUTS(),
                          start=pm.find_MAP(), random_seed=1, progressbar=True)

    return basic_model, trace


mdl_ols, trace = glm_mcmc_inference(df, iterations=5000)

print('Inference finished.')


BURNIN_LENGTH = 500

# Plot traces for each beta parameter.
# The first few traces are likely to be poor estimates of the parameter values,
# because the sampling has not yet converged or *burned-in*. We'll ignore the
# first 500 samples as *burn-in*, and only consider the rest.
pm.traceplot(trace[BURNIN_LENGTH:], figsize=(12, len(trace.varnames) * 1.5),
             lines={k: v['mean'] for k, v in pm.summary(trace[BURNIN_LENGTH:]).iterrows()})
plt.show()


# %% Plot the distribution of likely regression lines

# Plot a sample of posterior regression lines
sns.lmplot(x="x", y="y", data=df, size=10, fit_reg=False)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 4.0)
pm.plot_posterior_predictive_glm(trace, samples=100)
x = np.array([0, 1])
y = beta_0 + beta_1 * x
plt.plot(x, y, label="True Regression Line", lw=3., c="green")
plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.show()


# %% Plot 50% and 95% credible intervals.

# We expect 50% of the datapoints to fall within the 50% CR and likewise for the 95% CR.

def plot_posterior_cr(mdl, trc, rawdata, xlims, npoints=1000):
    '''
    Convenience fn: plot the posterior predictions from mdl given trcs
    '''

    ## extract traces
    trc_mu = pm.trace_to_dataframe(trc)[['Intercept', 'x']]
    trc_sd = pm.trace_to_dataframe(trc)['sd']

    ## recreate the likelihood
    x = np.linspace(xlims[0], xlims[1], npoints).reshape((npoints, 1))
    X = np.hstack((np.ones((npoints, 1)), x))
    like_mu = np.dot(X, trc_mu.T)
    like_sd = np.tile(trc_sd.T, (npoints, 1))
    like = np.random.normal(like_mu, like_sd)

    ## Calculate credible regions and plot over the datapoints
    dfp = pd.DataFrame(np.percentile(like, [2.5, 25, 50, 75, 97.5], axis=1).T
                       , columns=['025', '250', '500', '750', '975'])
    dfp['x'] = x

    pal = sns.color_palette('Purples')
    f, ax1d = plt.subplots(1, 1, figsize=(7, 7))
    ax1d.fill_between(dfp['x'], dfp['025'], dfp['975'], alpha=0.5
                      , color=pal[1], label='CR 95%')
    ax1d.fill_between(dfp['x'], dfp['250'], dfp['750'], alpha=0.4
                      , color=pal[4], label='CR 50%')
    ax1d.plot(dfp['x'], dfp['500'], alpha=0.5, color=pal[5], label='Median')
    plt.legend()
    ax1d.set_xlim(xlims)
    sns.regplot(x='x', y='y', data=rawdata, fit_reg=False,
                scatter_kws={'alpha': 0.8, 's': 80, 'lw': 2, 'edgecolor': 'w'}, ax=ax1d)


xlims = (df['x'].min() - np.ptp(df['x']) / 10,
         df['x'].max() + np.ptp(df['x']) / 10)

plot_posterior_cr(mdl_ols, trace, df, xlims)
plt.show()




# %% ===== Robust regression =====

# %% Obtain and plot data with outliers.

outs = pd.DataFrame(
    {"x": [.1, .15, .2], "y": [8, 6, 9]}
)

df_outs = pd.concat([df, outs])

xlims = (df_outs['x'].min() - np.ptp(df_outs['x']) / 10
         , df_outs['x'].max() + np.ptp(df_outs['x']) / 10)

x = np.array([xlims[0], xlims[1]])
# y = a + b*x
true_regression_line = beta_0 + beta_1 * x

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel='x', ylabel='y',
                     title='Generated data and underlying model')
ax.plot(x, true_regression_line, label='True Regression Line', lw=2.)
sns.regplot(x="x", y="y", data=df_outs, scatter_kws={"s": 50},
            fit_reg=False, ci=None, label='sampled data')
plt.legend(loc=0)
plt.show()


def glm_robust_regression(df, iterations=5000):
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        pm.glm.GLM.from_formula("y ~ x", df, family=pm.glm.families.StudentT())
        # Calculate the trace.
        #  * Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        #  * Use the No-U-Turn Sampler
        trace = pm.sample(draws=iterations, step=pm.step_methods.hmc.nuts.NUTS(),
                          start=pm.find_MAP(), random_seed=1, progressbar=True)

    return basic_model, trace


mdl_ols, trace = glm_robust_regression(df_outs, iterations=5000)
print('Inference for robust regression finished.')

# %% Plot the distribution of likely regression lines

# Plot a sample of posterior regression lines
sns.lmplot(x="x", y="y", data=df_outs, size=10, fit_reg=False)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 4.0)
pm.plot_posterior_predictive_glm(trace, samples=100)
x = np.array([0, 1])
y = beta_0 + beta_1 * x
plt.plot(x, y, label="True Regression Line (robust regression)", lw=3., c="green")
plt.title('Posterior predictive regression lines (robust regression)')
plt.legend(loc=0)
plt.show()

