"""
Jump-diffusion model, restructured to pool jump-specific parameters
(lambda, jump mean, jump volatility) across patients within a group
(a septic cluster, or the non-septic cohort), while still fitting
drift (mu) and ordinary volatility (sigma) individually per patient.

This fixes the boundary-hitting problem from the fully-individual
version: lambda and the jump-size parameters need many rare events
to estimate reliably, which no single patient's stay provides enough
of. Pooling many patients' data within a group fixes that, while mu
and sigma - which are estimable from ordinary (non-rare) data - stay
personalized per patient as originally intended.

Requires: pip install scipy numpy pandas
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def jump_diffusion_neg_log_likelihood(params, changes):
    """
    Negative log-likelihood of observed hourly changes under the
    full 5-parameter mixture. Used ONLY for pooled (group-level) fits,
    where there's enough data to estimate all 5 parameters reliably.
    """
    mu, sigma, lam, mu_j, sigma_j = params

    if sigma <= 0 or sigma_j <= 0 or not (0 <= lam <= 1):
        return np.inf

    diffusion_density = norm.pdf(changes, loc=mu, scale=sigma)
    jump_density = norm.pdf(changes, loc=mu + mu_j, scale=np.sqrt(sigma**2 + sigma_j**2))
    mixture_density = (1 - lam) * diffusion_density + lam * jump_density
    mixture_density = np.clip(mixture_density, 1e-300, None)

    return -np.sum(np.log(mixture_density))


def fit_pooled_jump_params(pooled_changes, min_observations=200):
    """
    Fit the FULL 5-parameter mixture model on pooled data from many
    patients within one group (a cluster, or the non-septic cohort).
    Returns all 5 parameters, but only lambda/mu_j/sigma_j are kept
    and reused downstream (mu and sigma here are just a group average,
    not what gets used per patient).
    """
    pooled_changes = pooled_changes.dropna().values
    if len(pooled_changes) < min_observations:
        return None

    mu0 = np.mean(pooled_changes)
    sigma0 = np.std(pooled_changes) if np.std(pooled_changes) > 0 else 1.0
    initial_params = [mu0, sigma0, 0.05, 0.0, sigma0 * 2]

    bounds = [
        (None, None),
        (1e-6, None),
        (1e-6, 0.5),
        (None, None),
        (1e-6, None),
    ]

    result = minimize(
        jump_diffusion_neg_log_likelihood,
        x0=initial_params,
        args=(pooled_changes,),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success:
        return None

    mu, sigma, lam, mu_j, sigma_j = result.x
    return {
        "group_mu": mu,
        "group_sigma": sigma,
        "lambda": lam,
        "mu_j": mu_j,
        "sigma_j": sigma_j,
        "log_likelihood": -result.fun,
        "n_obs": len(pooled_changes),
    }


def patient_diffusion_neg_log_likelihood(params, changes, lam, mu_j, sigma_j):
    """
    Negative log-likelihood for ONE patient's changes, but with
    lambda, mu_j, and sigma_j held FIXED (borrowed from the pooled
    group fit). Only mu and sigma are free parameters here - a much
    better-constrained 2-parameter problem than the original 5.
    """
    mu, sigma = params

    if sigma <= 0:
        return np.inf

    diffusion_density = norm.pdf(changes, loc=mu, scale=sigma)
    jump_density = norm.pdf(changes, loc=mu + mu_j, scale=np.sqrt(sigma**2 + sigma_j**2))
    mixture_density = (1 - lam) * diffusion_density + lam * jump_density
    mixture_density = np.clip(mixture_density, 1e-300, None)

    return -np.sum(np.log(mixture_density))


def fit_patient_diffusion_params(changes, lam, mu_j, sigma_j, min_observations=15):
    """
    Fit only mu and sigma for one patient, with the group's jump
    parameters (lambda, mu_j, sigma_j) held fixed. Needs far less
    data than the original 5-parameter fit, since mu/sigma are
    informed by every observation, not just rare jump events.
    """
    changes = changes.dropna().values
    if len(changes) < min_observations:
        return None

    mu0 = np.mean(changes)
    sigma0 = np.std(changes) if np.std(changes) > 0 else 1.0

    result = minimize(
        patient_diffusion_neg_log_likelihood,
        x0=[mu0, sigma0],
        args=(changes, lam, mu_j, sigma_j),
        bounds=[(None, None), (1e-6, None)],
        method="L-BFGS-B",
    )

    if not result.success:
        return None

    mu, sigma = result.x
    return {"mu": mu, "sigma": sigma, "n_obs": len(changes)}


def posterior_jump_probability(x, mu, sigma, lam, mu_j, sigma_j):
    """
    Posterior probability that a specific observed change x was a
    jump, given a patient's own (mu, sigma) and their group's
    (lambda, mu_j, sigma_j). Same Bayes' rule as before.
    """
    diffusion_density = norm.pdf(x, loc=mu, scale=sigma)
    jump_density = norm.pdf(x, loc=mu + mu_j, scale=np.sqrt(sigma**2 + sigma_j**2))

    numerator = lam * jump_density
    denominator = (1 - lam) * diffusion_density + lam * jump_density
    denominator = np.where(denominator <= 0, 1e-300, denominator)

    return numerator / denominator


def fit_all_patients_by_group(df, vitals, group_col="group", min_group_observations=200,
                                min_patient_observations=15):
    """
    Full pipeline: for each vital, for each group (cluster or cohort):
      1. Pool every patient's changes within that group.
      2. Fit lambda/mu_j/sigma_j once on the pooled data.
      3. Fit mu/sigma separately for each patient in that group,
         using the pooled jump parameters as fixed inputs.
      4. Compute a posterior jump probability for every hour.

    df must already have a `group_col` column assigning every patient
    to a group (e.g. cluster 0/1/2 for septic patients, or a single
    'non_septic' label for everyone else).
    """
    group_fits = []
    patient_fits = []

    for v in vitals:
        change_col = f"{v}_change"
        prob_col = f"{v}_jump_prob"
        df[prob_col] = np.nan

        for group_label, group_df in df.groupby(group_col):
            pooled_changes = group_df[change_col]
            pooled_fit = fit_pooled_jump_params(pooled_changes, min_observations=min_group_observations)

            if pooled_fit is None:
                print(f"  Skipping group '{group_label}' for {v}: not enough pooled data.")
                continue

            pooled_fit["group"] = group_label
            pooled_fit["vital"] = v
            group_fits.append(pooled_fit)

            lam = pooled_fit["lambda"]
            mu_j = pooled_fit["mu_j"]
            sigma_j = pooled_fit["sigma_j"]

            for patient_id, patient_df in group_df.groupby("patient_id"):
                p_fit = fit_patient_diffusion_params(
                    patient_df[change_col], lam, mu_j, sigma_j,
                    min_observations=min_patient_observations,
                )
                if p_fit is None:
                    continue

                p_fit["patient_id"] = patient_id
                p_fit["group"] = group_label
                p_fit["vital"] = v
                patient_fits.append(p_fit)

                idx = patient_df.index
                probs = posterior_jump_probability(
                    patient_df[change_col].values,
                    p_fit["mu"], p_fit["sigma"], lam, mu_j, sigma_j,
                )
                df.loc[idx, prob_col] = probs

    group_results = pd.DataFrame(group_fits)
    patient_results = pd.DataFrame(patient_fits)
    return df, group_results, patient_results


if __name__ == "__main__":
    pass