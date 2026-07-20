"""
Basic implementation of parallel tempering with a geometric ladder of MH walkers
"""

import matplotlib.pyplot as plt
import numpy as np

global sigma_proposal, times, y, y_err

# choose mcmc hyperparameter
sigma_proposal = 50

# generate some fake data
times = np.linspace(0, 10, 100)
m_true = 50
y = times * m_true
y_err = 10


def calculate_likelihood_prob(m, times=times, y=y, y_err=y_err, temp=1):
    model_prediction = m * times
    residual = model_prediction - y
    chi2_array = (residual / y_err) ** 2
    chi2 = np.sum(chi2_array)

    likelihood_prob = np.exp(-0.5 * chi2) ** (1 / temp)
    return likelihood_prob


def calculate_acceptance_prob(m_current, m_new):

    acceptance_prob = np.min(
        [
            1,
            calculate_likelihood_prob(m_new, temp=1)
            / calculate_likelihood_prob(m_current, temp=1),
        ]
    )

    return acceptance_prob


def calculate_swap_prob(m_temp1, m_temp2, temp1, temp2):

    # TODO: add logic to compute the swap probability swap_prob

    return swap_prob


def pick_next_state(m_current):
    m_new = np.random.normal(loc=m_current, scale=sigma_proposal)
    return m_new


def pt_mh_mcmc(
    m_initial_guess=47, num_steps=200, temp0=1, num_temps=5, geometric_lader_factor=10
):

    num_temp_pairs = num_temps - 1

    markov_chain = np.zeros(
        (num_steps + 1, num_temps)
    )  # num_steps + 1 because we're also storing initial state
    markov_chain[0, :] = m_initial_guess

    # choose a geometric spacing of temperatures (the "default")
    # TODO: make a "ladder" of temperatures (stored in variable temperatures)
    # such that temperatures[0] = temp0, and temperatures[i+1]/temperatures[i] = geometric_lader_factor

    for i in range(num_steps):
        for j in range(num_temps):
            m_current = markov_chain[i, j]
            m_proposal = pick_next_state(m_current)
            acceptance_prob = calculate_acceptance_prob(
                m_current, m_proposal, temp=temperatures[j]
            )

            random_number = np.random.uniform(0, 1, size=1)
            if acceptance_prob >= random_number:
                m_current = m_proposal

            markov_chain[i + 1, j] = m_current

        # compute swap probability & do swaps
        for j in range(num_temp_pairs):
            low_temp = temperatures[j]
            high_temp = temperatures[j + 1]

            m_lowtemp = markov_chain[i + 1, j]
            m_hightemp = markov_chain[i + 1, j + 1]

            swap_prob = calculate_swap_prob(m_lowtemp, m_hightemp, low_temp, high_temp)
            random_number = np.random.uniform(0, 1, size=1)
            if swap_prob >= random_number:  # do the swap!
                markov_chain[i + 1, j] = m_hightemp
                markov_chain[i + 1, j + 1] = m_lowtemp

    return markov_chain, temperatures


if __name__ == "__main__":
    np.random.seed(10)

    m_initial_guess = 50
    num_steps = 500
    num_temps = 3
    geometric_lader_factor = 25

    # TODO: play around with the values of num_temps and geometric_lader_factor.
    # What happens? Does the behavior you see agree with what you expect?

    markov_chain, temperatures = pt_mh_mcmc(
        m_initial_guess=m_initial_guess,
        num_steps=num_steps,
        num_temps=num_temps,
        geometric_lader_factor=geometric_lader_factor,
    )

    # TODO: identify where the temperature "swaps" happen in the plot

    plt.figure()
    for i in range(num_temps):
        plt.plot(markov_chain[:, i], label=f"T={temperatures[i]}", alpha=0.5)
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("m")
    plt.savefig("PT_MH_markov_chain.png", dpi=250)
