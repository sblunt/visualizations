import matplotlib.pyplot as plt
import numpy as np

global sigma_proposal, times, y, y_err

# choose mcmc hyperparameter
sigma_proposal = 50

# generate some fake data
times = np.linspace(0, 10, 1_00)
m_true = 50
y = times * m_true
y_err = 10


def calculate_posterior_prob(m, times=times, y=y, y_err=y_err):
    model_prediction = m * times
    residual = model_prediction - y
    chi2_array = (residual / y_err) ** 2
    chi2 = np.sum(chi2_array)

    posterior_prob = np.exp(-0.5 * chi2)
    return posterior_prob


def pick_next_state(m_current):
    m_new = np.random.normal(loc=m_current, scale=sigma_proposal)
    return m_new


# def calculate_proposal_prob(m_new, m_current):
#     proposal_prob = norm.pdf(m_new, loc=m_current, scale=sigma_proposal)
#     return proposal_prob


def calculate_acceptance_prob(m_current, m_new):

    acceptance_prob = np.min(
        [
            1,
            calculate_posterior_prob(m_new) / calculate_posterior_prob(m_current),
            # * calculate_proposal_prob(m_current, m_new)
            # / calculate_proposal_prob(m_new, m_current),
        ]
    )

    return acceptance_prob


def mh_mcmc(m_initial_guess=47, num_steps=200):

    markov_chain = []

    m_current = m_initial_guess
    for i in range(num_steps):
        m_proposal = pick_next_state(m_current)
        acceptance_prob = calculate_acceptance_prob(m_current, m_proposal)

        random_number = np.random.uniform(0, 1, size=1)
        if acceptance_prob >= random_number:
            m_current = m_proposal

        markov_chain.append(m_current)

    return markov_chain


if __name__ == "__main__":
    np.random.seed(10)

    m_initial_guess = 48
    num_steps = 5_000

    markov_chain = mh_mcmc(m_initial_guess=m_initial_guess, num_steps=num_steps)

    plt.plot(markov_chain)
    plt.xlabel("steps")
    plt.ylabel("m")
    plt.savefig("MH_markov_chain.png", dpi=250)
