"""
Basic implementation of the Affine-invariant stretch move.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

global times, y, y_err

# generate some fake data
times = np.linspace(0, 10, 1_00)
m_true = 50
y = times * m_true
y_err = 10


def calculate_log_posterior_prob(m, b, times=times, y=y, y_err=y_err):
    model_prediction = m * times + b
    residual = model_prediction - y
    chi2_array = (residual / y_err) ** 2
    log_posterior_prob = -0.5 * np.sum(chi2_array)

    return log_posterior_prob


def draw_from_1_over_sqrt(num_samples=1, run_unit_test=False, a=50):
    x = np.random.uniform(0, 1, size=num_samples)
    c = 0.5 * (a ** (1 / 2) - a ** (-1 / 2)) ** (-1)
    sample = (x / (2 * c) - np.sqrt(a)) ** 2

    if run_unit_test:
        samples = draw_from_1_over_sqrt(10000)
        plt.figure()
        plt.hist(samples, bins=100, density=True, label="random samples")

        x = np.linspace(1 / a, a, int(1e6))
        c = 0.5 * (a ** (1 / 2) - a ** (-1 / 2)) ** (-1)
        assert np.isclose(
            np.sum(c / np.sqrt(x) * np.diff(x)[0]), 1, atol=1e-4
        )  # assert it integrates to 1
        plt.plot(x, c / np.sqrt(x), label="analytical")
        plt.xlabel("z")
        plt.ylabel("1/sqrt(z)")
        plt.legend()
        plt.savefig("inverse_zsquared_samples.png", dpi=250)
    return sample.squeeze()


def get_next_state_single_walker(
    m_current, b_current, m_complementary, b_complementary
):
    # pick complementary m, b
    compl_idx = np.random.choice(np.arange(len(m_complementary)))
    m_comp = m_complementary[compl_idx]
    b_comp = b_complementary[compl_idx]

    # draw magnitude from 1/sqrt(z)
    z = draw_from_1_over_sqrt()

    # calculate new positon: add magnitude of z
    m_diff = m_current - m_comp
    b_diff = b_current - b_comp
    slope_of_connecting_line = b_diff / m_diff

    b_new = z * np.sin(slope_of_connecting_line)
    m_new = z * np.cos(slope_of_connecting_line)

    # accept/reject
    n_dim = 2
    log_acceptance_prob = np.min(
        [
            0,
            np.log(z ** (n_dim - 1))
            + calculate_log_posterior_prob(m_new, b_new)
            - calculate_log_posterior_prob(m_current, b_current),
        ]
    )

    log_random_number = np.log(np.random.uniform(0, 1, size=1))
    if log_acceptance_prob >= log_random_number:
        m_current = m_new
        b_current = b_new

    return m_current, b_current


def affine_invariant_mcmc(
    m_initial_guess=50, b_initial_guess=0, num_walkers=50, num_steps=200
):

    m_markov_chain = np.zeros((num_walkers, num_steps))
    b_markov_chain = np.zeros((num_walkers, num_steps))

    m_ensemble = np.random.normal(m_initial_guess, scale=100, size=num_walkers)
    b_ensemble = np.random.normal(b_initial_guess, scale=100, size=num_walkers)

    for i in range(num_steps):
        print(f"{i+1}/{num_steps} steps.", end="\r")
        for j in range(num_walkers):

            m_current, b_current = get_next_state_single_walker(
                m_ensemble[j],
                b_ensemble[j],
                np.delete(m_ensemble, j),
                np.delete(b_ensemble, j),
            )

            m_ensemble[j] = m_current
            b_ensemble[j] = b_current

            m_markov_chain[j, i] = m_current
            b_markov_chain[j, i] = b_current

    return m_markov_chain, b_markov_chain


if __name__ == "__main__":
    np.random.seed(10)

    num_steps = 1000
    num_walkers = 100

    m_markov_chain, b_markov_chain = affine_invariant_mcmc(
        num_walkers=num_walkers, num_steps=num_steps
    )

    fig, ax = plt.subplots(2, 1)

    for i in range(num_walkers):
        ax[0].plot(np.arange(num_steps), m_markov_chain[i, :], color="k", alpha=0.1)
        ax[1].plot(np.arange(num_steps), b_markov_chain[i, :], color="k", alpha=0.1)
    ax[1].set_xlabel("step")
    ax[0].set_ylabel("m")
    ax[1].set_ylabel("b")
    plt.savefig("AI_markov_chain.png", dpi=250)
    plt.close()

    # make a fun gif
    os.system("rm gif/*")
    print()
    for i in range(num_steps):
        print(f"Making gif frame {i}/{num_steps}.", end="\r")
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.text(-20, 200, f"Step {i+1}/{num_steps}")

        plt.scatter(
            m_markov_chain[:, i],
            b_markov_chain[:, i],
            color="grey",
            alpha=0.5,
        )

        plt.xlim(-30, 80)
        plt.ylim(-50, 250)
        plt.xlabel("m")
        plt.ylabel("b")
        plt.savefig(f"gif/step{i}.png", dpi=250)
        plt.close()

    frames = [Image.open(f"gif/step{i}.png") for i in range(num_steps)]

    print()
    print("Saving gif...")
    frames[0].save(
        "affine_sampler.gif",
        save_all=True,
        append_images=frames[1:],
        duration=20,
        loop=0,  # 0 means infinite loop
    )
