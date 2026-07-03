"""PS4 Problem 6: Balancing an inverted pendulum with model-based RL.

The cart-pole state space is discretized into 163 states.  We do not know the
transition dynamics, so we learn them by counting: run the current greedy
policy, tally (state, action, next_state) transitions and per-state rewards, and
whenever the pole falls, re-estimate the MDP and re-solve for the optimal value
function with value iteration.  Learning is declared converged once
NO_LEARNING_THRESHOLD consecutive value-iteration solves each finish in a single
sweep.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

from env import CartPole, Physics


def initialize_mdp_data(num_states):
    """Empty transition/reward counts, uniform transition model, tiny values."""
    return {
        'transition_counts': np.zeros((num_states, num_states, 2)),
        'transition_probs': np.ones((num_states, num_states, 2)) / num_states,
        # reward_counts[s, 0] = #times reward -1 seen entering s; [s, 1] = #visits
        'reward_counts': np.zeros((num_states, 2)),
        'reward': np.zeros(num_states),
        'value': np.random.rand(num_states) * 0.1,
        'num_states': num_states,
    }


def choose_action(state, mdp_data):
    """Greedy action w.r.t. the current value function and model."""
    expected = mdp_data['value'] @ mdp_data['transition_probs'][state]
    return int(np.argmax(expected))


def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    """Accumulate one observed transition and its reward."""
    mdp_data['transition_counts'][state, new_state, action] += 1
    if reward == -1:
        mdp_data['reward_counts'][new_state, 0] += 1
    mdp_data['reward_counts'][new_state, 1] += 1


def update_mdp_transition_probs_reward(mdp_data):
    """Re-estimate transition probabilities and rewards from the counts.

    State-action pairs never observed keep their uniform initialization.
    """
    counts = mdp_data['transition_counts']
    totals = counts.sum(axis=1)  # (num_states, 2)
    num_states = counts.shape[0]
    for s in range(num_states):
        for a in range(2):
            if totals[s, a] > 0:
                mdp_data['transition_probs'][s, :, a] = counts[s, :, a] / totals[s, a]

    rc = mdp_data['reward_counts']
    visited = rc[:, 1] > 0
    mdp_data['reward'][visited] = -rc[visited, 0] / rc[visited, 1]


def update_mdp_value(mdp_data, tolerance, gamma):
    """Value iteration to convergence; return True if it converged in one sweep."""
    probs = mdp_data['transition_probs']
    iters = 0
    while True:
        iters += 1
        value = mdp_data['value']
        # For each state, max over actions of E_{s'}[V(s')].
        expected = np.tensordot(value, probs, axes=([0], [1]))  # (num_states, 2)
        new_value = mdp_data['reward'] + gamma * expected.max(axis=1)
        mdp_data['value'] = new_value
        if np.max(np.abs(new_value - value)) < tolerance:
            break
    return iters == 1


def main(plot=True, seed=0, data_dir=None):
    np.random.seed(seed)

    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20
    max_failures = 500

    cart_pole = CartPole(Physics())
    state_tuple = (0.0, 0.0, 0.0, 0.0)
    state = cart_pole.get_state(state_tuple)
    mdp_data = initialize_mdp_data(NUM_STATES)

    time = 0
    time_at_start = 0
    num_failures = 0
    time_steps_to_failure = []
    consecutive_no_learning_trials = 0

    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        action = choose_action(state, mdp_data)
        state_tuple = cart_pole.simulate(action, state_tuple)
        time += 1
        new_state = cart_pole.get_state(state_tuple)
        R = -1 if new_state == NUM_STATES - 1 else 0

        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        if new_state == NUM_STATES - 1:
            update_mdp_transition_probs_reward(mdp_data)
            converged = update_mdp_value(mdp_data, TOLERANCE, GAMMA)
            consecutive_no_learning_trials = (
                consecutive_no_learning_trials + 1 if converged else 0)

            num_failures += 1
            if num_failures >= max_failures:
                break
            time_steps_to_failure.append(time - time_at_start)
            time_at_start = time

            # Reinitialize with a random cart position.
            x = -1.1 + np.random.uniform() * 2.2
            state_tuple = (x, 0.0, 0.0, 0.0)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    tstf = np.array(time_steps_to_failure)
    print(f'[p06] converged after {num_failures} failures (seed={seed})')
    print(f'[p06] steps-to-failure: first trial={tstf[0]}, '
          f'last trial={tstf[-1]}, best={tstf.max()}')
    print(f'[p06] mean over final 20 trials = {tstf[-20:].mean():.0f} steps')

    if plot:
        os.makedirs('output', exist_ok=True)
        log_tstf = np.log(tstf)
        plt.figure()
        plt.plot(np.arange(len(tstf)), log_tstf, 'k')
        window = 30
        w = np.full(window, 1 / window)
        weights = lfilter(w, 1, log_tstf)
        xs = np.arange(window // 2, len(log_tstf) - window // 2)
        plt.plot(xs, weights[window:len(log_tstf)], 'r--')
        plt.xlabel('Failure number')
        plt.ylabel('log(steps balanced before failure)')
        plt.title(f'Cart-pole learning curve (seed={seed})')
        plt.savefig(f'output/control_{seed}.png')
        plt.close()

    return tstf


if __name__ == '__main__':
    main()
