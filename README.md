# POLICY ITERATION ALGORITHM
## AIM

To implement the Value Iteration algorithm and compute the optimal value function and optimal deterministic policy for a given finite Markov Decision Process (MDP). Evaluate the learned policy by measuring its empirical success rate when executed on the environment.

## PROBLEM STATEMENT

Given an MDP with finite states and actions and transition dynamics represented as P[s][a] = [(prob, next_state, reward, done), ...], compute:

The optimal state-value function V* (expected maximum discounted return from each state).

The optimal deterministic policy π* (action index for each state that maximizes expected return).

The empirical success rate of π* by simulating a fixed number of episodes on the environment.

## POLICY ITERATION ALGORITHM (Steps)

Initialize V(s) = 0 for all states s.

Repeat until convergence (maximum change delta < theta):

For each state s, compute action-values Q(s,a) = sum_{s',r} P(s'|s,a) * [r + gamma * V(s')] for all actions a.

Update V(s) <- max_a Q(s,a).

After convergence, extract the greedy policy: pi(s) = argmax_a Q(s,a).

(Optional) Evaluate pi by running multiple episodes to compute success rate and average return.

## POLICY ITERATION FUNCTION
```
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,mean_return(env, pi_frozenlake)))

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V

V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

def policy_improvement(V,P,gamma=1.0):
  Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
  for s in range(len(P)):
    for a in range(len(P[s])):
      for prob,next_state,reward,done in P[s][a]:
        Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
  new_pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
  return new_pi

pi_2 = lambda s: {
    0: DOWN,
    1: RIGHT,
    2: RIGHT,
    3: DOWN,
    4: LEFT,
    5: RIGHT,
    6: DOWN,
    7: LEFT,
    8: DOWN,
    9: RIGHT,
    10: LEFT,
    11: DOWN,
    12: RIGHT,
    13: DOWN,
    14: RIGHT,
    15: LEFT  # Stop at goal
}[s]

pi_2 = policy_improvement(V1, P)
print('Name: CHANDRAPRIYADHARSHINI C        Register Number: 212223240019')
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

V2 = policy_evaluation(pi_2, P, gamma=0.99)

print("\nState-value function for Your Policy:")
print_state_value_function(V2, P, n_cols=4, prec=5)

if(np.sum(V1>=V2)==16):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==16):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")

def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  ramdon_actions=np.random.choice(tuple (P[0].keys()), len(P))
  pi=lambda s: {s: a for s, a in enumerate(random_actions)} [s]
  while True:
    old_pi={s:pi(s) for s in range (len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi

optimal_V, optimal_pi = policy_iteration(P)

print('Name: Chandrapriyadharshini C')
print('Register No: 212223240019')
print()
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))

print_state_value_function(optimal_V, P, n_cols=4, prec=5)
```

## OUTPUT
#### Optimal policy and state-value function

<img width="647" height="210" alt="image" src="https://github.com/user-attachments/assets/34435563-988d-4f73-b51c-dd4ae35b9196" />

#### State-value function

<img width="718" height="131" alt="image" src="https://github.com/user-attachments/assets/b5eb61c4-1f00-457e-81b2-6ba3429da3da" />

## RESULT
Ths the program successfully executed.
