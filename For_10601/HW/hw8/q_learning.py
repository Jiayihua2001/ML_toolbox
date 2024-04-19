import argparse
import numpy as np

from environment import MountainCar, GridWorld

from typing import Union, Tuple, Optional # for type annotations

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *NOT* have the bias term 
folded in.
"""

def set_seed(seed: int):
    '''
    DO NOT MODIFY THIS FUNCITON.
    Sets the numpy random seed.
    '''
    np.random.seed(seed)


def round_output(places: int):
    '''
    DO NOT MODIFY THIS FUNCTION.
    Decorator to round output of a function to certain 
    number of decimal places. You do not need to know how this works.
    '''
    def wrapper(fn):
        def wrapped_fn(*args, **kwargs):
            return np.round(fn(*args, **kwargs), places)
        return wrapped_fn
    return wrapper


def parse_args() -> Tuple[str, str, str, str, int, int, float, float, float]:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        (env_type, mode, weight_out, returns_out, 
         episodes, max_iterations, epsilon, gamma, lr) = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return (args.env, args.mode, args.weight_out, args.returns_out, 
            args.episodes, args.max_iterations, 
            args.epsilon, args.gamma, args.learning_rate)


@round_output(5) # DON'T DELETE THIS LINE
def Q(W: np.ndarray, state: np.ndarray, 
      action: Optional[int] = None) -> Union[float, np.ndarray]:
    '''
    Helper function to compute Q-values for function-approximation 
    Q-learning.

    Note: Do not delete or move the `@round_output(5)` line on top of 
          this function. This just ensures that your Q-value is rounded to 5
          decimal places, which avoids some *pernicious* cross-platform 
          rounding errors.

    Parameters:
        W     (np.ndarray): Weight matrix with folded-in bias with 
                            shape (action_space, state_space+1).
        state (np.ndarray): State encoded as vector with shape (state_space,).
        action       (int): Action taken. Satisfies 0 <= action < action_space.

    Returns:
        If action argument was provided, returns float Q(state, action).
        Otherwise, returns array of Q-values for all actions from state,
        [Q(state, a_0), Q(state, a_1), Q(state, a_2)...] for all a_i.
    '''
    # TODO: Implement this!
    state_with_bias = np.append(1,state)  # Adding bias term
    Q_values = np.dot(W, state_with_bias)
    return Q_values if action is None else Q_values[action]
    raise NotImplementedError
    return 0.0


if __name__ == "__main__":
    set_seed(10301) # DON'T DELETE THIS

    # Read in arguments
    (env_type, mode, weight_out, returns_out, 
     episodes, max_iterations, epsilon, gamma, lr) = parse_args()

    # Create environment
    if env_type == "mc":
        env = MountainCar(mode=mode) # TODO: Replace me!
    elif env_type == "gw":
        env = GridWorld(mode=mode) # TODO: Replace me!
    else: 
        raise Exception(f"Invalid environment type {env_type}")

    # TODO: Initialize your weight matrix. Remember to fold in a bias!
    # W = ...
    W = np.zeros((env.action_space, env.state_space+1))

    for episode in range(episodes):

        # TODO: Get the initial state by calling env.reset()
        state= env.reset()
        returns = 0
        for iteration in range(max_iterations):

            # TODO: Select an action based on the state via the epsilon-greedy 
            #       strategy.
            def epsilon_greedy_action_selection(Q, epsilon,actions):
                if np.random.rand() < epsilon:
                    # Exploration: choose a random action
                    return np.random.choice(actions)
                else:
                    # Exploitation: choose the best action from Q-table
                    best_action = np.argmax(Q)
                    return best_action
                
            actions=[i for i in range(env.action_space)]
            
            Q_s=Q(W,state)
            action= epsilon_greedy_action_selection(Q_s,epsilon,actions)

            # TODO: Take a step in the environment with this action, and get the 
            #       returned next state, reward, and done flag.
            next_state,reward,done_flag = env.step(action)
            Q_true = reward + gamma* np.max(Q(W,next_state))

            # TODO: Using the original state, the action, the next state, and 
            #       the reward, update the parameters. Don't forget to update 
            #       the bias term!
            state_bias = np.append(1,state)
            W[action] -= lr*(Q(W,state,action)-Q_true)*state_bias 

            # TODO: Remember to break out of this inner loop if the environment 
            #       signals done!
            returns += reward
            state  = next_state
            if done_flag :
                state= env.reset()
                break
        with open(returns_out,'a') as f:
            f.write(str(returns))
            f.write('\n')

    # TODO: Save your weights and returns. The reference solution uses 
    np.savetxt(weight_out,W, fmt="%.18e", delimiter=" ")
    

