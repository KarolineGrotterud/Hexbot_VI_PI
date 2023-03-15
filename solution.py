import sys
import time
from urllib.robotparser import RobotFileParser
from constants import *
from environment import *
from state import State
import math
import numpy as np
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22
"""


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.values = None
        self.policy = None
        self.converged = False
        self.policy_converged = False
        self.state_list = []

        
        #
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.
        #
        pass

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        #
        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here. You should not
        #  perform value iteration in this method.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        init_state = self.environment.get_init_state()
        self.state_list = bfs(State(self.environment, init_state.robot_posit, init_state.robot_orient, init_state.widget_centres, init_state.widget_orients))
        print(len(self.state_list))
        self.values = {state: 0 for state in self.state_list}
        self.policy = {state: FORWARD for state in self.state_list} #Random action?
        self.differences = []

        pass


    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        return self.converged


    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        #
        # TODO: Implement code to perform a single iteration of Value Iteration here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        new_values = dict()
        new_policy = dict()
        for s in self.state_list:
            action_values = dict()
            if self.environment.is_solved(s):
                new_values[s] = 0
            else:
                for a in ROBOT_ACTIONS:
                    #print('not goal-state')
                    total = 0
                    s_next = s
                    reward, _ = self.environment.apply_dynamics(s, a)
                    for stoch_action, p in self.stoch_action(a).items():
                        for move in stoch_action:
                            _, s_next = self.environment.apply_dynamics(s_next, move)
                        total += p * (reward + (self.environment.gamma * self.values[s_next]))
                    action_values[a] = total
                new_values[s] = max(action_values.values())
                new_policy[s] = dict_argmax(action_values)

        differences = [abs(self.values[s] - new_values[s]) for s in self.state_list]
        max_diff = max(differences)
        self.differences.append(max_diff)

        if max_diff < self.environment.epsilon:
            self.converged = True
        
        self.values = new_values
        self.policy = new_policy
        #print('max diff: ', max_diff)

        pass

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        #
        # TODO: Implement code to return the value V(s) for the given state (based on your stored VI values) here. If a
        #  value for V(s) has not yet been computed, this function should return 0.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        return self.values[state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        init_state = self.environment.get_init_state()
        self.state_list = bfs(State(self.environment, init_state.robot_posit, init_state.robot_orient, init_state.widget_centres, init_state.widget_orients))

        self.t_model = np.zeros([len(self.state_list), len(ROBOT_ACTIONS), len(self.state_list)])
        for i, s in enumerate(self.state_list):
            for j, a in enumerate(ROBOT_ACTIONS):
                transitions = self.get_transition_probabilities(s, a)
                for next_state, prob in transitions.items():
                    self.t_model[i][j][self.state_list.index(next_state)] = round(prob,4)

        # Reward matrix
        r_model = np.zeros([len(self.state_list), len(ROBOT_ACTIONS)])
        for i, s in enumerate(self.state_list):
            for j, a in enumerate(ROBOT_ACTIONS):
                r_model[i][j] = self.get_reward(s,a)
        self.r_model = r_model
        # # print(r_model)

    
       # lin alg policy
        la_policy = np.zeros([len(self.state_list)], dtype=np.int64)
        for i, s in enumerate(self.state_list):
            la_policy[i] = 0 # Allocate arbitrary initial policy, FORWARD
        self.la_policy = la_policy

        self.values = {state: 0 for state in self.state_list}
        self.policy = {pi: FORWARD for pi in self.state_list}


        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        return self.policy_converged 


    def pi_iteration(self):

        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        new_values = dict()
        new_policy = dict()

        self.policy_evaluation()
        new_policy = self.policy_improvement()
        self.convergence_check(new_policy)

        pass

    def policy_evaluation(self):
        
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        state_numbers = np.array(range(len(self.state_list)))  # indices of every state
        t_pi = self.t_model[state_numbers, self.la_policy]
        #print('Dimensions of t_pi: ', t_pi.shape)
        r_vec = np.full(len(self.state_list), -10)  
        for s_index in state_numbers:
            r_vec[s_index] = self.r_model[s_index,self.la_policy[s_index]]
        #print('Size r_vec', len(r_vec))
        values = np.linalg.solve(np.identity(len(self.state_list)) - (self.environment.gamma * t_pi), r_vec)
        self.values = {s: values[i] for i, s in enumerate(self.state_list)}

    def policy_improvement(self):
        #policy improvement
        new_policy = {s: ROBOT_ACTIONS[self.la_policy[i]] for i, s in enumerate(self.state_list)}

        for s in self.state_list:
            # Keep track of maximum value
            action_values = dict()
            for a in ROBOT_ACTIONS:
                total = 0
                s_next = s
                reward,_ = self.environment.apply_dynamics(s, a)
                for stoch_action, p in self.stoch_action(a).items():
                    for move in stoch_action:
                    # Apply action
                        _,s_next = self.environment.apply_dynamics(s_next, move)
                    total += p * (reward + (self.environment.gamma * self.values[s_next]))
                action_values[a] = total
            # Update policy
            new_policy[s] = dict_argmax(action_values)
        #self.print_policy()
        return new_policy

        # if new_policy == self.policy:
        #     self.converged = True

        # self.policy = new_policy
        # for i, s in enumerate(self.state_list):
        #     self.la_policy[i] = self.policy[s]

        # for i, s in enumerate(self.state_list):
        #     self.r_model[i] = self.get_reward(s)

        pass

    def convergence_check(self, new_policy):
        diff = 1716
        for i, s in enumerate(self.state_list):
            if self.policy[s] == new_policy[s]:
                diff -=1
        print('Difference in pi: ', diff)

        if new_policy == self.policy:
            self.policy_converged = True
        self.policy = new_policy

        for i, s in enumerate(self.state_list):
            self.la_policy[i] = self.policy[s]


    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
    #     """
    #     Retrieve the optimal action for the given state (based on values computed by Value Iteration).
    #     :param state: the current state
    #     :return: optimal action for the given state (element of ROBOT_ACTIONS)
    #     """
    #     #
    #     # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
    #     #
    #     # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
    #     #

        return self.policy[state]
        
    #     pass

    # # === Helper Methods ===============================================================================================
    # #
    # #
    # # TODO: Add any additional methods here
    # #
    # #

    def stoch_action(self, a):
        """ Returns the probabilities with which each action will actually occur,
            given that action a was requested.

        The keys in the dictionary are possible movements represented in ROBOT_ACTIONS:
            CW, CCW, CW + double, CCW + double, double move, action
        Parameters:
            a: The action requested by the agent.

        Returns:
            The probability distribution over actual actions that may occur.
        """

        cw_prob = self.environment.drift_cw_probs[a]*(1-self.environment.double_move_probs[a])
        ccw_prob = self.environment.drift_ccw_probs[a]*(1-self.environment.double_move_probs[a])
        cw_double_prob = self.environment.drift_cw_probs[a]*self.environment.double_move_probs[a]
        ccw_double_prob = self.environment.drift_ccw_probs[a]*self.environment.double_move_probs[a]
        double_prob = self.environment.double_move_probs[a]*(1-self.environment.drift_cw_probs[a]-self.environment.drift_ccw_probs[a])

        if a == FORWARD:
            prob_single_action = (1 - self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*(1 - self.environment.double_move_probs[a])

            return {(FORWARD,): round(prob_single_action,4),
                    (FORWARD, FORWARD): round(double_prob, 4),
                    (SPIN_LEFT, FORWARD): round(ccw_prob, 4),
                    (SPIN_RIGHT,FORWARD): round(cw_prob,4), 
                    (SPIN_RIGHT, FORWARD, FORWARD): round(cw_double_prob,4), 
                    (SPIN_LEFT, FORWARD, FORWARD): round(ccw_double_prob,4)}

        elif a == REVERSE:
            prob_single_action = (1 - self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*(1 - self.environment.double_move_probs[a])
            return {(REVERSE,): round(prob_single_action,4),
                    (REVERSE, REVERSE) : round(double_prob,4), 
                    (SPIN_LEFT, REVERSE) : round(ccw_prob,4), 
                    (SPIN_RIGHT,REVERSE): round(cw_prob,4), 
                    (SPIN_RIGHT, REVERSE, REVERSE): round(cw_double_prob,4),
                    (SPIN_LEFT,REVERSE,REVERSE): round(ccw_double_prob,4)}
        elif a == SPIN_LEFT:
            return {(SPIN_LEFT,) : 1}
        elif a == SPIN_RIGHT:
            return {(SPIN_RIGHT,): 1}

    # def print_values(self):
    #     for state, value in self.values.items():
    #         print(state, value)
    
    # def print_policy(self):
    #     for state, policy in self.policy.items():
    #         print(state, policy)

    def get_transition_probabilities(self, s, a):
        probabilities = {}
        s_next = s
        for stoch_action, p in self.stoch_action(a).items():
            for move in stoch_action:
                _,s_next = self.environment.apply_dynamics(s_next,move)
            probabilities[s_next] = probabilities.get(s_next, 0) + p
            if self.environment.is_solved(s):
                probabilities[s_next] = 0
        return probabilities

    def get_reward(self, state: State, action):

        if self.environment.is_solved(state):
            reward = 0
        if self.policy == None:
            reward = -10
        else:
            reward = 0
            cost,_ = self.environment.apply_dynamics(state,action)
            for _, p in self.stoch_action(action).items():
                reward += (cost*p)
        return reward

def bfs(state: State):
    """
    Bredth First Search from the input state.
    Purpose: run this func on the init state to make list over all valid states
    """
    container = [state]
    visited = [state]

    while len(container) > 0: #While frontier is not empty
        node = container.pop()

        for action in ROBOT_ACTIONS:
            movements = node.environment.apply_action_noise(action)
            for move in movements:
                _, next_state = node.environment.apply_dynamics(node, move)
                if next_state not in visited:
                    visited.append(next_state)
                    container.append(next_state) 

    return visited

def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k


