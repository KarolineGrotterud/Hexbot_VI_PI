#from audioop import reverse
#from tkinter import RIGHT
#from urllib import request
import sys
import time
from constants import *
from environment import *
from state import State
import numpy as np
#from typing import Dict

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
        #
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.

        """I just want to state that all of my code is "inspired" from the tutorial 7 code. """
        self.state_list = []
        self.state_values = None
        self.policy = None
        #self.converged = False
        self.converged = False
        self.probabilities = {}
        #

        
        
        pass

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        # Make a list of possible game states (robot and widget pos and orients)
        


        # Write a function which loops over each possible result of environment.apply_action_noise() 
        # (producing a list of sequences of movements) and call environment.apply_dynamics() for each movement 
        # (possibly multiple times if drift or double move occurs) to get a list of possible next states -> you can treat this similar
        # way to the get_successors() function we used in Module 1 (Search). 

        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here. You should not
        #  perform value iteration in this method.
        

        # frontier = [State(self.environment, self.environment.robot_init_posit, self.environment.robot_init_orient, self.environment.widget_init_posits,
        #              self.environment.widget_init_orients)]    
        # stateList = [State(self.environment, self.environment.robot_init_posit, self.environment.robot_init_orient, self.environment.widget_init_posits,
        #              self.environment.widget_init_orients)]

        # while len(frontier) > 0:
        #     current_state = frontier.pop()

        #     for i in ROBOT_ACTIONS:
        #         possibleMovements = self.environment.apply_action_noise(i)
        #         for i in possibleMovements:
        #             reward, nextState = self.environment.apply_dynamics(current_state,i)
        #             #print(reward,nextState)
        #             if nextState not in stateList:
        #                 stateList.append(nextState)
        #                 frontier.append(nextState)
        # self.stateList = stateList

        init_state = self.environment.get_init_state()
        self.stateList = bfs(State(self.environment, init_state.robot_posit, init_state.robot_orient, init_state.widget_centres, init_state.widget_orients))

        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.state_values = {state: 0 for state in self.stateList}
        self.policy = {state: FORWARD for state in self.stateList}
        self.differences = []

        pass

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.converged

        pass

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        #
        # TODO: Implement code to perform a single iteration of Value Iteration here.
        new_state_values = dict()
        new_policy_value = dict()
        max_diff = 0.0
        for state in self.stateList:
            action_values = dict()  #Going ot contain, for each action, whats the value of the action
            if self.environment.is_solved(state):
                new_state_values[state] = 0
            else:
                for action in ROBOT_ACTIONS: #loops through forward, backward, left, right
                    action_value = 0        #The value for the action
                    next_state = state
                    min_reward, _ = self.environment.apply_dynamics(state,action)
                    for stoch_action, probability in self.stoch_action(action).items():
                        for i in stoch_action:
                            reward, next_state = self.environment.apply_dynamics(next_state,i)
                        action_value += probability * (min_reward + (self.environment.gamma * self.state_values[next_state]))
                    action_values[action] = action_value
                new_state_values[state] = max(action_values.values())
                new_policy_value[state] = dict_argmax(action_values)

             
        differences = [abs(self.state_values[state] - new_state_values[state]) for state in self.stateList]
        max_diff = max(differences)
        self.differences.append(max_diff)
        #print(max_diff)
        if max_diff < self.environment.epsilon:
            self.converged = True
        
        self.state_values = new_state_values
        self.policy = new_policy_value
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
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
        if state not in self.state_values.keys():
            return 0
        else:
            return self.state_values[state]
        
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #for i in self.state_values:

        return self.policy[state]

        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        # frontier = [State(self.environment, self.environment.robot_init_posit, self.environment.robot_init_orient, self.environment.widget_init_posits,
        #              self.environment.widget_init_orients)]    
        # stateList = [State(self.environment, self.environment.robot_init_posit, self.environment.robot_init_orient, self.environment.widget_init_posits,
        #              self.environment.widget_init_orients)]

        # while len(frontier) > 0:
        #     current_state = frontier.pop()

        #     for i in ROBOT_ACTIONS:
        #         possibleMovements = self.environment.apply_action_noise(i)
        #         for i in possibleMovements:
        #             reward, nextState = self.environment.apply_dynamics(current_state,i)
        #             #print(reward,nextState)
        #             if nextState not in stateList:
        #                 stateList.append(nextState)
        #                 frontier.append(nextState)
        # self.stateList = stateList

        init_state = self.environment.get_init_state()
        self.state_list = bfs(State(self.environment, init_state.robot_posit, init_state.robot_orient, init_state.widget_centres, init_state.widget_orients))
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #


        #Making the matrix for the linear solver
        self.t_model = np.zeros([len(self.state_list), len(ROBOT_ACTIONS), len(self.state_list)])
        for i, s in enumerate(self.state_list):
            for j, a in enumerate(ROBOT_ACTIONS):
                transitions = self.get_transition_probabilities(s,a)
                for next_state, prob in transitions.items():
                    self.t_model[i][j][self.state_list.index(next_state)] = round(prob,4)

        
        r_model = np.zeros([len(self.state_list)])
        for i,s in enumerate(self.state_list):           
            for j, a in enumerate(ROBOT_ACTIONS):       
                r_model[i] = self.get_reward(s,a)
        self.r_model = r_model



        la_policy = np.zeros([len(self.state_list)], dtype=np.int64)
        for i, s in enumerate(self.state_list):
            la_policy[i] = 0
        self.la_policy = la_policy

        self.state_values = {state: 0 for state in self.state_list}
        self.policy = {pi: FORWARD for pi in self.state_list}

        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        return self.converged 
    
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #   The reward vector needs to be updatec each iteration to match the latest policy see #422 ED


        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        new_value = dict()
        
        new_policy = dict()
        self.policy_evaluation()
        new_policy = self.policy_improvement()
        self.check_convergence(new_policy)

        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #Remeber to update the reward vector, 
        pass

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        return self.policy[state]

        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    def stoch_action(self, a):
        if a == FORWARD:
            prob_all = (1 - self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*(1 - self.environment.double_move_probs[a])

            return {(FORWARD,): round(prob_all,4),
                    (FORWARD, FORWARD): round((1-self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*self.environment.double_move_probs[a], 4),
                    (SPIN_LEFT, FORWARD): round(self.environment.drift_ccw_probs[a]*(1-self.environment.double_move_probs[a]), 4),
                    (SPIN_RIGHT,FORWARD): round(self.environment.drift_cw_probs[a]*(1-self.environment.double_move_probs[a]),4), 
                    (SPIN_RIGHT, FORWARD, FORWARD): round(self.environment.drift_cw_probs[a]*self.environment.double_move_probs[a],4), 
                    (SPIN_LEFT, FORWARD, FORWARD): round(self.environment.drift_ccw_probs[a]*self.environment.double_move_probs[a],4)}

        elif a == REVERSE:
            prob_all = (1 - self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*(1 - self.environment.double_move_probs[a])
            return {(REVERSE,): round(prob_all,4),
                    (REVERSE, REVERSE) : round((1-self.environment.drift_ccw_probs[a] - self.environment.drift_cw_probs[a])*self.environment.double_move_probs[a],4), 
                    (SPIN_LEFT, REVERSE) : round(self.environment.drift_ccw_probs[a]*(1-self.environment.double_move_probs[a]),4), 
                    (SPIN_RIGHT,REVERSE): round(self.environment.drift_cw_probs[a]*(1-self.environment.double_move_probs[a]),4), 
                    (SPIN_LEFT,REVERSE,REVERSE): round((self.environment.drift_ccw_probs[a]*self.environment.double_move_probs[a]),4), 
                    (SPIN_RIGHT, REVERSE, REVERSE): round((self.environment.drift_cw_probs[a]*self.environment.double_move_probs[a]),4)}
        elif a == SPIN_LEFT:
            return {(SPIN_LEFT,) : 1}
        elif a == SPIN_RIGHT:
            return {(SPIN_RIGHT,): 1}

    
    def print_values(self):
        for state, value in self.state_values.items():
            print(state, value)
    
    def print_policy(self):

        for state, action in self.policy.items():
            print(state, ROBOT_ACTIONS[action])
    
    def get_transition_probabilities(self, s, a):
        probabilities = {}
        next_state = s
        for stoch_action, probability in self.stoch_action(a).items():
            for i in stoch_action:
                _,next_state = self.environment.apply_dynamics(next_state, i)
            if self.environment.is_solved(s):
                probabilities[next_state] = 0
            else:
                probabilities[next_state] = probabilities.get(next_state, 0) + probability
        return probabilities

    def get_reward(self, s, a):
        ## Returning the reward for given state. We have a reward vector/dictionary. It gives us the reward for being in said state given that we 
        # are going to follow the current policy. For each entry, we calculate the expected reward, which is a sum of prob*reward over the 
        # the 6 possible actions that could occur. 
        if self.environment.is_solved(s):
            return 0
        if self.policy == None:
            return -10
        else:
            reward = 0
            cost, _ = self.environment.apply_dynamics(s, a)
            for _, p in self.stoch_action(a).items():
                reward += (p * cost)
            return reward
     #Just returning a cost when the policy dict is empty. Not very elegant, but works fine.


    def policy_evaluation(self):
        state_numbers = np.array(range(len(self.state_list))) 
        t_pi = self.t_model[state_numbers, self.la_policy]     #t_i inneholder raden med actions for en state
        values = np.linalg.solve(np.identity(len(self.state_list)) - (self.environment.gamma * t_pi), self.r_model)
        self.state_values = {s: values[i] for i, s in enumerate(self.state_list)}
    
    def policy_improvement(self):
        new_policy = {s: ROBOT_ACTIONS[self.la_policy[i]] for i, s in enumerate(self.state_list)}

        for state in self.state_list:
            action_values = dict()  #Going ot contain, for each action, whats the value of the action
            for a in ROBOT_ACTIONS: #loops through forward, backward, left, right
                total = 0        #The value for the action
                next_state = state
                min_reward, _ = self.environment.apply_dynamics(state,a)
                for stoch_action, probability in self.stoch_action(a).items():
                    for i in stoch_action:
                        reward, next_state = self.environment.apply_dynamics(next_state,i)
                    total += probability * (min_reward + (self.environment.gamma * self.state_values[next_state]))
                action_values[a] = total
            new_policy[state] = dict_argmax(action_values)
        return new_policy

    def check_convergence(self, new_policy):
        if new_policy == self.policy:
            self.converged = True
        self.policy = new_policy

        for i, s in enumerate(self.state_list):
            self.la_policy[i] = self.policy[s]


def dict_argmax(d):
    max_value = max(d.values()) # TODO handle multiple keys with the same max value
    for key, value in d.items():
        if value == max_value:
            return key


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