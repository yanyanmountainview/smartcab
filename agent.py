import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.gamma = 0.8
        self.alpha = 0.5
        self.epsilon = 0.1
        self.epsilon_decay_rate = 0.0001
        self.valid_actions = [None, 'forward', 'left', 'right']


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    # build a state by taking parameters from inputs and next_way point
    def return_state(self, inputs):
        return (
            inputs['light'],
            inputs['oncoming'],
            inputs['left'],
            # inputs['right'],
            self.next_waypoint,
            # self.env.get_deadline(self)
        )

    # read Q_value from Q_table based on state and action
    def get_Q_value(self, state, action):
        if (state, action) in self.Q_table:
            return self.Q_table[(state, action)]
        else:
            return 0

    # return the max Q_value based on the state and all possible actions
    def get_Max_Q(self, state):
        max_Q_value = 0
        for action in self.valid_actions:
            if max_Q_value < self.get_Q_value(state, action):
                max_Q_value = self.get_Q_value(state, action)
        return max_Q_value

    # update Q_value in Q_Table
    def update_Q_value(self, state, action, reward):
        exist_Q_value = self.get_Q_value(state, action)
        self.next_waypoint = self.planner.next_waypoint()
        next_state = self.return_state(self.env.sense(self))
        Q_value = exist_Q_value + self.alpha*(reward + self.gamma * self.get_Max_Q(next_state) - exist_Q_value)
        self.Q_table[(state, action)] = Q_value

    # define action policy that take the action which result in the max Q_value from the current state
    def action_policy(self, state):
        action_to_take = None
        best_Q_value = 0
        for action in self.valid_actions:
            if self.get_Q_value(state, action) > best_Q_value:
                best_Q_value  = self.get_Q_value(state, action)
                action_to_take = action
            elif self.get_Q_value(state, action) == best_Q_value:
                action_to_take = action
        return action_to_take

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # get the state parameters and display them.
        self.state = self.return_state(inputs)

        # TODO: Select action according to your policy

        # action  = random.choice(Environment.valid_actions[1:])

        # select action according to the policy with greedy strategy
        if self.epsilon > random.random():
            action = random.choice(self.valid_actions)
        else:
            action = self.action_policy(self.state)
            self.epsilon -= self.epsilon_decay_rate

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_Q_value(self.state, action, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.005, display = False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
