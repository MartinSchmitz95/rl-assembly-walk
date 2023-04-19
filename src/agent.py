from typing import Dict
import numpy as np
from model import GraphDQN
import random
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph


class RandomWalkAgent:
    def __init__(self,):
        pass

    def get_action(self, obs, legal_actions):
        # legal_actions.append(None)
        return random.choice(legal_actions)


class GreedyWalkAgent:
    def __init__(self, config: Dict):
        self.__config = config

    def get_action(self, obs, legal_actions):
        # legal_actions.append(None)
        dists = obs['graph'].x[:, 2]  # node distance slice
        max_dist = 0
        best_action = legal_actions[0]
        for action in legal_actions:
            d = dists[action[1]].item()
            if d > max_dist:
                best_action = action
                max_dist = d
            # print(best_action)
        # print(max_dist)
        return best_action


class AssemblyWalkAgent:
    def __init__(
        self,
        config, inference=False
    ):
        """
        Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.q_network = GraphDQN(config).to(config['device'])

        """if 'cuda' in config['device']:
            cuda_device_id = int(config['device'].split(":")[1])
            self.q_values = self.q_values.cuda(cuda_device_id)"""

        self.num_gnn_layers = config['num_gnn_layers']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']
        self.epsilon = config['start_epsilon']
        # reduce the exploration over time
        self.epsilon_decay = config['start_epsilon'] / (config['n_episodes'] / 2.)
        self.final_epsilon = config['final_epsilon']

        if inference:
            self.final_epsilon = 1
            self.final_epsilon = 1

        self.training_error = []

    def get_action(self, obs, legal_actions):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        threshold = np.random.random()
        if threshold < self.epsilon:
            legal_actions.append(None)
            return random.choice(legal_actions)
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return self.get_greedy_action(obs, legal_actions)

    def get_greedy_action(self, obs, legal_actions):
        """
        :param obs:
        :param legal_actions:
        :return: the best legal action according to the q-function
        """
        self.q_network.eval()
        # create k-hop subgraph
        subset, edge_index, _, _ = k_hop_subgraph(obs['agent_location'], self.num_gnn_layers,
                                                  obs['graph'].edge_index, relabel_nodes=True,
                                                  flow='target_to_source')  # directed=False
        _, e = subgraph(subset, obs['graph'].edge_index,
                        edge_attr=obs['graph'].edge_attr, relabel_nodes=False)
        x = obs['graph'].x[subset]

        # here the q-value of the env get be computed
        edge_values, node_values = self.q_network(edge_index, x, e)

        # find which node is the current agent position
        stop_action_index = torch.argwhere(
            subset == obs['agent_location']).item()
        stop_action = node_values[stop_action_index]

        # retrieve edge index ids, to check with q-value outputs are from the legal actions
        _, edge_index_norelabel, _, _ = k_hop_subgraph(obs['agent_location'], self.num_gnn_layers,
                                                       obs['graph'].edge_index, relabel_nodes=False,
                                                       flow='target_to_source')  # directed=False
        comp = edge_index_norelabel[0].numpy()
        edge_action_index = np.argwhere(comp == obs['agent_location'])
        print("hiiiii", edge_action_index)
        print(legal_actions)
        legal_actions_recomputed = edge_index_norelabel.T[edge_action_index.squeeze(
        )].view(-1, 2)
        # if len(legal_actions_recomputed) == 1:  # if only one legal action, the tensor is squeeed otherwise
        #    legal_actions_recomputed.unsqueeze(1)
        legal_q_action_values = edge_values[edge_action_index.squeeze()].view(-1, 1)
        # get best edge action and action indices
        action_value, action_index = torch.max(legal_q_action_values, dim=0)
        action_value = action_value.item()
        action_index = action_index.item()
        # action_index = torch.argmax(legal_q_action_values, dim=0)
        # compare best edge action with stop action and return best
        if stop_action > action_value:
            action = None
        else:
            source = legal_actions_recomputed[action_index][0].item()
            target = legal_actions_recomputed[action_index][1].item()
            action = (source, target)

        return action

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates the Q-value of an action."""
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)

    def __build_target_network(self):
        pass

    def __build_policy_network(self):
        pass
