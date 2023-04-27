from typing import Dict
import numpy as np
from model import GraphDQN
import random
import torch
from torch import optim
from torch_geometric.utils import k_hop_subgraph, subgraph
from memory import ReplayBuffer, Transition


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

        # ? The policy network is trained to select optimal direction, and the target network predicts the value of it
        self.policy_network = GraphDQN(config).to(config['device'])
        self.target_network = GraphDQN(config).to(config['device'])
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.num_gnn_layers = config['num_gnn_layers']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']
        self.epsilon = config['start_epsilon']
        # reduce the exploration over time
        self.epsilon_decay = config['start_epsilon'] / (config['n_episodes'] / 2.)
        self.final_epsilon = config['final_epsilon']

        self.cumulative_reward = 0.
        self.loss_function = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_network.parameters(), betas=[config["adam_beta1"], config["adam_beta2"]], lr=config["learning_rate"])

        self.__edge_qvalues = None
        self.__node_qvalues = None

        self.__batch_size = 64
        self.__replay_memory = ReplayBuffer(self.__batch_size)

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
        self.policy_network.eval()
        # create k-hop subgraph
        subset, edge_index, _, _ = k_hop_subgraph(obs['agent_location'], self.num_gnn_layers,
                                                  obs['graph'].edge_index, relabel_nodes=True,
                                                  flow='target_to_source')
        _, e = subgraph(subset, obs['graph'].edge_index, edge_attr=obs['graph'].edge_attr, relabel_nodes=False)
        x = obs['graph'].x[subset]

        # here the q-value of the env get be computed
        edge_values, node_values = self.policy_network(edge_index, x, e)
        self.target_network(edge_index, x, e)

        self.__edge_qvalues = edge_values
        self.__node_qvalues = node_values

        # find which node is the current agent position
        stop_action_index = torch.argwhere(subset == obs['agent_location']).item()
        stop_action = node_values[stop_action_index]

        # retrieve edge index ids, to check with q-value outputs are from the legal actions
        _, edge_index_norelabel, _, _ = k_hop_subgraph(obs['agent_location'], self.num_gnn_layers,
                                                       obs['graph'].edge_index, relabel_nodes=False,
                                                       flow='target_to_source')  # directed=False
        comp = edge_index_norelabel[0].numpy()
        edge_action_index = np.argwhere(comp == obs['agent_location'])

        legal_actions_recomputed = edge_index_norelabel.T[edge_action_index.squeeze()].view(-1, 2)
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

        action_values = torch.concat([edge_values, node_values])

        return action, action_values

    def update(self, obs, next_obs, action, action_values, reward, terminated):
        """Updates the Q-value of an action."""
        if terminated or self.__edge_qvalues is None or self.__node_qvalues is None:
            return

        if len(self.__replay_memory) < self.__batch_size:
            return


        next_edge_values, next_node_values = self.target_network(edge_index, x, e)
        next_values = torch.concatenate([next_edge_values, next_node_values])
        qvalues = torch.concatenate([self.__edge_qvalues, self.__node_qvalues])

        loss = self.loss_function(next_values, qvalues)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.__replay_memory.empty()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)
