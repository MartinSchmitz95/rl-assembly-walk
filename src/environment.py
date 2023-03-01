"""
Load environment and perform steps
"""
import os
import random
from torch_geometric.utils.convert import to_networkx
import torch
import networkx as nx
class GraphWalkEnv():

    def __init__(self, graph_folder, dag=True):
        self.graph_dataset = self.get_graph_data(graph_folder)
        self.dag = dag
        self.active_graph_pyg = None
        self.active_graph_nx = None
        self.active_node = None
        self.legal_actions = None
        self.malicious_edges = None
        self.accumulated_reward = 0
        self.reset()
        self.visited_nodes = []

    def get_graph_data(self, graph_folder):
        graph_list = []
        for f in os.listdir(graph_folder):
            graph_list.append(os.path.join(graph_folder, f))
        return graph_list

    def _get_obs(self):
        return {"agent_location": self.active_node, "graph": self.active_graph_pyg}

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        """
        load a random graph from the training set.
        choose a random node as starting position.
        mark active node
        """
        graph_path = random.choice(self.graph_dataset)
        self.active_graph_pyg = torch.load(graph_path)
        self.active_graph_nx = to_networkx(self.active_graph_pyg)
        self.accumulated_reward = 0
        self.visited_nodes = []

        self.malicious_edges = {}
        edge_index = self.active_graph_pyg.edge_index.T
        for i in range(self.active_graph_pyg.num_edges):
            if self.active_graph_pyg.y[i] == 1:
                mal = False
            else:
                mal = True
            e = edge_index[i].tolist()
            key = (e[0], e[1])
            self.malicious_edges[key] = mal


        self.active_node = random.choice(list(self.active_graph_nx.nodes))
        observation = self._get_obs()
        return observation
    def get_legal_actions(self):
        """
        :return: list of legal actions
        """
        return list(self.active_graph_nx.edges(self.active_node))

    def step(self, action):
        """
        next_state: This is the observation that the agent will receive after taking the action.
        reward: This is the reward that the agent will receive after taking the action.
        terminated: This is a boolean variable that indicates whether or not the environment has terminated.
        """
        if action == None:  # agent decides to stop
            terminated = True
            reward = 0
        else:
            self.active_node = action[1]
            if not self.dag:
                if self.active_node in self.visited_nodes:  # terminate if node already visited
                    terminated = True
                self.visited_nodes.append(self.active_node)
            #print(action)
            if self.malicious_edges[action]:  # terminate if malicious edge is crossed
                terminated = True
                reward = - self.accumulated_reward
            else:
                terminated = False  # continue if move is normal
                reward = 1
                self.accumulated_reward += 1

            if not (self.get_legal_actions()):  # terminate if no legal actions available
                terminated = True

        observation = self._get_obs()
        return observation, reward, terminated
