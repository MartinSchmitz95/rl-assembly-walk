import torch
import torch.nn as nn
from layers import GatedGCN, EdgePredictor, NodePredictor, TwoLayerMLP


class QValueModel(nn.Module):
	def __init__(self, config, dag=True, cpu=False):
		super().__init__()
		dtype = torch.float32
		self.device = config['device']
		self.encoder = TwoLayerMLP(config['node_features'], config['dim_latent'])#.cuda()
		self.linear1_edge = nn.Linear(config['edge_features'], config['hidden_edge_features'], dtype=dtype)
		self.linear2_edge = nn.Linear(config['hidden_edge_features'], config['dim_latent'], dtype=dtype)
		self.gnn = GatedGCN(config['num_gnn_layers'], config['dim_latent'], config['device'], config['batch_norm'], dag=dag)
		#self.predictor = ScorePredictorF(config['dim_latent'], config['hidden_edge_scores'])
		self.edge_values = EdgePredictor(config['dim_latent'], config['hidden_edge_scores'])
		self.node_values = NodePredictor(config['dim_latent'], config['hidden_edge_scores'])

	def forward(self, edge_index, x, e):

		x = x.to(self.device)
		e = e.to(self.device)
		edge_index = edge_index.to(self.device)

		x = self.encoder(x)

		# encode nodes and edges
		e = self.linear1_edge(e)
		e = torch.relu(e)
		e = self.linear2_edge(e)

		# use gnn
		x, e = self.gnn(edge_index, x, e)
		edge_values = self.edge_values(edge_index, x, e)
		node_values = self.node_values(x)

		return edge_values, node_values
