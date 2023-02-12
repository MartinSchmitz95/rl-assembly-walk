import torch
import torch.nn as nn
from layers import GatedGCN, ScorePredictor, ScorePredictorF
from encoders import SequenceEncoderBasic, SequenceEncoderAttention, SequenceEncoderMultiAttention, EncoderCNN, SequenceEncoderNoPadding, TwoLayerMLP


class QValueModel(nn.Module):
	def __init__(self, config, cpu=False):
		super().__init__()
		dtype = torch.float32
		if cpu:
			self.device = 'cpu'
		else:
			self.device = config['device']
		self.encoder = TwoLayerMLP(config['node_features'], config['dim_latent'])
		self.linear1_edge = nn.Linear(config['edge_features'], config['hidden_edge_features'], dtype=dtype)
		self.linear2_edge = nn.Linear(config['hidden_edge_features'], config['dim_latent'], dtype=dtype)
		self.gnn = GatedGCN(config['num_gnn_layers'], config['dim_latent'], config['device'], config['batch_norm'])
		#self.predictor = ScorePredictorF(config['dim_latent'], config['hidden_edge_scores'])
		self.predictor = ScorePredictor(config['dim_latent'], config['hidden_edge_scores'])

	def forward(self, edge_index, x, e):
		# load data to gpu
		x = x.to(self.device)
		#if self.node_feature_mode == 'direct_f':
		#	f = torch.clone(x)[:, 2:]
		#	x = x[:, :2]
		#	#fe = torch.clone(e)[:, 2:]
		#	#e = e[:, :2]
		#	#fe=None
		#else:
		#	f = None
		#	fe = None
		x = self.encoder(x)
		e = e.to(self.device)

		# encode nodes and edges
		e = self.linear1_edge(e)
		e = torch.relu(e)
		e = self.linear2_edge(e)

		# use gnn
		x, e = self.gnn(edge_index, x, e)
		scores = self.predictor(edge_index, x, e)
		return scores
