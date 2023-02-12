import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class GatedGCN(nn.Module):
	def __init__(self, num_gnn_layers, dim_latent, device, batch_norm):
		super().__init__()
		self.convs = nn.ModuleList([
			DoubleGatedGCNLayer(dim_latent, device, batch_norm) for _ in range(num_gnn_layers)
		])

	def forward(self, edge_index, h, e):
		for i in range(len(self.convs)):
			h, e = self.convs[i](edge_index, h, e)
		return h, e


class DoubleGatedGCNLayer(MessagePassing):
	def __init__(self, dim_latent, device, batch_norm, dropout=0.1):
		super().__init__(aggr='add')  # "Add" aggregation
		#self.training = training
		dtype = torch.float32
		self.device = device
		self.batch_norm = batch_norm
		# self.propagate_edges = Gate(dim_latent)  # 'source_to_target'  target_to_source
		# self.sum_sigma = SGate(dim_latent)  # 'source_to_target'  target_to_source
		self.bn_h = nn.BatchNorm1d(dim_latent, track_running_stats=False)
		self.bn_e = nn.BatchNorm1d(dim_latent, track_running_stats=False)
		self.dropout = nn.Dropout(p=dropout)

		self.A = nn.Linear(dim_latent, dim_latent, dtype=dtype)
		self.B = nn.Linear(dim_latent, dim_latent, dtype=dtype)
		self.C = nn.Linear(dim_latent, dim_latent, dtype=dtype)

		self.U = nn.Linear(dim_latent, dim_latent, dtype=dtype)
		self.V_f = nn.Linear(dim_latent, dim_latent, dtype=dtype)
		self.V_b = nn.Linear(dim_latent, dim_latent, dtype=dtype)

		self.conc = nn.Linear(3*dim_latent, dim_latent, dtype=dtype)

	def forward(self, edge_index, h, e_in):

		U = self.U(h)  # A1h
		V_f = self.V_f(h)  # A2h
		V_b = self.V_b(h)  # A3h

		A = self.A(h)  # B1h
		B = self.B(h)  # B2h
		C = self.C(e_in)  # B3e

		h_in = h.clone()
		h_f, e = self.directed_gate(edge_index, A, B, C, V_f, e_in, h_in, forward=True)
		h_b, _ = self.directed_gate(edge_index, A, B, C, V_b, e_in, h_in, forward=False)

		h = h_f + h_b + U  # concat instead and add a ff layer?
		#h = torch.cat(([ h_f , h_b , U ]), dim=1) # concat instead and add a ff layer?
		#h = self.conc(h)

		if self.batch_norm:
			h = self.bn_h(h)
		h = F.relu(h)
		h = h + h_in
		#h = F.dropout(h, self.dropout, training=self.training)
		h =self.dropout(h)
		return h, e

	def directed_gate(self, edge_index, A, B, C, V, e, h_in, forward):

		if forward:
			src, dst = edge_index
			new_edge_index = edge_index
		else:
			dst, src = edge_index
			new_edge_index = torch.vstack((dst, src))

		new_e = A[src] + B[dst] + C  # B1h + B2h + B3e

		if self.batch_norm:
			new_e = self.bn_e(new_e)
		new_e = F.relu(new_e)
		new_e = new_e + e
		sigma_f = torch.sigmoid(new_e)  # put values into interval [0,1]
		h = self.propagate(new_edge_index, V=V, sigma_f=sigma_f)

		return h, new_e

	def message(self, V_j, sigma_f):
		return V_j * sigma_f / (sigma_f + 1e-6)


class ScorePredictor(nn.Module):
	def __init__(self, dim_latent, hidden_edge_scores):
		super().__init__()
		self.W1 = nn.Linear(3 * dim_latent, hidden_edge_scores)
		self.W2 = nn.Linear(hidden_edge_scores, hidden_edge_scores)
		self.W3 = nn.Linear(hidden_edge_scores, 1)

		self.dropout = nn.Dropout(p=0.2)

	def forward(self, edge_index, x, e, f):
		src, dst = edge_index
		data = torch.cat(([x[src], x[dst], e]), dim=1)
		h = self.W1(data)
		h = torch.relu(h)
		h = self.dropout(h)
		h = self.W2(h)
		h = torch.relu(h)
		h = self.dropout(h)
		score = self.W3(h)
		return score

class ScorePredictorF(nn.Module):
	def __init__(self, dim_latent, hidden_edge_scores):
		super().__init__()
		self.W1 = nn.Linear(3 * dim_latent + 2, hidden_edge_scores)
		self.W2 = nn.Linear(hidden_edge_scores, hidden_edge_scores)
		self.W3 = nn.Linear(hidden_edge_scores, 1)
	def forward(self, edge_index, x, e, f):
		src, dst = edge_index
		data = torch.cat(([x[src], x[dst], e, f[src], f[dst]]), dim=1)
		#data = torch.cat(([x[src], x[dst], e, f]), dim=1)

		h = self.W1(data)
		h = torch.relu(h)
		h = self.W2(h)
		h = torch.relu(h)
		score = self.W3(h)
		return score