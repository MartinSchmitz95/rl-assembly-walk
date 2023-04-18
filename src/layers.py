import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_dim, dim_latent):
        super(TwoLayerMLP, self).__init__()
        self.f1 = nn.Linear(input_dim, 16)
        self.f2 = nn.Linear(16, dim_latent)

        self.relu = torch.nn.ReLU()  # instead of Heaviside step fn
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        output = self.f1(x)
        output = self.relu(output)
        output = self.f2(output)
        output = self.sigm(output)
        return output


class GatedGCN(nn.Module):
    def __init__(self, num_gnn_layers, dim_latent, device, batch_norm, dag):
        super().__init__()
        self.convs = nn.ModuleList([DoubleGatedGCNLayer(
            dim_latent, device, batch_norm, dag) for _ in range(num_gnn_layers)])

        """self.convs = nn.ModuleList([])
        for n in range(num_gnn_layers):
            dg_gcn_layer = DoubleGatedGCNLayer(
                dim_latent, device, batch_norm, dag)
            if "cuda" in device:
                cuda_device_id = int(device.split(":")[1])
                dg_gcn_layer = dg_gcn_layer.cuda(cuda_device_id)
            self.convs.add_module(f"dg_gcn:{n}", dg_gcn_layer)"""
        # if "cuda" in device:
        #     cuda_device_id = int(device.split(":")[1])
        #     self.convs = self.convs.cuda(cuda_device_id)

    def forward(self, edge_index, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](edge_index, h, e)
        return h, e


class DoubleGatedGCNLayer(MessagePassing):
    def __init__(self, dim_latent, device, batch_norm, dag, dropout=0.1):
        super().__init__(aggr='add')  # "Add" aggregation
        #self.training = training
        dtype = torch.float32
        self.device = device
        self.batch_norm = batch_norm
        self.dag = dag
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
        if not self.dag:
            self.V_b = nn.Linear(dim_latent, dim_latent, dtype=dtype)

        self.conc = nn.Linear(3*dim_latent, dim_latent, dtype=dtype)

    def forward(self, edge_index, h, e_in):

        U = self.U(h)  # A1h
        V_f = self.V_f(h)  # A2h
        if not self.dag:
            V_b = self.V_b(h)  # A3h

        A = self.A(h)  # B1h
        B = self.B(h)  # B2h
        C = self.C(e_in)  # B3e

        h_in = h.clone()
        h_f, e = self.directed_gate(
            edge_index, A, B, C, V_f, e_in, h_in, forward=True)
        if not self.dag:
            h_b, _ = self.directed_gate(
                edge_index, A, B, C, V_b, e_in, h_in, forward=False)
        if self.dag:
            h = h_f + U
        else:
            h = h_f + U + h_b

        if self.batch_norm:
            h = self.bn_h(h)
        h = F.relu(h)
        h = h + h_in
        #h = F.dropout(h, self.dropout, training=self.training)
        h = self.dropout(h)
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


class EdgePredictor(nn.Module):
    def __init__(self, dim_latent, hidden_edge_scores):
        super().__init__()
        self.W1 = nn.Linear(3 * dim_latent, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, 1)

    def forward(self, edge_index, x, e):
        src, dst = edge_index
        data = torch.cat(([x[src], x[dst], e]), dim=1)
        h = self.W1(data)
        h = torch.relu(h)
        h = self.W2(h)
        h = torch.relu(h)
        score = self.W3(h)
        return score


class NodePredictor(nn.Module):
    def __init__(self, dim_latent, hidden_edge_scores):
        super().__init__()
        self.W1 = nn.Linear(dim_latent, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, 1)

    def forward(self, x):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.W2(h)
        h = torch.relu(h)
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
