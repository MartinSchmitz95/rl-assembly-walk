import argparse
import os
import networkx as nx
import pickle
import torch
from torch_geometric.utils.convert import from_networkx


"""
Author: 

"""

def create_dag_ftrs(nx_graph):

    read_length = nx.get_node_attributes(nx_graph, "read_length")
    added_sequence = nx.get_edge_attributes(nx_graph, "overlap_length")
    for e in added_sequence.keys():
        added_sequence[e] = read_length[e[1]] - added_sequence[e]
    nx.set_edge_attributes(nx_graph, added_sequence, 'added_sequence')

    seq_dist = dag_longest_paths(nx_graph, weight="added_sequence", default_weight=1).unsqueeze(dim=1)
    node_dist = dag_longest_paths(nx_graph, weight="none", default_weight=1).unsqueeze(dim=1)
    anc = torch.zeros(nx_graph.number_of_nodes())
    desc = torch.zeros(nx_graph.number_of_nodes())
    # print(g.nodes, len(list(topo_order)))
    topo_order = nx.topological_sort(nx_graph)

    for v in topo_order:
        anc[v] = len(nx.ancestors(nx_graph, v))
        desc[v] = len(nx.descendants(nx_graph, v))
    anc /= 1000  # max_anc
    desc /= 1000  # max_desc
    seq_dist /= 1000000
    node_dist /= 1000

    # edge features
    no_trans_g = nx.transitive_reduction(nx_graph.copy())

    non_trans_edges_dict = {}

    for e in nx_graph.edges:
        non_trans_edges_dict[e] = 0

    for e in no_trans_g.edges:
        non_trans_edges_dict[e] = 1

    non_trans_edges = torch.Tensor(list(non_trans_edges_dict.values())).unsqueeze(dim=1)
    anc_desc = torch.hstack((anc.unsqueeze(dim=1), desc.unsqueeze(dim=1)))

    return anc_desc, node_dist, seq_dist, non_trans_edges

def create_read_length(nx_graph):
    read_length_attr = nx.get_node_attributes(nx_graph, 'read_length')
    read_length = torch.Tensor(list(read_length_attr.values()))
    relative_read_length = ((read_length - read_length.mean()) / read_length.std()).unsqueeze(dim=1)
    read_length = read_length.unsqueeze(dim=1)
    return read_length, relative_read_length

def dag_longest_paths(G, weight="weight", default_weight=1, topo_order=None):
    if not G:
        print("graph empty")
        return []

    if topo_order is None:
        topo_order = nx.topological_sort(G)

    dist = {}  # stores {v : (length, u)}
    # only_dist =  {}
    only_dist = torch.zeros(G.number_of_nodes())
    # print(topo_order)
    for v in topo_order:
        us = [
            (dist[u][0] + data.get(weight, default_weight), u)
            for u, data in G.pred[v].items()
        ]
        # Use the best predecessor if there is one and its distance is
        # non-negative, otherwise terminate.
        maxu = max(us, key=lambda x: x[0]) if us else (0, v)
        dist[v] = maxu if maxu[0] >= 0 else (0, v)
        only_dist[v] = maxu[0]

    # m = torch.max(only_dist)
    # only_dist /= m
    return only_dist

def create_in_out(nx_graph):
    in_out_degrees = torch.zeros(nx_graph.number_of_nodes(), 2)
    for n in nx_graph.nodes:
        in_out_degrees[n][0] = nx_graph.in_degree(n)
        in_out_degrees[n][1] = nx_graph.out_degree(n)

    #in_out_degrees /= (in_out_degrees - in_out_degrees.mean()) / in_out_degrees.std()
    in_out_degrees /= 10  # want small numbers
    return in_out_degrees

def create_edge_features(nx_graph):
    ol_len = nx.get_edge_attributes(nx_graph, 'overlap_length')
    ol_sim = nx.get_edge_attributes(nx_graph, 'overlap_similarity')

    ol_len = torch.Tensor(list(ol_len.values()))
    ol_sim = torch.Tensor(list(ol_sim.values()))

    ol_len = (ol_len - ol_len.mean()) / ol_len.std()

    ol_sim = torch.nan_to_num(ol_sim, nan=1.0)
    edge_features = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
    return edge_features

def create_gt(nx_graph):
    read_start_dict = nx.get_node_attributes(nx_graph, "read_start")
    read_end_dict = nx.get_node_attributes(nx_graph, "read_end")
    read_strand_dict = nx.get_node_attributes(nx_graph, "read_strand")

    correct_edges = []
    for edge in nx_graph.edges():
        src, dst = edge

        on_pos_strand = read_start_dict[dst] < read_end_dict[src] and read_start_dict[dst] > read_start_dict[src] and read_strand_dict[src] == 1 and read_strand_dict[dst] == 1
        on_neg_strand = read_start_dict[src] < read_end_dict[dst] and read_start_dict[src] > read_start_dict[dst] and read_strand_dict[src] == -1 and read_strand_dict[dst] == -1

        if on_pos_strand or on_neg_strand:
            correct_edges.append(edge)

    gt_dict = {}
    for e in nx_graph.edges():
        if e in correct_edges:
            gt_dict[e] = 1.
        else:
            gt_dict[e] = 0.
    print(f"Total fraction of edges: {len(correct_edges)}, of {len(nx_graph.edges)} edges in total")
    return torch.Tensor(list(gt_dict.values()))

def process_graph_dag(out_dir, data_path, filename, id):

    filename = filename[:-4]
    if not os.path.isfile(data_path):
        return
    with open(data_path, 'rb') as pickle_file:
        nx_graph = pickle.load(pickle_file)

    id_dict = {}
    for i, n in enumerate(nx_graph.nodes):
        id_dict[n] = i
    nx_graph = nx.relabel_nodes(nx_graph, id_dict, copy=True)
    print(f"Loaded {filename}")

    ground_truth = create_gt(nx_graph)
    read_length, relative_read_length = create_read_length(nx_graph)
    in_out = create_in_out(nx_graph)
    anc_desc, node_dist, seq_dist, non_trans_edges = create_dag_ftrs(nx_graph)
    edge_ftrs = create_edge_features(nx_graph)
    empty_graph = nx.DiGraph(nx_graph.edges())
    pyg_graph = from_networkx(empty_graph)
    pyg_graph.edge_attr = torch.hstack((non_trans_edges, edge_ftrs))
    pyg_graph.y = ground_truth
    pyg_graph.x = torch.hstack((in_out, relative_read_length,  read_length, seq_dist, node_dist, anc_desc))

    with open(os.path.join(out_dir, f'{filename}.pt'), 'wb') as handle:
        torch.save(pyg_graph, handle)

def process_graph(out_dir, data_path, filename, id):

    filename = filename[:-4]
    if not os.path.isfile(data_path):
        return
    with open(data_path, 'rb') as pickle_file:
        nx_graph = pickle.load(pickle_file)

    id_dict = {}
    for i, n in enumerate(nx_graph.nodes):
        id_dict[n] = i
    nx_graph = nx.relabel_nodes(nx_graph, id_dict, copy=True)
    print(f"Loaded {filename}")

    ground_truth = create_gt(nx_graph)
    #read_length, relative_read_length = create_read_length(nx_graph)
    empty_graph = nx.DiGraph(nx_graph.edges())
    pyg_graph = from_networkx(empty_graph)
    pyg_graph.edge_attr = create_edge_features(nx_graph)
    pyg_graph.y = ground_truth
    pyg_graph.x = create_in_out(nx_graph)

    with open(os.path.join(out_dir, f'{filename}.pt'), 'wb') as handle:
        torch.save(pyg_graph, handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data', type=str, default='../data/', help='Path to folder with data')
    parser.add_argument('--dag', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    in_dir = os.path.join(os.path.abspath(args.data), 'raw_graphs')
    out_dir = os.path.join(os.path.abspath(args.data), 'processed_graphs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for id, filename in enumerate(os.listdir(in_dir)):
        print(f'Process {filename}, graph {id + 1}/{len(os.listdir(in_dir))}')
        data_path = os.path.join(in_dir, filename)
        if args.dag:
            process_graph_dag(out_dir, data_path, filename, id)
        else:
            process_graph(out_dir, data_path, filename, id)

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    print(f"Created data in {out_dir}")
    print(f"Please move the data manually into {train_dir} and {val_dir}")
