from environment import GraphWalkEnv
from hyperparameters import get_hyperparameters
from agent import AssemblyWalkAgent, RandomWalkAgent

from tqdm import tqdm
import networkx as nx
import torch
import os
import pickle
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def dag_longest_paths(G, weight="weight", default_weight=1, topo_order=None):
    if not G:
        print("graph empty")
        return []

    if topo_order is None:
        topo_order = nx.topological_sort(G)

    dist = {}  # stores {v : (length, u)}
    # only_dist =  {}
    only_dist = torch.zeros(G.number_of_nodes())
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

def set_best_startnode(graph, env):
    # sets best startnode and returns max possible reward

    env.active_node =  torch.argmax(total_dist).item()
    max_reward = torch.max(total_dist).item()
    return max_reward

def walk_to_sequence(walks, graph):
    reads = nx.get_node_attributes(graph, 'read_sequence')
    overlap = nx.get_edge_attributes(graph, 'overlap_length')
    contigs = []

    for i, walk in enumerate(walks):
        sequence = ''
        current = walk[0]
        last = walk[-1]
        for target in walk[1:-1]:
            overlap_length = overlap[(current, target)]
            prefix = len(reads[current]) - overlap_length
            sequence += reads[current][:prefix]
            current = target
        sequence += reads[last]

        record = SeqRecord(
            Seq(sequence),
            id=f'contig_{i+1}',
            name="",
            description=f'length={len(sequence)}',
        )
        """sequence = SeqIO.SeqRecord(sequence)
        sequence.id = f'contig_{i+1}'
        sequence.description = f'length={len(sequence)}'"""
        contigs.append(record)
    print(f'Number of Contigs: {len(contigs)}')
    return contigs

def get_paths(env, agent):
    """Iteratively search for contigs in a graph until the threshold is met."""

    """Iteratively search for contigs in a graph until the threshold is met."""
    # graph.remove_edges_from(nx.selfloop_edges(graph)) # self loops handled later
    all_contigs = []  ##
    visited = set()
    idx_contig = -1
    total_max_reward = 0

    node_dist = dag_longest_paths(env.active_graph_nx, default_weight=1)
    node_dist_reversed = dag_longest_paths(env.active_graph_nx.reverse(copy=True), default_weight=1)
    total_dist = node_dist + node_dist_reversed
    total_dist_dict = {}
    for i, n in enumerate(env.active_graph_nx.nodes()):
        total_dist_dict[n] = total_dist[i]
    nx.set_node_attributes(env.active_graph_nx, total_dist.tolist(), 'total_dist')
    full_graph = env.active_graph_nx

    while True:
        max_dist = 0
        env.active_graph_nx = full_graph.subgraph(full_graph.nodes() - visited)
        for node in env.active_graph_nx.nodes():
            if total_dist_dict[node] > max_dist:
                max_dist = total_dist_dict[node]
                start_node = node
                source_degree = env.active_graph_nx.in_degree(start_node) + env.active_graph_nx.out_degree(start_node)

            elif total_dist_dict[node] == max_dist:
                node_degree = env.active_graph_nx.in_degree(node) + env.active_graph_nx.out_degree(node)
                if node_degree > source_degree:
                    start_node = node
                    source_degree = node_degree

        idx_contig += 1

        if len(env.active_graph_nx.edges()) < 10:  # if not sufficient edges in subgraph anymore: stop
            break

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {env.active_graph_nx.number_of_nodes()}, nb_original_nodes: {env.active_graph_nx.number_of_nodes()}')

        # play one episode
        walk_f, visited_f, reward_f = find_walk(agent, env, start_node, backwards=False)
        walk_b, visited_b, reward_b = find_walk(agent, env, start_node, backwards=True)

        # concatenate two walks
        walk = walk_b + walk_f
        visited = visited | visited_f | visited_b

        trans = set()
        for ss, dd in zip(walk[:-1], walk[1:]):
            t1 = set(env.active_graph_nx.neighbors(ss)) & set(env.active_graph_nx.predecessors(dd))
            t2 = {t ^ 1 for t in t1}
            trans = trans | t1 | t2
        visited = visited | trans

        # If longest contig is longer than len_threshold, add it and continue, else break
        if len(walk) < 3:
            continue
        all_contigs.append(walk)
        break  # debug on single contig
    return all_contigs

def find_walk(agent, env, start_node, backwards = False):
    walk = []
    new_visited = set()
    env.reset()
    env.active_node = start_node
    obs = {}
    obs["agent_location"] = env.active_node
    terminated = False
    first_iter = True
    if backwards:
        env.active_graph_nx = env.active_graph_nx.reverse(copy=False)

    while not terminated:
        # reset env and set start node
        if not(backwards and first_iter):
            walk.append(obs["agent_location"])
            first_iter=False
        legal_actions = env.get_legal_actions()
        action = agent.get_action(obs, legal_actions)
        if action is not None:
            new_visited.add(action[1])
            new_visited.add(action[1]^1)

        next_obs, _, terminated = env.step_inference(action)
        # update if the environment is done and the current obs
        obs = next_obs

    if backwards:
        env.active_graph_nx = env.active_graph_nx.reverse(copy=False)
        walk = list(reversed(walk))
    return walk, new_visited, len(walk)

def assemble_all_graphs(env, agent, out_folder):
    for i, graph_path in enumerate(env.graph_dataset):
        raw_graph_path = os.path.join(raw_graphs_folder, env.graph_names[i] + '.pkl')
        env.reset(graph_path=graph_path)
        paths = get_paths(env, agent)

        with open(raw_graph_path, 'rb') as f:
            raw_graph = pickle.load(f)
        contigs = walk_to_sequence(paths, raw_graph)

        assembly_path = os.path.join(out_folder, f'{env.graph_names[i]}.fasta')
        SeqIO.write(contigs, assembly_path, 'fasta')

def evaluate_with_minigraph(ref_path, out, dataset):

    procs = []

    for idx in dataset.indices:
        end_char = dataset.graphs[idx].find('_')
        chr = dataset.graphs[idx][:end_char]

        ref = os.path.join(ref_path, 'chromosomes', f'{chr}.fasta')
        asm = os.path.join(out, f'{chr}_assembly.fasta')
        paf = os.path.join(out, f'{chr}_asm.paf')
        p = run_minigraph(ref, asm, paf)
        procs.append(p)

    for p in procs:
        p.wait()

    procs = []
    for idx in dataset.indices:
        end_char = dataset.graphs[idx].find('_')
        chr = dataset.graphs[idx][:end_char]
        idx = os.path.join(ref_path, 'indexed', f'{chr}.fasta.fai')
        paf = os.path.join(out, f'{chr}_asm.paf')
        report = os.path.join(out, f'{chr}_minigraph.txt')
        p = parse_pafs(idx, report, paf)
        procs.append(p)

    for p in procs:
        p.wait()

    parse_minigraph_for_chrs(out, dataset)

def run_minigraph(ref, asm, paf):
    minigraph = f'/home/schmitzmf/minigraph/minigraph'
    # paftools = f'/home/vrcekl/minimap2-2.24_x64-linux/paftools.js'
    # paf = os.path.join(save_path, f'asm.paf')
    # cmdaa = f'{minigraph} -xasm -g10k -r10k --show-unmap=yes {ref} {asm} > {paf} && k8 {paftools} asmstat {idx} {paf} > {report}'
    cmd = f'{minigraph} -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_pafs(idx, report, paf):
    # paftools = f'/home/schmitzmf/minimap2/paftools.js'
    # paf = os.path.join(save_path, f'asm.paf')
    # cmd = f'k8 {paftools} asmstat {idx} {paf}'.split()
    cmd = f'paftools.js asmstat {idx} {paf}'.split()

    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_minigraph_for_chrs(save_path, data):
    ng50, nga50 = {}, {}

    for idx in data.indices:
        end_char = data.graphs[idx].find('_')
        chr = data.graphs[idx][:end_char]
        stat_path = f'{save_path}/{chr}_minigraph.txt'
        with open(stat_path) as f:
            for line in f.readlines():
                if line.startswith('NG50'):
                    try:
                        ng50_l = int(re.findall(r'NG50\s*(\d+)', line)[0])
                    except IndexError:
                        ng50_l = 0
                    ng50[f'{chr}'] = ng50_l
                if line.startswith('NGA50'):
                    try:
                        nga50_l = int(re.findall(r'NGA50\s*(\d+)', line)[0])
                    except IndexError:
                        nga50_l = 0
                    nga50[f'{chr}'] = nga50_l

    print('NG50')
    print(*ng50.values(), sep='\n')
    print()

    print('NGA50')
    print(*nga50.values(), sep='\n')
    print()

processed_graphs_folder = '../../scratch/from_my_ionode/rl_processed_graphs'
raw_graphs_folder = '../../scratch/from_my_ionode/raw'
assembly_folder = '../../scratch/from_my_ionode/assembly'
ref_path = '../../scratch/from_my_ionode/assembly'

"""raw_graphs_folder = '../data/raw_graphs'
processed_graphs_folder = '../data/processed_graphs'
assembly_folder = '../data/assemblies'"""

if not os.path.exists(assembly_folder):
    os.makedirs(assembly_folder)
if not os.path.exists(processed_graphs_folder):
    os.makedirs(processed_graphs_folder)

config = get_hyperparameters()
env = GraphWalkEnv(processed_graphs_folder)
rnd_agent = RandomWalkAgent()
rl_agent = AssemblyWalkAgent(config, inference=True)

rnd_ass_folder = os.path.join(assembly_folder, 'random')
if not os.path.exists(rnd_ass_folder):
    os.makedirs(rnd_ass_folder)
rl_ass_folder = os.path.join(assembly_folder, 'rl')
if not os.path.exists(rl_ass_folder):
    os.makedirs(rl_ass_folder)

assemble_all_graphs(env, rnd_agent, rnd_ass_folder)
#assemble_all_graphs(env, rl_agent, rl_ass_folder)

evaluate_with_minigraph(ref_path, rnd_ass_folder, dataset)
