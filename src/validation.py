from agent import AssemblyWalkAgent, GreedyWalkAgent, RandomWalkAgent
from environment import GraphWalkEnv
from hyperparameters import get_hyperparameters

def validate_agent(agent, seed_nodes, name):
    total_reward = 0
    for graph_path in env.graph_dataset:
        for seed in seed_nodes:
            obs = env.reset(graph_path, seed)
            done = False
            # play one episode
            while not done:
                legal_actions = env.get_legal_actions()
                action = agent.get_action(obs, legal_actions)
                next_obs, reward, terminated = env.step(action)
                # update if the environment is done and the current obs
                done = terminated
                obs = next_obs
            total_reward = env.accumulated_reward
            print(f"Total reward of {name} agent: {total_reward}")

graph_folder = '../data/processed_graphs'
config = get_hyperparameters()
env = GraphWalkEnv(graph_folder)
greedy_agent = GreedyWalkAgent()
rnd_agent = RandomWalkAgent()
rl_agent = AssemblyWalkAgent(config, inference=True)
seed_nodes = [0, 42, 100, 1000]

validate_agent(greedy_agent, seed_nodes, 'Greedy')
validate_agent(rnd_agent, seed_nodes, 'Random')
validate_agent(rl_agent, seed_nodes, 'RL')



