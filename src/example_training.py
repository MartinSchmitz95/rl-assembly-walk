from agent import AssemblyWalkAgent, GreedyWalkAgent
from environment import GraphWalkEnv
from hyperparameters import get_hyperparameters
from tqdm import tqdm


graph_folder = 'data/processed_graphs'
config = get_hyperparameters()
env = GraphWalkEnv(graph_folder)
agent = AssemblyWalkAgent(config)

for episode in tqdm(range(config['n_episodes'])):
    obs = env.reset()
    done = False

    # play one episode
    while not done:
        legal_actions = env.get_legal_actions()
        action, action_values = agent.get_action(obs, legal_actions)
        next_obs, reward, terminated = env.step(action)

        # update the agent
        agent.update(obs, next_obs, action, action_values, reward, terminated)

        # update if the environment is done and the current obs
        done = terminated
        obs = next_obs

    agent.decay_epsilon()
