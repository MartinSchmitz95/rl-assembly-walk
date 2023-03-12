from agent import AssemblyWalkAgent
from environment import GraphWalkEnv
from hyperparameters import get_hyperparameters
from tqdm import tqdm


graph_folder = '../data/processed_graphs'
config = get_hyperparameters()
env = GraphWalkEnv(graph_folder)
agent = AssemblyWalkAgent(config)

for episode in tqdm(range(config['n_episodes'])):
    obs = env.reset()
    done = False

    # play one episode
    while not done:
        legal_actions = env.get_legal_actions()
        action = agent.get_action(obs, legal_actions)
        next_obs, reward, terminated = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated
        obs = next_obs

    agent.decay_epsilon()

