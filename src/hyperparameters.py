import torch


def get_hyperparameters():
    return {
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'seed': 0,
        'wandb_mode': 'online',  # switch between 'online' and 'disabled'
        'wandb_project': 'rl-assembly-walk',

        # Model
        'dim_latent': 64,
        'num_gnn_layers': 4,
        'node_features': 8,
        'edge_features': 3,
        'hidden_edge_features': 16,
        'hidden_edge_scores': 64,
        'batch_norm': False,

        # Training
        'gamma': 0.99,
        'n_episodes': 1_000,
        'learning_rate': 2e-5,
        'discount_factor': 0.95,
        'start_epsilon': 1.0,
        'final_epsilon': 0.1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,

    }
