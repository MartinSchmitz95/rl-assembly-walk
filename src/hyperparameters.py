import torch

def get_hyperparameters():
    return {
        'device': 'cuda:5' if torch.cuda.is_available() else 'cpu',
        'seed': 0,
        'wandb_mode': 'online',  # switch between 'online' and 'disabled'
        'wandb_project': 'rl-assembly-walk',

        # Model
        'dim_latent': 256,
        'num_gnn_layers': 16,
        'node_features': 2,
        'edge_features': 2,
        'hidden_edge_features': 16,
        'hidden_edge_scores': 64,
        'batch_norm': True,

        # Training
        'num_epochs': 150,
        'lr': 2e-5,
        'patience': 2,
        'decay': 0.95,

	}
