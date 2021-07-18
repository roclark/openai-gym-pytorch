settings = {
    'batch_size': 32,
    'environment': 'PongNoFrameskip-v4',
    'epsilon_start': 1.0,
    'epsilon_final': 0.01,
    'epsilon_decay': 100000,
    'gamma': 0.99,
    'initial_learning': 10000,
    'learning_rate': 1e-4,
    'memory_capacity': 20000,
    'num_episodes': 10000,
    'target_model': True,
    'target_update_frequency': 1000
}
