class RLManager:
    def __init__(self, config, mcts_class, network_manager_class, replay_buffer_class, game_class):
        self.config = config
        
        ###
        self.network_manager = network_manager_class(config)
        self.replay_buffer = replay_buffer_class(config)
        self.mcts_class = mcts_class
        self.game_class = game_class

    def self_play(self, num_episodes):
        """Play games MCTS and collect info """
        for episode in range(num_episodes):
            game = self.game_class(self.config)
            mcts = self.mcts_class(self.network_manager, self.config)

            while not game.terminal():
                root = mcts.run(game.get_current_state())
                action = root.select_action()
                game.apply(action)
                game.store_search_statistics(root)

            self.replay_buffer.save_game(game)

    def train_network(self, num_batches):
        """train from replay buffer data   """
        for _ in range(num_batches):
            batch = self.replay_buffer.sample_batch()
            loss = self.network_manager.train_step(batch)

    def train(self, num_iterations, episodes_per_iteration, batches_per_iteration):
        """ training loop"""
        for iteration in range(num_iterations):
            print(f"=== Iteration {iteration} ===")
            self.self_play(episodes_per_iteration)
            self.train_network(batches_per_iteration)
