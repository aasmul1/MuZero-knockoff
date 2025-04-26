class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.REPLAY_BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        self.num_unroll_steps = config.NUM_UNROLL_STEPS
        self.td_steps = config.TD_STEPS
        
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) + 1 > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = np.random.choice(self.buffer, size=self.bathc_size / 2)