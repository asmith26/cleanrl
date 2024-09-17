import numpy as np

from cleanrl.keras_core_jax.simple_env import SimpleEnvironment


class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.action_space))
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error


if __name__ == "__main__":
    # Create a SimpleEnvironment instance
    env = SimpleEnvironment()

    # Create a QLearningAgent instance
    agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

    # Run the environment
    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    print(agent.q_table)
    # Test the learned policy
    state = env.reset()
    env.render()
    done = False

    while not done:
        action = np.argmax(agent.q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        print(f"Action taken: {action}, Reward received: {reward}")
        env.render()

    print("Game over!")
