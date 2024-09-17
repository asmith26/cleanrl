import numpy as np


class SimpleEnvironment:
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        done = False
        if action == 1:
            reward = 1
            self.state = min(self.state_space - 1, self.state + 1)
            if self.state == self.state_space - 1:
                done = True
        else:
            reward = -1
            self.state = max(0, self.state - 1)

        return self.state, reward, done, {}

    def render(self):
        print(f"Current state: {self.state}")


if __name__ == "__main__":
    # Example usage
    env = SimpleEnvironment()
    state = env.reset()
    env.render()
    done = False

    while not done:
        action = np.random.choice(range(env.action_space))
        next_state, reward, done, _ = env.step(action)
        env.render()

    print("Game over!")
