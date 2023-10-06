# Based on: https://github.com/vwxyzjn/cleanrl/blob/bbec22d1b60f54903ffaf4993df443f9001d0951/cleanrl/sac_continuous_action_jax.py
import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple
os.environ["KERAS_BACKEND"] = "jax"

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


import jax
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm
import gymnasium as gym
import keras_core as keras
import jax.numpy as jnp


@dataclass
class Args:
    seed: int = 1  # seed of the experiment
    capture_video: bool = True  # capture videos of the agent performances (`videos` folder)
    env_id: str = "HalfCheetah-v4"  # the id of the environment
    save_networks_interval = 50_000  # run this many steps then save networks
    # eval_freq: int = -1  # evaluate the agent every `eval_freq` steps (if negative, no evaluation)
    # n_eval_episodes: int = 10  # number of episodes to use for evaluation
    # n_eval_envs: int = 5  # number of environments for evaluation
    # --- Algorithm specific arguments ---
    total_timesteps: int = 1_000_000  # total timesteps of the experiments
    buffer_size: int = 1_000_000  # the replay memory buffer size
    gamma: float = 0.99  # the discount factor gamma
    tau: float = 0.005  # target smoothing coefficient
    batch_size: int = 256  # the batch size of sample from the replay memory
    learning_starts: int = 5_000  # timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the Q network network optimizer
    n_critics: int = 2  # the number of critic networks
    policy_frequency: int = 1  # the frequency of training policy (delayed)
    target_network_frequency: int = 1  # the frequency of updates for the target networks
    alpha: float = 0.2  # entropy regularization coefficient.
    autotune: bool = True  # automatic tuning of the entropy coefficient


class Utils:
    root_save_dir: str

    @staticmethod
    def get_datetime() -> str:
        current_datetime = datetime.datetime.now()
        return current_datetime.strftime("%Y%m%d-%H%M%S")

    @classmethod
    def update_model_state(cls, model_class):
        for variable, value in zip(model_class.model.trainable_variables, model_class.t):
            variable.assign(value)
        for variable, value in zip(model_class.model.non_trainable_variables, model_class.nt):
            variable.assign(value)
        return model_class

    @classmethod
    def update_models_state_and_save(cls, global_step: int, actor, critic, ent_coef):
        actor = cls.update_model_state(actor)
        actor.save(f"{cls.root_save_dir}/models/a_{global_step}.keras")
        critic = cls.update_model_state(critic)
        critic.save(f"{cls.root_save_dir}/models/c_{global_step}.keras")
        ent_coef = cls.update_model_state(ent_coef)
        ent_coef.save(f"{cls.root_save_dir}/models/ec_{global_step}.keras")


class Env:
    def __init__(self):
        self.envs = DummyVecEnv([self.make_env(0)])
        self.envs.observation_space.dtype = np.float32
        assert isinstance(self.envs.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.action_dim = int(np.prod(self.envs.action_space.shape))
        self.obs_dim = int(np.prod(self.envs.observation_space.shape))

    def make_env(self, idx: int, run_name: str) -> Callable:
        def thunk():
            env = gym.make(Args.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if Args.capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{Utils.root_save_dir}/videos/")
            env.seed(Args.seed)
            env.action_space.seed(Args.seed)
            env.observation_space.seed(Args.seed)
            return env
        return thunk


class Critic:
    def __init__(self):
        self.model = self.get_critic()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.q_lr)
        self.optimizer.build(self.model.trainable_variables)

    def get_critic(self) -> keras.Model:
        n_units = 256

        observation_input = keras.Input(shape=(None, G.obs_dim), name="observation")
        action_input = keras.Input(shape=(None, G.action_dim), name="action")
        x = keras.layers.concatenate([observation_input, action_input])
        x = keras.layers.Dense(n_units)(x)
        x = keras.layers.ReLU(x)
        x = keras.layers.Dense(n_units)(x)
        x = keras.layers.ReLU(x)
        q_value = keras.layers.Dense(1)(x)

        model = keras.Model(
            inputs=[observation_input, action_input],
            outputs=q_value,
        )
        return model


# class VectorCritic(nn.Module):
#     n_units: int = 256
#     n_critics: int = 2
#
#     @nn.compact
#     def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
#         # Idea taken from https://github.com/perrin-isir/xpag
#         # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
#         vmap_critic = nn.vmap(
#             Critic,
#             variable_axes={"params": 0},  # parameters not shared between the critics
#             split_rngs={"params": True},  # different initializations
#             in_axes=None,
#             out_axes=0,
#             axis_size=self.n_critics,
#         )
#         q_values = vmap_critic(
#             n_units=self.n_units,
#         )(obs, action)
#         return q_values


class Actor:
    def __init__(self):
        self.model = self.get_actor()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.policy_lr)
        self.optimizer.build(self.model.trainable_variables)

    def get_actor(self) -> keras.Model:
        n_units = 256
        log_std_min: float = -20
        log_std_max: float = 2

        observation_input = keras.Input(shape=(G.obs_dim,), name="observation")
        x = keras.layers.Dense(n_units)(observation_input)
        x = keras.layers.ReLU(x)
        x = keras.layers.Dense(n_units)(x)
        x = keras.layers.ReLU(x)
        mean = keras.layers.Dense(G.action_dim)(x)
        log_std = keras.layers.Dense(G.action_dim)(x)
        log_std = jnp.clip(log_std, log_std_min, log_std_max)

        model = keras.Model(
            inputs=observation_input,
            outputs=[mean, log_std],
        )
        return model

    @staticmethod
    @jax.jit
    def sample_action(t, nt, observations: jnp.ndarray, key: jax.random.KeyArray) -> Tuple[jnp.array, list, jnp.array]:
        key, subkey = jax.random.split(key, 2)
        (mean, log_std), nt = G.s_a_m.stateless_call(t, nt, observations, training=False)
        action_std = jnp.exp(log_std)
        gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
        action = jnp.tanh(gaussian_action)
        return action, nt, key

    @staticmethod
    def scale_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = action_space.low, action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    @staticmethod
    def unscale_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = action_space.low, action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class EntropyCoef:
    def __init__(self):
        self.model = self.get_entropy_coefficient()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.q_lr)
        self.optimizer.build(self.trainable_variables)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.target_entropy = -np.prod(envs.action_space.shape).astype(np.float32)

    def get_entropy_coefficient(self) -> keras.Model:
        ent_coef_init = 1.0
        log_ent_coef = jnp.full((), jnp.log(ent_coef_init))
        ent_coef = keras.Model(inputs=[],
                               outputs=jnp.exp(log_ent_coef))
        return ent_coef

    @staticmethod
    def train_step(self, entropy: float):
        t, nt, ov, mv = self.state
        def temperature_loss():
            ent_coef_value, nt = self.stateless_call(t, nt, training=True)
            ent_coef_loss = ent_coef_value * (entropy - G.target_entropy).mean()
            return ent_coef_loss, nt

        grad_fn = jax.value_and_grad(temperature_loss, has_aux=True)
        (ent_coef_loss, nt), grads = grad_fn()
        t, ov = self.optimizer.stateless_apply(ov, grads, t)

        state = t, nt, ov, mv
        return ent_coef_loss, state


def main():
    Utils.root_save_dir = f"runs/{Args.env_id}__{Args.seed}__{Utils.get_datetime()}"
    writer = SummaryWriter(Utils.root_save_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in Args.__dict__.items()])),
    )

    # # TRY NOT TO MODIFY: seeding
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    key = jax.random.PRNGKey(Args.seed)
    # Env
    e = Env()
    # Networks
    actor = Actor()
    critic = Actor()
    # automatic entropy tuning
    if Args.autotune:
        ent_coef = EntropyCoef(0)
    else:
        ent_coef_value = jnp.array(Args.alpha)

    # TRY NOT TO MODIFY: start the game
    rb = ReplayBuffer(
        Args.buffer_size,
        e.envs.observation_space,
        e.envs.action_space,
        device="cpu",   # force cpu device to easy torch -> numpy conversion
        handle_timeout_termination=True,
    )
    obs = e.envs.reset()
    start_time = time.time()
    for global_step in tqdm(range(1, Args.total_timesteps+1)):
        # ALGO LOGIC: put action logic here
        if global_step < Args.learning_starts:
            actions = np.array([e.envs.action_space.sample() for _ in range(e.envs.num_envs)])
        else:
            actions, key = actor.sample_action(obs, key)
            actions = np.array(actions)
            # Clip due to numerical instability
            actions = np.clip(actions, -1, 1)
            # Rescale to proper domain when using squashing
            actions = actor.unscale_action(e.envs.action_space, actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = e.envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step + 1}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break


        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Store the scaled action
        scaled_actions = actor.scale_action(e.envs.action_space, actions)
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > Args.learning_starts:
            data = rb.sample(Args.batch_size)












            if Args.autotune:
                ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

            qf_state, (qf_loss_value, qf_values), key = update_critic(
                actor_state,
                qf_state,
                ent_coef_value,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
                key,
            )

            if global_step % Args.policy_frequency == 0:  # TD 3 Delayed update support
                (actor_state, qf_state, actor_loss_value, key, entropy) = update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_value,
                    data.observations.numpy(),
                    key,
                )

                if Args.autotune:
                    ent_coef_state, ent_coef_loss = update_temperature(ent_coef_state, entropy)

            # update the target networks
            if global_step % Args.target_network_frequency == 0:
                qf_state = soft_update(Args.tau, qf_state)











            if global_step % 100 == 0:
                writer.add_scalar("losses/qf_values", qf_values.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss_value.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                writer.add_scalar("losses/alpha", ent_coef_value.item(), global_step)
                if tqdm is None:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if Args.autotune:
                    writer.add_scalar("losses/alpha_loss", ent_coef_loss.item(), global_step)
            if global_step % Args.save_networks_interval == 0:
                Utils.update_models_state_and_save(global_step, actor, critic, ent_coef)

    e.envs.close()
    writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
