# Based on: https://github.com/vwxyzjn/cleanrl/blob/bbec22d1b60f54903ffaf4993df443f9001d0951/cleanrl/sac_continuous_action_jax.py
import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import optax

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


class BaseModel:
    model: keras.Model
    optimizer: keras.Optimizer

    @jax.jit
    def jitted_stateless_call(self, *args, **kwargs):
        return self.model.stateless_call(*args, **kwargs)

    def __init__(self):
        self.optimizer.build(self.model.trainable_variables)
        self.t = self.model.trainable_variables
        self.nt = self.model.non_trainable_variables
        self.ov = self.optimizer.variables
        self.tt = None  # target
        self.tnt = None  # target

    def get_state(self, include_ov: bool = True, include_target=False):
        state = [self.t, self.nt]
        if include_ov:
            state.append(self.ov)
        if include_target:
            state.extend([self.tt, self.tnt])
        return state


class Utils:
    @staticmethod
    def get_datetime() -> str:
        current_datetime = datetime.datetime.now()
        return current_datetime.strftime("%Y%m%d-%H%M%S")

    @classmethod
    def _update_model_state(cls, model_instance: BaseModel):
        for variable, value in zip(model_instance.model.trainable_variables, model_instance.t):
            variable.assign(value)
        for variable, value in zip(model_instance.model.non_trainable_variables, model_instance.nt):
            variable.assign(value)
        return model_instance

    @classmethod
    def update_models_state_and_save(cls, global_step: int):
        actor = cls._update_model_state(G.a)
        actor.save(f"{G.root_save_dir}/models/a_{global_step}.keras")
        critic = cls._update_model_state(critic)
        critic.save(f"{G.root_save_dir}/models/c_{global_step}.keras")
        ent_coef = cls._update_model_state(ent_coef)
        ent_coef.save(f"{G.root_save_dir}/models/ec_{global_step}.keras")

    def _make_env(self, idx: int) -> Callable:
        def thunk():
            env = gym.make(Args.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if Args.capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{G.root_save_dir}/videos/")
            env.seed(Args.seed)
            env.action_space.seed(Args.seed)
            env.observation_space.seed(Args.seed)
            return env

        return thunk

    def setup_envs(cls):
        G.envs = DummyVecEnv([cls._make_env(0)])
        G.envs.observation_space.dtype = np.float32
        # assert isinstance(self.envs.action_space, gym.spaces.Box), "only continuous action space is supported"
        G.action_dim = int(np.prod(G.envs.action_space.shape))
        G.obs_dim = int(np.prod(G.envs.observation_space.shape))
        G.target_entropy = -np.prod(G.envs.action_space.shape).astype(np.float32)


class Critic(BaseModel):
    def __init__(self):
        self.model = self.get_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.q_lr)
        super(Critic, self).__init__()
        self.tt = self.model.trainable_variables
        self.tnt =self.model.non_trainable_variables


    def get_model(self) -> keras.Model:
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

    @staticmethod
    @jax.jit
    def update(a_state, qf_state, ent_coef_value: jnp.ndarray,
               observations: np.ndarray, actions: np.ndarray, next_observations: np.ndarray,
               rewards: np.ndarray, dones: np.ndarray, key: jax.random.KeyArray):
        key, subkey = jax.random.split(key, 2)
        a_t, a_nt, a_ov = a_state
        qf_t, qf_nt, qf_ov, qft_t, qft_nt = qf_state

        (mean, log_std), a_nt = G.a.jitted_stateless_call(a_t, a_nt, next_observations)
        next_state_actions, next_log_prob = Actor.sample_action_and_log_prob(mean, log_std, subkey)

        qf_next_values, qft_nt = G.qf.jitted_stateless_call(qft_t, qft_nt, next_observations, next_state_actions)
        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * Args.gamma * next_q_values

        def mse_loss(_qf_t, _qf_nt):
            # shape is (n_critics, batch_size, 1)
            current_q_values, _qf_nt = G.qf.jitted_stateless_call(_qf_t, _qf_nt, observations, actions)
            # mean over the batch and then sum for each critic
            critic_loss = 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()
            return critic_loss, (_qf_nt, current_q_values.mean())

        (qf_loss_value, (qf_nt, qf_values)), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_t, qf_nt)
        qf_t, qf_ov = G.qf.optimizer.stateless_apply(qf_ov, grads, qf_t)

        a_state = a_t, a_nt, a_ov
        qf_state = qf_t, qf_nt, qf_ov, qft_t, qft_nt
        return ((a_state, qf_state), (qf_loss_value, qf_values), key)

    @staticmethod
    @jax.jit
    def soft_update(qf_t, qf_nt, qft_t, qft_nt):
        qft_t = optax.incremental_update(qf_t, qft_t, Args.tau)
        qft_nt = optax.incremental_update(qf_nt, qft_nt, Args.tau)
        return qft_t, qft_nt


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


class Actor(BaseModel):
    def __init__(self):
        self.model = self.get_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.policy_lr)
        super(Actor, self).__init__()

    def get_model(self) -> keras.Model:
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
    def sample_action(state, observations: jnp.ndarray, key: jax.random.KeyArray) -> Tuple[jnp.array, list, jnp.array]:
        t, nt, _ = state
        key, subkey = jax.random.split(key, 2)
        (mean, log_std), nt = G.a.model.stateless_call(t, nt, observations, training=False)
        action_std = jnp.exp(log_std)
        gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
        action = jnp.tanh(gaussian_action)
        return action, nt, key

    @staticmethod
    @jax.jit
    def sample_action_and_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, subkey: jax.random.KeyArray):
        action_std = jnp.exp(log_std)
        gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
        log_prob = -0.5 * ((gaussian_action - mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
        log_prob = log_prob.sum(axis=1)
        action = jnp.tanh(gaussian_action)
        log_prob -= jnp.sum(jnp.log((1 - action ** 2) + 1e-6), 1)
        return action, log_prob

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

    @staticmethod
    @jax.jit
    def update(a_state, qf_state, ent_coef_value: jnp.ndarray,
               observations: np.ndarray, key: jax.random.KeyArray):
        key, subkey = jax.random.split(key, 2)
        a_t, a_nt, a_ov = a_state
        qf_t, qf_nt, qf_ov = qf_state

        def actor_loss(_a_t, _a_nt, _qf_t, _qf_nt):
            (mean, log_std), _a_nt = G.a.jitted_stateless_call(_a_t, _a_nt, observations)
            actions, log_prob = Actor.sample_action_and_log_prob(mean, log_std, subkey)
            qf_pi, _qf_nt = G.qf.jitted_stateless_call(_qf_t, _qf_nt, observations, actions)
            # Take min among all critics
            min_qf_pi = jnp.min(qf_pi, axis=0)
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, (_a_nt, _qf_nt, -log_prob.mean())

        (actor_loss_value, (a_nt, qf_nt, entropy)), grads = jax.value_and_grad(actor_loss, has_aux=True)(a_t, a_nt)
        a_t, a_ov = G.a.optimizer.stateless_apply(a_ov, grads, a_t)

        a_state = a_t, a_nt, a_ov
        qf_state = qf_t, qf_nt, qf_ov
        return a_state, qf_state, actor_loss_value, key, entropy


class EntropyCoef(BaseModel):
    def __init__(self):
        self.model = self.get_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=Args.q_lr)
        self.loss_fn = keras.losses.MeanSquaredError()
        super(EntropyCoef, self).__init__()

    def get_model(self) -> keras.Model:
        ent_coef_init = 1.0
        log_ent_coef = jnp.full((), jnp.log(ent_coef_init))
        ent_coef = keras.Model(inputs=[],
                               outputs=jnp.exp(log_ent_coef))
        return ent_coef

    @staticmethod
    def update(state, entropy: float):
        t, nt, ov = state
        
        def temperature_loss(_t, _nt):
            ent_coef_value, _nt = G.ec.model.stateless_call(_t, _nt, training=True)
            ent_coef_loss = ent_coef_value * (entropy - G.target_entropy).mean()
            return ent_coef_loss, _nt

        (ent_coef_loss, nt), grads = jax.value_and_grad(temperature_loss, has_aux=True)()
        t, ov = G.ec.optimizer.stateless_apply(ov, grads, t)

        state = t, nt, ov
        return ent_coef_loss, state


class G:  # globals
    a: Actor
    qf: Critic
    ec: EntropyCoef
    target_entropy: np.ndarray
    root_save_dir: str
    action_dim: int
    obs_dim: int
    envs: DummyVecEnv


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
    Utils.setup_envs()
    # Networks
    G.a = Actor()
    G.qf = Critic()
    # automatic entropy tuning
    if Args.autotune:
        G.ec = EntropyCoef()
    else:
        ent_coef_value = jnp.array(Args.alpha)

    # TRY NOT TO MODIFY: start the game
    rb = ReplayBuffer(
        Args.buffer_size,
        G.envs.observation_space,
        G.envs.action_space,
        device="cpu",   # force cpu device to easy torch -> numpy conversion
        handle_timeout_termination=True,
    )
    obs = G.envs.reset()
    start_time = time.time()
    for global_step in tqdm(range(1, Args.total_timesteps+1)):
        # ALGO LOGIC: put action logic here
        if global_step < Args.learning_starts:
            actions = np.array([G.envs.action_space.sample() for _ in range(G.envs.num_envs)])
        else:
            actions, key = G.a.sample_action(obs, key)
            actions = np.array(actions)
            # Clip due to numerical instability
            actions = np.clip(actions, -1, 1)
            # Rescale to proper domain when using squashing
            actions = G.a.unscale_action(G.envs.action_space, actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = G.envs.step(actions)

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
        scaled_actions = G.a.scale_action(G.envs.action_space, actions)
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: traininG.
        if global_step > Args.learning_starts:
            data = rb.sample(Args.batch_size)

            if Args.autotune:
                ent_coef_value, G.ec.nt = G.ec.jitted_stateless_call(G.ec.get_state(include_ov=False))

            ((a_state, qf_state), (qf_loss_value, qf_values), key) = Critic.update(
                G.a.get_state(),
                G.qf.get_state(include_target=True),
                ent_coef_value,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
                key,
            )
            G.a.t, G.a.nt, G.a.ov = a_state
            G.qf.t, G.qf.nt, G.qf.ov. G.qf.tt, G.qf.tnt = qf_state
            if global_step % Args.policy_frequency == 0:  # TD 3 Delayed update support
                (a_state, qf_state, actor_loss_value, key, entropy) = Actor.update(
                    G.a.get_state(),
                    G.qf.get_state(),
                    ent_coef_value,
                    data.observations.numpy(),
                    key,
                )
                G.a.t, G.a.nt, G.a.ov = a_state
                G.qf.t, G.qf.nt, G.qf.ov = qf_state

                if Args.autotune:
                    ent_coef_state, ent_coef_loss = EntropyCoef.update(G.ec.get_state(), entropy)

            # update the target networks
            if global_step % Args.target_network_frequency == 0:
                G.qf.tt, G.qf.tnt = Critic.soft_update(G.qf.t, G.qf.tt,  G.qf.tt, G.qf.tnt)

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
                Utils.update_models_state_and_save(global_step)

    G.envs.close()
    writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
