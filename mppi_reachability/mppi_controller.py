import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
import chex

@struct.dataclass
class MPPIParams:
    gamma_mean: float
    gamma_sigma: float
    discount: float
    a_mean: jnp.ndarray
    a_var: jnp.ndarray

class MPPIController:
    def __init__(self, env, control_params: MPPIParams, N: int, H: int, lam: float) -> None:
        self.env = env
        self.init_control_params = control_params
        self.N = N  # Number of samples
        self.H = H  # Time horizon
        self.lam = lam

    def reset(self, env_state=None, env_params=None, control_params=None, key=None) -> MPPIParams:
        return self.init_control_params

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs: jnp.ndarray,
        env_state: jnp.ndarray,
        env_params,
        rng_act: chex.PRNGKey,
        control_params: MPPIParams,
        info,
    ) -> chex.Array:
        env_state = info["noisy_state"]

        # Shift the action mean and variance (move one step ahead)
        a_mean_old = control_params.a_mean
        a_var_old = control_params.a_var

        control_params = control_params.replace(
            a_mean=jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]]),
            a_var=jnp.concatenate([a_var_old[1:], a_var_old[-1:]]),
        )

        # Sample actions from a normal distribution (scalar actions)
        rng_act, act_key = jax.random.split(rng_act)
        act_keys = jax.random.split(act_key, self.N)

        def single_sample(key, traj_mean, traj_var):
            return traj_mean + jnp.sqrt(traj_var) * jax.random.normal(key, shape=(self.H,))

        # Generate N samples of action sequences
        a_sampled = jax.vmap(single_sample, in_axes=(0, None, None))(
            act_keys, control_params.a_mean, control_params.a_var
        )

        a_sampled = jnp.clip(a_sampled, -1.0, 1.0)

        # Rollout the trajectories using the sampled actions
        rng_act, step_key = jax.random.split(rng_act)

        # Prepare initial states for all samples
        state_repeat = jnp.repeat(env_state[None, :], self.N, axis=0)
        done_repeat = jnp.full(self.N, False)
        reward_repeat = jnp.full(self.N, 0.0)

        def rollout_fn(carry, action):
            env_state, reward_before, done_before = carry
            obs, env_state, reward, done, info = self.env.step_env(
                step_key, env_state, action, None
            )
            reward = jnp.where(done_before, reward_before, reward)
            done = done | done_before
            return (env_state, reward, done), (reward, env_state)

        # Perform the rollouts
        _, (rewards, states) = jax.lax.scan(
            rollout_fn,
            (state_repeat, reward_repeat, done_repeat),
            a_sampled.T,
            length=self.H,
        )

        # Compute the total discounted rewards
        rewards = rewards.transpose(1, 0)
        discounts = control_params.discount ** jnp.arange(self.H)
        discounted_rewards = jnp.sum(rewards * discounts, axis=1)

        # Compute the costs and weights
        cost = -discounted_rewards
        cost_exp = jnp.exp(-(cost - jnp.min(cost)) / self.lam)
        weight = cost_exp / jnp.sum(cost_exp)

        # Update the action mean and variance
        a_mean = (
            jnp.sum(weight[:, None] * a_sampled, axis=0) * control_params.gamma_mean
            + control_params.a_mean * (1 - control_params.gamma_mean)
        )
        a_var = (
            jnp.sum(
                weight[:, None] * (a_sampled - a_mean) ** 2,
                axis=0,
            )
            * control_params.gamma_sigma
            + control_params.a_var * (1 - control_params.gamma_sigma)
        )
        control_params = control_params.replace(a_mean=a_mean, a_var=a_var)

        # Select the first action to apply
        u = control_params.a_mean[0]

        # Information for debugging
        info = {"pos_mean": None, "pos_std": None}

        return u, control_params, info
