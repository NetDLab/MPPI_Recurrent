import jax
import jax.numpy as jnp
from functools import partial

class CarEnv:
    def __init__(self, v=3.0, R=1.0, dt=0.01):
        self.v = v
        self.R = R
        self.dt = dt

    @partial(jax.jit, static_argnums=(0,))
    def dynamics(self, state, control):
        d = state[..., 0]
        theta = state[..., 1]
        omega = state[..., 2]
        u = control

        # State derivatives
        d_dot = -self.v * jnp.cos(theta)
        theta_dot = omega
        omega_dot = u

        # Euler integration
        d_next = d + d_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        omega_next = omega + omega_dot * self.dt

        next_state = jnp.stack([d_next, theta_next, omega_next], axis=-1)
        return next_state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, rng, state, control, params):
        next_state = self.dynamics(state, control)
        d_next = next_state[..., 0]

        # Define the target set (e.g., d <= 0, meaning collision with the obstacle)
        done = d_next <= 0.0

        # Reward can be defined as negative distance (to be maximized)
        reward = -d_next

        obs = next_state
        info = {}
        return obs, next_state, reward, done, info

    def reset(self, key, state=None):
        if state is None:
            # Default initial state
            state = jnp.array([5.0, 0.0, 0.0])  # [d, theta, omega]
        obs = state
        return state, obs
