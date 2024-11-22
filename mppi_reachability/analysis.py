import jax
import jax.numpy as jnp

def reachability_analysis(env, mppi_controller, T=0.5, N_samples=100, dt=0.01):
    H = int(T / dt)

    # State space grid for d and theta
    d_values = jnp.linspace(0.5, 5.0, 20)
    theta_values = jnp.linspace(-jnp.pi, jnp.pi, 20)

    reachable_set = []

    for d0 in d_values:
        for theta0 in theta_values:
            x0 = jnp.array([d0, theta0, 0.0])  # Initial state [d, theta, omega]
            all_reach_target = True

            # Sample control policies
            for sample_idx in range(N_samples):
                # Reset controller parameters
                control_params = mppi_controller.reset()
                state = x0
                done = False

                for t in range(H):
                    # Use a unique key for each action to ensure different samples
                    rng_key = jax.random.PRNGKey(sample_idx * H + t)
                    rng, rng_action = jax.random.split(rng_key)
                    control_input, control_params, _ = mppi_controller(
                        obs=state,
                        env_state=state,
                        env_params=None,
                        rng_act=rng_action,
                        control_params=control_params,
                        info={"noisy_state": state},
                    )
                    control_input = jnp.clip(control_input, -1.0, 1.0)

                    # Environment step
                    obs, next_state, reward, done, info = env.step_env(
                        rng, state, control_input, None
                    )
                    state = next_state

                    # Check if reached the target set
                    if state[0] <= 0.0:
                        # Reached the obstacle surface
                        break

                else:
                    # Did not reach the target set within the time horizon
                    all_reach_target = False
                    break  # No need to test more samples

            if all_reach_target:
                reachable_set.append((d0.item(), theta0.item()))

    return reachable_set
