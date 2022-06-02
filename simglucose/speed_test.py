import time
import jax
import jax.numpy as jnp
import simglucose

jax.config.update("jax_platform_name", "cpu")  # running on CPU is faster than GPU

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
patient, init_state, params, unused_params = simglucose.initialize_patient(
    key, 0, False
)

planned_meal = jnp.array(0.0)

controls = simglucose.Controls(carbs=jnp.array(0.0), insulin=jnp.array(0.0))
state = init_state

carbs = controls.carbs  # CHO
new_planned_meal, to_eat = simglucose.eat(planned_meal, carbs)
Dbar = to_eat

num_iters = 100
# Compilation
dstate = simglucose.t1d_dynamics(
    patient, state, controls, params, to_eat
).block_until_ready()
start = time.time()
for _ in range(num_iters):
    dstate = simglucose.t1d_dynamics(
        patient, state, controls, params, to_eat
    ).block_until_ready()
end = time.time()
jax_time = (end - start) / num_iters
print("JAX Time elapsed per ODE step:", jax_time)

start = time.time()
for _ in range(num_iters):
    dstate_np = simglucose.t1d_dynamics_np(patient, state, controls, params, Dbar)
end = time.time()
np_time = (end - start) / num_iters
print("Numpy Time elapsed per ODE step:", np_time)

print(f"Speedup factor: {np_time / jax_time:.2f}")

vmap_t1d_dynamics = jax.vmap(simglucose.t1d_dynamics, (None, 0, None, None, None))
num_patients = 100
dstate = vmap_t1d_dynamics(
    patient,
    jnp.tile(state[None, :], (num_patients, 1)),
    controls,
    params,
    to_eat,
).block_until_ready()
start = time.time()
for _ in range(num_iters):
    dstate = vmap_t1d_dynamics(
        patient,
        jnp.tile(state[None, :], (num_patients, 1)),
        controls,
        params,
        to_eat,
    ).block_until_ready()
end = time.time()
vmap_time = (end - start) / num_iters
print("JAX Time elapsed per vmapped ODE step (100 patients):", vmap_time)
print(
    f"Vmap speedup over sequential looping: {num_patients * jax_time / vmap_time: .2f}"
)
