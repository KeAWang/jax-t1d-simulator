import jax
import jax.numpy as jnp
import simglucose
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
patient, init_state, params, unused_params = simglucose.initialize_patient(
    key, 0, False
)

# TODO: how does basal insulin work? Is it all injected at once? Injected at a regular rate?
# TODO: what are reasonable ranges for basal and bolus dosages?
# TODO: what are tests we could do to check on the "realisticness" of our simulator?
# TODO: what are example patient dosages and profiles? How do we calculate/prescribe dosages?
basal = patient.u2ss * patient.BW / 6000  # U/min
planned_meal = jnp.array(0.0)

states = []
obs = []

carbs = []
insulins = []

state = init_state
t = jnp.arange(60 * 24)  # one day
for i in t:
    ins = basal
    carb = 0
    if (i % 100 == 0) and (i < len(t) // 2):
        carb = 80.0
        ins = 80.0 / 6.0 + basal
    controls = simglucose.Controls(carbs=carb, insulin=ins)
    carbs.append(carb)
    insulins.append(ins)

    state, planned_meal = simglucose.step(
        patient, state, planned_meal, controls, None, params
    )
    obs.append(simglucose.observe(state, patient.Vg))
    states.append(state)
states = jnp.stack(states)
obs = jnp.stack(obs)

plt.ion()

# Show the states trajectories
# fig, ax = plt.subplots()
# ax.plot(t, states)

# Show the CGM observations and the controls
fig, ax = plt.subplots()
ax.plot(t, obs)
ax.plot(t, carbs)
ax.plot(t, insulins)
ax.set(ylim=[0, 400])
input("Press any key to exit...")
