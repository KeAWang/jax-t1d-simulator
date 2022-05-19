import jax
import jax.numpy as jnp
import simglucose
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, namedtuple

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
patient, init_state, params, unused_params = simglucose.initialize_patient(
    key, 0, False
)

basal = patient.u2ss * patient.BW / 6000  # U/min
planned_meal = jnp.array(0.0)

num_days = 7
total_time = 60.0 * 24 * num_days  # mins in a week
sample_time = 5.0  # min
T = 1 + int(total_time / sample_time)
MealDistribution = namedtuple("MealDistribution", ["m", "s"])
meals = OrderedDict(
    {
        "breakfast": MealDistribution(m=60.0 * 8, s=20),  # minutes
        "lunch": MealDistribution(m=60.0 * 4, s=40),
        "dinner": MealDistribution(m=60.0 * 6, s=60),
    }
)


def sample_meals(meals, key):
    while True:
        total_time = 0
        for meal_name, meal in meals.items():
            key, subkey = jax.random.split(key)
            time_delta = jax.random.poisson(subkey, meal.m)
            key, subkey = jax.random.split(key)
            meal_size = jax.random.poisson(subkey, meal.s)  # Fixed for now
            total_time = total_time + time_delta
            yield jnp.stack([time_delta, meal_size])
        yield jnp.array([24 * 60 - total_time, 0.0])  # midnight is a dummy meal


num_meals = num_days * (3 + 1)
meal_events = jnp.stack(
    [tup[1] for tup in zip(range(num_meals), sample_meals(meals, key))]
)
meal_events = meal_events[:-1]  # TODO: why?
meal_times = jnp.cumsum(meal_events[:, 0])
meal_sizes = meal_events[:, 1]
meal_times_idx = jnp.floor(meal_times / sample_time) + 1

ins_times_idx = meal_times_idx - 2  # Hard coded to coincide with meal
ins_sizes = meal_sizes / 6 + basal

meal_times_idx_dense = np.zeros(T)
meal_times_idx_dense[np.array(meal_times_idx, dtype=int)] = meal_sizes
ins_times_idx_dense = np.zeros(T)
ins_times_idx_dense[np.array(ins_times_idx, dtype=int)] = ins_sizes


plt.ion()

# fig, ax = plt.subplots()
# ax.plot(meal_times_idx_dense)
# ax.plot(ins_times_idx_dense)

states = []
obs = []

state = init_state
t = np.arange(T)
for i in t:
    ins = ins_times_idx_dense[i]
    carb = meal_times_idx_dense[i]
    controls = simglucose.Controls(carbs=carb, insulin=ins)

    state, planned_meal = simglucose.step(
        patient, state, planned_meal, controls, None, params
    )
    obs.append(simglucose.observe(state, patient.Vg))
    states.append(state)
states = jnp.stack(states)
obs = jnp.stack(obs)

fig, ax = plt.subplots(figsize=(21, 3))
ax.plot(t, obs)
ax.plot(t, meal_times_idx_dense)
ax.plot(t, ins_times_idx_dense)
ax.set(ylim=[0, 400])
input("Press any key to exit...")
