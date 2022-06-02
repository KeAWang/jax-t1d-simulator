import jax
import jax.numpy as jnp
import simglucose
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, namedtuple

jax.config.update("jax_platform_name", "cpu")  # running on CPU is faster than GPU

key = jax.random.PRNGKey(0)

all_obs = []
all_meals = []
all_ins = []
for _ in range(10):
    key, subkey = jax.random.split(key)
    patient_state, params, unused_params = simglucose.initialize_patient(key, 0, False)
    basal = patient_state.patient.u2ss * patient_state.patient.BW / 6000  # U/min
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

    t = np.arange(T)
    for i in t:
        ins = ins_times_idx_dense[i]
        carb = meal_times_idx_dense[i]
        controls = simglucose.Controls(carbs=carb, insulin=ins)

        patient_state = simglucose.step(patient_state, controls, params)
        obs.append(simglucose.observe(patient_state.state, patient_state.patient.Vg))
        states.append(patient_state.state)
    states = jnp.stack(states)
    obs = jnp.stack(obs)

    all_obs.append(obs)
    all_meals.append(meal_times_idx_dense)
    all_ins.append(ins_times_idx_dense)

fig, axes = plt.subplots(figsize=(21, 6), nrows=2, sharex=True)
ax = axes[0]
ax.plot(t, all_obs[0])
ax.plot(t, all_meals[0])
ax.plot(t, all_ins[0])
ax.set(ylim=[0, 400])

ax = axes[1]
for obs in all_obs:
    ax.plot(t, obs, c="blue", alpha=0.2)
    ax.set(ylim=[0, 400])
    ax.set(xlabel="t")
input("Press any key to exit...")
