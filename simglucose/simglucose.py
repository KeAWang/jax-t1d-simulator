import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from collections import namedtuple


# TODO: make sure that all units make sense
# TODO: make into brax interface
# TODO: make vpatient_params.csv installable via pip

"""
# TODO:
1. rollout function that takes in the number of timesteps to unroll T1D simulator for (Euler steps), controls sequence (i.e. carbs and insulin)
2. step function that takes in static patient params (e.g. BW, u2ss), the planned meals, controls, and returns the new state
3. initial state distribution, patient parameter distribution, patient profile

"""

# DONE
# TODO: separate out the speed tests into unit tests

SAMPLE_TIME = 1  # min
EAT_RATE = 5  # g/min CHO

Patient = namedtuple("Patient", ["BW", "u2ss", "Vg"])

Params = namedtuple(
    "Params",
    [
        "b",  # the second flex point of gastric emptying rate
        "d",  # the first flex point of gastric emptying rate (called c in the original paper on oral glucose)
        "kmin",
        "kmax",
        "kabs",
        "kp1",
        "kp2",
        "kp3",
        "Fsnc",
        "ke1",
        "ke2",
        "f",  # subcutaneous glucose (mg / kg)
        "Vm0",
        "Vmx",
        "Km0",
        "k1",
        "k2",
        "m1",
        "m2",
        "m30",
        "m4",
        "ka1",
        "ka2",
        "Vi",
        "p2u",
        "Ib",
        "ki",
        "kd",
        "ksc",
    ],
)

Controls = namedtuple("Controls", ["carbs", "insulin"])


def load_patient(patient_id):
    """
    Construct patient by patient_id
    id are integers from 1 to 30.
    1  - 10: adolescent#001 - adolescent#010
    11 - 20: adult#001 - adult#001
    21 - 30: child#001 - child#010
    """
    patient_params = pd.read_csv("simglucose/vpatient_params.csv")
    params = patient_params.iloc[patient_id - 1, :]
    return params


def eat(planned_meal, new_meal, eat_rate=EAT_RATE):
    # planned_meal is meal that hasn't been eaten yet
    planned_meal = planned_meal + new_meal
    to_eat = (planned_meal > 0.0) * jnp.minimum(eat_rate, planned_meal)
    planned_meal = jnp.maximum(0, planned_meal - to_eat)
    return planned_meal, to_eat


@jax.jit
def step(
    patient: Patient,
    state,
    planned_meal,
    controls: Controls,
    static_params,
    params: Params,
    step_size=SAMPLE_TIME,
):
    # Note we handle food going into stomach differently than simglucose
    carbs = controls.carbs  # CHO
    new_planned_meal, to_eat = eat(planned_meal, carbs)
    dstate = t1d_dynamics(patient, state, controls, static_params, params, to_eat)
    new_state = state + step_size * dstate
    return new_state, new_planned_meal


def t1d_dynamics_np(patient, x, action, params, Dbar):
    # Original implementation from simglucose
    # Useful as a reference
    dxdt = np.zeros(13)
    d = action.carbs * 1000  # g -> mg
    insulin = action.insulin * 6000 / patient.BW  # U/min -> pmol/kg/min
    basal = patient.u2ss * patient.BW / 6000  # U/min

    # Glucose in the stomach
    qsto = x[0] + x[1]

    # Stomach solid
    dxdt[0] = -params.kmax * x[0] + d

    if Dbar > 0:
        aa = 5 / 2 / (1 - params.b) / Dbar
        cc = 5 / 2 / params.d / Dbar
        kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
            np.tanh(aa * (qsto - params.b * Dbar))
            - np.tanh(cc * (qsto - params.d * Dbar))
            + 2
        )
    else:
        kgut = params.kmax

    # stomach liquid
    dxdt[1] = params.kmax * x[0] - x[1] * kgut

    # intestine
    dxdt[2] = kgut * x[1] - params.kabs * x[2]

    # Rate of appearance
    Rat = params.f * params.kabs * x[2] / patient.BW
    # Glucose Production
    EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
    # Glucose Utilization
    Uiit = params.Fsnc

    # renal excretion
    if x[3] > params.ke2:
        Et = params.ke1 * (x[3] - params.ke2)
    else:
        Et = 0

    # glucose kinetics
    # plus dextrose IV injection input u[2] if needed
    dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
    dxdt[3] = (x[3] >= 0) * dxdt[3]

    Vmt = params.Vm0 + params.Vmx * x[6]
    Kmt = params.Km0
    Uidt = Vmt * x[4] / (Kmt + x[4])
    dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
    dxdt[4] = (x[4] >= 0) * dxdt[4]

    # insulin kinetics
    dxdt[5] = (
        -(params.m2 + params.m4) * x[5]
        + params.m1 * x[9]
        + params.ka1 * x[10]
        + params.ka2 * x[11]
    )  # plus insulin IV injection u[3] if needed
    It = x[5] / params.Vi
    dxdt[5] = (x[5] >= 0) * dxdt[5]

    # insulin action on glucose utilization
    dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

    # insulin action on production
    dxdt[7] = -params.ki * (x[7] - It)

    dxdt[8] = -params.ki * (x[8] - x[7])

    # insulin in the liver (pmol/kg)
    dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
    dxdt[9] = (x[9] >= 0) * dxdt[9]

    # subcutaneous insulin kinetics
    dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
    dxdt[10] = (x[10] >= 0) * dxdt[10]

    dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
    dxdt[11] = (x[11] >= 0) * dxdt[11]

    # subcutaneous glucose
    dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
    dxdt[12] = (x[12] >= 0) * dxdt[12]

    return dxdt


@jax.jit
def t1d_dynamics(
    patient: Patient,
    state,
    controls: Controls,
    static_params,
    params: Params,
    Dbar,
):
    """
    Notes on history of UVA/Padova T1D simulator:
    1. A System Model of Oral Glucose Absorption: Validation on Gold Standard Data (2006)
        - Introduced the equations that describe insulin rate of appearance in plasma from ingested glucose
    2. Meal Simulation Model of the Glucose-Insulin System (2007)
        - Introduced the quations that describe glucose insulin dynamics
    3. GIM, Simulation Software of Meal Glucose–Insulin Model (2007)
        - Introduced the MATLAB code for Meal Glucose-Insulin Model
    4. In Silico Preclinical Trials: A Proof of Concept in Closed-Loop Control of Type 1 Diabetes (2008/2009)
        - Implement software for simulation-based (in silico) testing of control algorithms for T1D management
        - Have 300 virtual patients (sampled from some distribution of parameters)
        - Account for CGM sensor errors for some CGMs
        - Simulate insulin delivery for some insulin pumps
        - FDA approved as a substitute for animal trials in preclinical testing of closed-loop controls
    5. The UVA/PADOVA Type 1 Diabetes Simulator
        - Improved glucose kinetics in hypoglycemia
        - Add glucagon kinetics
        - Better describes hypoglycemic events compared to the 2008 simulator
    6. The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day
        - Updates T1D sim from single meal to single day


    The notation and abbreviations are clarified in
        Physical Activity into the Meal Glucose–Insulin Model of Type 1 Diabetes: In Silico Studies
    """
    # TODO: assign all at once instead of one at a time
    dstate = jnp.zeros_like(state)

    BW = patient.BW  # body weight
    u2ss = patient.u2ss  # steady state insulin rate per kilogram (pmol / (L * kg))

    carbs = controls.carbs  # CHO
    insulin = controls.insulin
    insulin = insulin * 6000 / BW  # U/min -> pmol/kg/min
    # For logging and debugging
    carbs_mg = 1000 * carbs  # g -> mg
    basal = u2ss * BW / 6000  # U / min

    """ Description of the state dimensions
    # Stomach equations that describe oral glucose absorption:
    See
        A System Model of Oral Glucose Absorption: Validation on Gold Standard Data 
        by Dalla Man et al.
    for more details
    state[0] := q_sto1 (amount of solid glucose in stomach)
    state[1] := q_sto2 (amount of liquid glucose in stomach)
    state[2] := q_gut (glucose mass in the intestine)

    q_sto = q_sto1 + q_sto2
    kgut(q_sto1, q_sto2) is a nonlinear function (the tanh equations)

    Ra(t) = f \cdot k_abs \cdot q_gut(t) is the rate of appearance of glucose in plasma produced from ingested glucose 

    Some more notes on notation:
        D is the amount of ingested glucose (same as Dbar)
        delta is the impulse function, i.e. D \cdot delta(t)


    """
    # Glucose in the stomach
    Qsto1, Qsto2 = state[0], state[1]
    Qsto = Qsto1 + Qsto2

    # Stomach solid
    # d(Q_sto1)/dt
    dstate = dstate.at[0].set(-params.kmax * state[0] + carbs_mg)

    aa = 5 / 2 / (1 - params.b) / Dbar  # same as alpha in original paper
    cc = 5 / 2 / params.d / Dbar  # same as beta in original paper
    kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
        jnp.tanh(aa * (Qsto - params.b * Dbar))
        - jnp.tanh(cc * (Qsto - params.d * Dbar))
        + 2
    )
    kgut = jnp.where(Dbar > 0.0, kgut, params.kmax)

    # stomach liquid
    # d(Q_sto2)/dt
    dstate = dstate.at[1].set(params.kmax * state[0] - state[1] * kgut)

    # intestine
    # d(Q_gut)/dt
    dstate = dstate.at[2].set(kgut * state[1] - params.kabs * state[2])

    # Rate of appearance
    Rat = params.f * params.kabs * state[2] / BW
    """
    # Equations that describe glucose-insulin dynamics:
    See
        Meal Simulation Model of the Glucose-Insulin System
        by Dalla Man et al.

    state[3] := G_p (mg/kg of glucose masses in plasma and rapidly equilibrating tissues)
    state[4] := G_t (mg/kg of glucose masses in slowly equilibrating tissues)
    Some notation:
        Suffix b denotes basal state. e.g G_pb, G_tb, G_b are the basal states of G_p, G_t, G
        Plasma glucose concentration (G) (mg/dl)
        Insulin concentration (I)
        Rate of appearance of glucose in plasma (Ra)  (mg/kg/min)
        Rate of endogenous glucose production (EGP)  (mg/kg/min)
        Glucose utilization (U)
        Degradation (D)
        Renal excretion (E) (mg/kg/min)
        Insulin independent and dependent glucose utilizations (U_ii, U_id)  (mg/kg/min)
        Distribution volume of glucose (V_G) (dl/kg)
        
        Insulin masses in plasma (I_p) (pmol/kg)
        Insulin masses in liver (I_l) (pmol/kg)
        Plasma insulin concentration (I) (pmol/l)
        Insulin secretion (S) (pmol/kg/min)
        Distribution volume of insulin (V_I) (l/kg)
    """

    # Glucose Production
    EGPt = params.kp1 - params.kp2 * state[3] - params.kp3 * state[8]
    # Glucose Utilization
    Uiit = params.Fsnc

    # renal excretion
    """
    Glucose excretion by the
    kidney occurs if plasma glucose exceeds a certain threshold and
    can be modeled by a linear relationship with plasma glucose
    """
    Et = jnp.maximum(params.ke1 * (state[3] - params.ke2), 0.0)

    # glucose kinetics
    # plus dextrose IV injection input u[2] if needed
    dstate = dstate.at[3].set(
        jnp.maximum(EGPt, 0)
        + Rat
        - Uiit
        - Et
        - params.k1 * state[3]
        + params.k2 * state[4]
    )
    dstate = dstate.at[3].multiply(state[3] >= 0.0)

    Vmt = params.Vm0 + params.Vmx * state[6]
    Kmt = params.Km0
    Uidt = Vmt * state[4] / (Kmt + state[4])
    dstate = dstate.at[4].set(-Uidt + params.k1 * state[3] - params.k2 * state[4])
    dstate = dstate.at[4].multiply(state[4] >= 0.0)

    # insulin kinetics
    dstate = dstate.at[5].set(
        -(params.m2 + params.m4) * state[5]
        + params.m1 * state[9]
        + params.ka1 * state[10]
        + params.ka2 * state[11]
    )  # plus insulin IV injection u[3] if needed
    It = state[5] / params.Vi
    dstate = dstate.at[5].multiply(state[5] >= 0.0)

    # insulin action on glucose utilization
    dstate = dstate.at[6].set(-params.p2u * state[6] + params.p2u * (It - params.Ib))

    # insulin action on production
    dstate = dstate.at[7].set(-params.ki * (state[7] - It))

    dstate = dstate.at[8].set(-params.ki * (state[8] - state[7]))

    # insulin in the liver (pmol/kg)
    dstate = dstate.at[9].set(
        -(params.m1 + params.m30) * state[9] + params.m2 * state[5]
    )
    dstate = dstate.at[9].multiply(state[9] >= 0.0)

    # subcutaneous insulin kinetics
    dstate = dstate.at[10].set(insulin - (params.ka1 + params.kd) * state[10])
    dstate = dstate.at[10].multiply(state[10] >= 0.0)

    dstate = dstate.at[11].set(params.kd * state[10] - params.ka2 * state[11])
    dstate = dstate.at[11].multiply(state[11] >= 0.0)

    # subcutaneous glucose
    dstate = dstate.at[12].set(-params.ksc * state[12] + params.ksc * state[3])
    dstate = dstate.at[12].multiply(state[12] >= 0.0)

    return dstate


def observe(state, Vg):
    GM = state[12]  # subcutaneous glucose (mg/kg)
    Gsub = GM / Vg
    observation = Gsub
    return observation


def initialize_patient(key, patient_id, random_init_bg=True):
    all_params = load_patient(patient_id)
    patient = Patient(
        **{
            k: jnp.array(float(v))
            for k, v in all_params.items()
            if k in Patient._fields
        }
    )
    params = Params(
        **{k: jnp.array(float(v)) for k, v in all_params.items() if k in Params._fields}
    )
    unused_params = {
        k: v for k, v in all_params.items() if k not in Params._fields + Patient._fields
    }

    init_state = all_params.iloc[2:15].array.astype(float)
    if random_init_bg:
        # Currently equivalent to N(0, I) noise
        mean = jnp.array(
            [1.0 * init_state[3], 1.0 * init_state[4], 1.0 * init_state[12]]
        )
        cov = jnp.array(
            [0.1 * init_state[3], 0.1 * init_state[4], 0.1 * init_state[12]]
        )
        bg_init = jax.random.normal(key, (3,)) * jnp.sqrt(cov) + mean
        bg_init = np.array(bg_init)
        init_state[3] = bg_init[0]
        init_state[4] = bg_init[1]
        init_state[12] = bg_init[2]

    init_state = jnp.asarray(init_state)
    return patient, init_state, params, unused_params
