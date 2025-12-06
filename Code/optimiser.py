from Code.archive.assumptions import *
from profile import generate_hourly_solar_profile

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import time
from pyomo.opt import SolverFactory

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))

def optimise_bess(
    solar_profile,
    solar_capex,
    bess_energy_capex,
    power_capex,          # NEW: cost per kW of BESS inverter
    load=1.0,             # [MW] Average load to serve
    availability=0.8,     # [%] Target availability or percentage of demand to meet
    efficiency=0.9,       # round-trip efficiency approx (simplified)
    start_soc=0.5         # initial SoC as fraction of energy capacity
):
    """
    Optimizes Solar and BESS capacity to meet a specified demand target at minimum cost.

    New (updated):
    - Separates BESS energy capacity (kWh) and power capacity (kW)
    - Cost function:
        CAPEX = solar_capex*solar_capacity +
                bess_energy_capex*bess_energy +
                power_capex*bess_power
    """

    periods = len(solar_profile)
    demand = np.full(periods, load)
    T = range(periods)

    model = pyo.ConcreteModel(name="Solar_BESS_Optimisation")
    model.T = pyo.Set(initialize=T)

    # --------------------------------------------------------
    # DECISION VARIABLES
    # --------------------------------------------------------
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy    = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_power     = pyo.Var(within=pyo.NonNegativeReals)   # NEW inverter size variable

    model.bess_flow      = pyo.Var(model.T, within=pyo.Reals)      # + discharge, - charge
    model.soc            = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # --------------------------------------------------------
    # SOC BALANCE
    # --------------------------------------------------------
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == m.bess_energy * start_soc
        return m.soc[t] == m.soc[t-1] - m.bess_flow[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    # Storage upper bound
    model.soc_limit = pyo.Constraint(model.T, rule=lambda m, t: m.soc[t] <= m.bess_energy)

    # --------------------------------------------------------
    # POWER LIMITS (UPDATED)
    # --------------------------------------------------------
    model.power_limits = pyo.ConstraintList()
    for t in T:

        # BESS inverter rating now independent
        model.power_limits.add(model.bess_flow[t] <=  model.bess_power)
        model.power_limits.add(model.bess_flow[t] >= -model.bess_power)

        # Cannot discharge more than what’s stored
        model.power_limits.add(model.bess_flow[t] <= model.soc[t] * efficiency)

        # Charging limited by solar availability
        model.power_limits.add(-model.bess_flow[t] <= model.solar_capacity * solar_profile[t])

    # --------------------------------------------------------
    # ENERGY SERVED
    # --------------------------------------------------------
    def served_rule(m, t):
        # Served ≤ direct solar + BESS discharge
        return m.energy_served_t[t] <= m.solar_capacity * solar_profile[t] + m.bess_flow[t]
    model.served_balance = pyo.Constraint(model.T, rule=served_rule)

    # Cannot exceed demand
    model.demand_cap = pyo.Constraint(
        model.T, rule=lambda m, t: m.energy_served_t[t] <= demand[t]
    )

    # Availability constraint
    model.availability_constraint = pyo.Constraint(
        expr=sum(model.energy_served_t[t] for t in T) >= availability * sum(demand)
    )

    # --------------------------------------------------------
    # OBJECTIVE FUNCTION (UPDATED)
    # --------------------------------------------------------
    model.cost = pyo.Objective(
        expr=( model.solar_capacity * solar_capex
             + model.bess_energy    * bess_energy_capex
             + model.bess_power     * power_capex ),   # NEW term
        sense=pyo.minimize
    )

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------
    solver = SolverFactory("cbc")
    solver.solve(model, tee=False)

    cost        = pyo.value(model.cost)
    solar_cap   = pyo.value(model.solar_capacity)
    bess_energy = pyo.value(model.bess_energy)
    bess_power  = pyo.value(model.bess_power)

    # --------------------------------------------------------
    # RESULTS + Post-processing
    # --------------------------------------------------------
    usable_fraction = 0.90
    soc_min_frac = 0.05
    soc_max_frac = 0.95

    bess_energy_true = bess_energy / usable_fraction

    bess_flow   = np.array([pyo.value(model.bess_flow[t]) for t in T])
    soc_raw     = np.array([pyo.value(model.soc[t])       for t in T])
    served      = np.array([pyo.value(model.energy_served_t[t]) for t in T])
    solar_gen   = np.array([solar_cap * solar_profile[t]  for t in T])

    soc_scaled = soc_min_frac * bess_energy_true + \
                 (soc_raw / bess_energy) * usable_fraction * bess_energy_true

    bess_discharge = np.clip(bess_flow, 0, None)
    solar_charge   = np.clip(-bess_flow, 0, None)

    solar_used = np.minimum(
        solar_gen - solar_charge,
        demand - bess_discharge
    )
    solar_used = np.clip(solar_used, 0, None)

    solar_curtailed = np.clip(solar_gen - solar_used - solar_charge, 0, None)
    unserved        = np.clip(demand - served, 0, None)

    results_data = pd.DataFrame({
        "Hour":                list(T),
        "Solar_Gen_MWh":       solar_gen,
        "BESS_Flow_MWh":       bess_flow,
        "BESS_Discharge_MWh":  bess_discharge,
        "Solar_Charge_MWh":    solar_charge,
        "Solar_Used_MWh":      solar_used,
        "Solar_Curtailed_MWh": solar_curtailed,
        "SOC_MWh":             soc_scaled,
        "Energy_Served_MWh":   served,
        "Energy_Unserved_MWh": unserved,
    })

    # cycles per year based on full-energy throughput / usable capacity
    cycles = bess_discharge.sum() / bess_energy_true

    return (
        cost,
        round(solar_cap, 2),
        round(bess_energy_true, 2),
        round(bess_power, 2),
        results_data,
        cycles
    )

def optimise_availability(
    solar_profile,
    solar_capacity,
    bess_energy,
    load,
    efficiency=0.9,
    start_soc=0.5
):
    """
    Dispatch optimiser for fixed Solar + BESS capacities.
    Uses the same single bess_flow formulation as optimise_bess().

    Returns
    -------
    availability : float
        Fraction of demand served over the horizon.
    results_df : pd.DataFrame
        Hourly timeseries with the same columns as optimise_bess():
            Hour
            Solar_Gen_MWh
            BESS_Flow_MWh
            BESS_Discharge_MWh
            Solar_Charge_MWh
            Solar_Used_MWh
            Solar_Curtailed_MWh
            SOC_MWh
            Energy_Served_MWh
            Energy_Unserved_MWh
    """

    periods = len(solar_profile)
    T = range(periods)
    demand = np.full(periods, load)

    model = pyo.ConcreteModel(name="Solar_BESS_Dispatch")
    model.T = pyo.Set(initialize=T)

    # --- Vars ---
    model.bess_flow       = pyo.Var(model.T, within=pyo.Reals)
    model.soc             = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # --- Constraints ---
    usable_fraction = 0.9 # accounting for depth-of-discharge
    usable_bess_energy = bess_energy * usable_fraction

    # SoC balance
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == usable_bess_energy * start_soc
        return m.soc[t] == m.soc[t-1] - m.bess_flow[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    # Storage capacity
    def soc_limit_rule(m, t):
        return m.soc[t] <= usable_bess_energy
    model.soc_limit = pyo.Constraint(model.T, rule=soc_limit_rule)

    # Power & operation limits
    model.power_limits = pyo.ConstraintList()
    for t in T:
        model.power_limits.add(model.bess_flow[t] <= solar_capacity)
        model.power_limits.add(model.bess_flow[t] >= -solar_capacity)
        model.power_limits.add(model.bess_flow[t] <= model.soc[t] * efficiency)
        model.power_limits.add(-model.bess_flow[t] <= solar_capacity * solar_profile[t])

    # Energy served constraint:
    def served_rule(m, t):
        return m.energy_served_t[t] <= solar_capacity * solar_profile[t] + m.bess_flow[t]
    model.served_balance = pyo.Constraint(model.T, rule=served_rule)

    # Demand cap
    def demand_cap_rule(m, t):
        return m.energy_served_t[t] <= demand[t]
    model.demand_cap = pyo.Constraint(model.T, rule=demand_cap_rule)

    # Objective: maximise total energy served
    model.obj = pyo.Objective(
        expr=sum(model.energy_served_t[t] for t in T),
        sense=pyo.maximize
    )

    # --- Solve ---
    solver = SolverFactory("cbc")
    result = solver.solve(model, tee=False)

    from pyomo.opt import TerminationCondition
    if result.solver.termination_condition == TerminationCondition.infeasible:
        print("⚠️ Infeasible — returning availability = 0")
        return 0.0, pd.DataFrame()

    # --- Extract results ---
    bess_flow   = [pyo.value(model.bess_flow[t])      for t in T]
    soc         = [pyo.value(model.soc[t])            for t in T]
    served      = [pyo.value(model.energy_served_t[t])for t in T]
    solar_gen   = [solar_capacity * solar_profile[t]  for t in T]

    bess_flow_arr = np.array(bess_flow)
    solar_gen_arr = np.array(solar_gen)
    served_arr    = np.array(served)
    demand_arr    = demand.astype(float)

    bess_discharge = np.clip(bess_flow_arr, 0.0, None)
    solar_charge   = np.clip(-bess_flow_arr, 0.0, None)

    solar_used = np.minimum(
        solar_gen_arr - solar_charge,
        demand_arr - bess_discharge
    )
    solar_used = np.clip(solar_used, 0.0, None)

    solar_curtailed = solar_gen_arr - solar_used - solar_charge
    solar_curtailed = np.clip(solar_curtailed, 0.0, None)

    unserved = demand_arr - served_arr
    unserved = np.clip(unserved, 0.0, None)

    total_energy  = served_arr.sum()
    total_demand  = demand_arr.sum()
    availability  = total_energy / total_demand if total_demand > 0 else 0.0

    df = pd.DataFrame({
        "Hour":                list(T),
        "Solar_Gen_MWh":       solar_gen_arr,
        "BESS_Flow_MWh":       bess_flow_arr,
        "BESS_Discharge_MWh":  bess_discharge,
        "Solar_Charge_MWh":    solar_charge,
        "Solar_Used_MWh":      solar_used,
        "Solar_Curtailed_MWh": solar_curtailed,
        "SOC_MWh":             soc,
        "Energy_Served_MWh":   served_arr,
        "Energy_Unserved_MWh": unserved,
    })

    return round(availability, 2), df

if __name__ == "__main__":
    latitude = 19.4326
    longitude = 99.1332
    #country = "Australia"
    """
    # Filter the row where 'Country' matches
    row = countries_df[countries_df['Country'] == country]

    if not row.empty:
        latitude = row.iloc[0]['Latitude']
        longitude = row.iloc[0]['Longitude']
        print(f"{country}: lat={latitude}, lon={longitude}")
    else:
        print(f"Country '{country}' not found in DataFrame.")
    """
    solar_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    print("got solar profile")
    profile = solar_profile
    solar_capex = 690
    bess_energy_capex = 191
    bess_power_capex = 300
    cost, solar_capacity, bess_energy, bess_power, results_1, discharge = optimise_bess(solar_profile, solar_capex, bess_energy_capex, bess_power_capex, availability=0.95)
    print(f"solar cap is {solar_capacity}, bess is {bess_energy}, bess power is {bess_power}, bess cycles are {discharge}")
    availability, results_2 = optimise_availability(profile, solar_capacity, bess_energy, 1)
    results_1.to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS results\opti_results.csv')
    results_2.to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS results\avail_results.csv')
    print(f"availability is {availability}")
    # Setting up environment
    """
    latitude = 19.4326
    longitude = 99.1332
    print("getting solar profile...")
    yearly_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    print("got solar profile")
    # demand_profile = np.full(len(yearly_profile), 100)  # Demand profile in MW

    # Run optimization
    optimise_bess(yearly_profile, capex_learning_df, 2020)
    
    
    wind test
        #latitude = 55.3781
    #longitude = 3.4360
    wind_profile = parse_renewables_ninja(r"C:\\Users\\barna\Downloads\\ninja_wind_54.7867_-1.9809_corrected.csv")
"""