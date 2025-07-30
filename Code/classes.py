import numpy as np

class Solar:
    def __init__(self, capex_per_mw, hourly_output):
        self.capex_per_mw = capex_per_mw
        self.hourly_output = np.array(hourly_output)

    def get_output(self, capacity_mw):
        return self.hourly_output * capacity_mw

    def print_output(self):
        print(f'hourly output {self.hourly_output}')

    def get_cost(self, capacity_mw):
        return self.capex_per_mw * capacity_mw

class BESS:
    def __init__(self, cost_per_mw, cost_per_mwh, rte):
        self.cost_per_mw = cost_per_mw
        self.cost_per_mwh = cost_per_mwh
        self.rte = rte

    def get_cost(self, power_mw, energy_mwh):
        return self.cost_per_mw * power_mw + self.cost_per_mwh * energy_mwh

    def simulate(self, solar_output, power_mw, energy_mwh, demand_profile, charge_eff=0.9, discharge_eff=0.9,
                 initial_soc=0.0):
        soc = initial_soc
        soc_profile = []
        discharge_profile = []
        charge_profile = []
        unmet_demand = 0
        delivered_energy = 0

        for hour, solar_gen in enumerate(solar_output):
            demand = demand_profile[hour]

            # Charge battery from solar
            charge = min(solar_gen, power_mw)
            soc = min(soc + charge * charge_eff, energy_mwh)

            # Discharge battery to meet demand
            discharge = min(soc * discharge_eff, min(power_mw, demand))
            soc -= discharge / discharge_eff

            # Track served demand
            served = min(discharge, demand)
            unmet_demand += max(0, demand - served)
            delivered_energy += served

            # Record profiles
            soc_profile.append(soc)
            discharge_profile.append(discharge)
            charge_profile.append(charge)

        return delivered_energy, soc_profile, discharge_profile, charge_profile, unmet_demand

def objective_lcoe(x, solar, bess, hourly_output, demand_profile, discount_rate=0.08, lifetime_years=20):
    solar_capacity, bess_power, bess_energy = x
    solar_output = solar.get_output(solar_capacity)
    annual_energy, _, _, _, _ = bess.simulate(solar_output, bess_power, bess_energy, demand_profile=demand_profile)
    if annual_energy <= 0:
        return 1e6
    total_cost = solar.get_cost(solar_capacity) + bess.get_cost(bess_power, bess_energy)
    discounted_energy = sum(annual_energy / ((1 + discount_rate)**year) for year in range(1, lifetime_years+1))
    if discounted_energy <= 0:
        return 1e6
    return total_cost/discounted_energy
