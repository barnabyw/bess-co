import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional
from lcoe.lcoe import lcoe

from optimiser import optimise_bess
from profile import generate_hourly_solar_profile
from assumptions import load, target, capex_learning_df


@dataclass
class OptimizationParams:
    solar_year: int = 2023
    discount_rate: float = 0.08
    project_lifetime: int = 20
    load_factor: float = 1.0
    target_hours: int = 8760
    target_availability: float = target


class SolarBESSAnalyzer:
    """Solar-BESS optimization and analysis across multiple scenarios"""

    def __init__(self, capex_learning_df: pd.DataFrame,
                 country_coords_df: pd.DataFrame,
                 params: OptimizationParams = None):
        self.capex_df = capex_learning_df
        self.country_coords = country_coords_df
        self.params = params or OptimizationParams()

    # ---- CORE HELPERS ----
    @lru_cache(maxsize=128)
    def _get_solar_profile(self, lat: float, lon: float) -> np.ndarray:
        return generate_hourly_solar_profile(lat, lon, solar_year=self.params.solar_year)

    def _calc_lcoe(self, total_cost: float) -> float:
        load_total = load * self.params.target_hours * self.params.load_factor
        return 1000 * lcoe(load_total, total_cost, 0,
                           self.params.discount_rate, self.params.project_lifetime)

    def _get_costs_for_year(self, year: int) -> Dict[str, float]:
        row = self.capex_df.loc[self.capex_df['year'] == year].iloc[0]
        return dict(solar=row['solar_cost_per_mw'], bess=row['bess_energy_cost_per_mwh'])

    # ---- MAIN METHODS ----
    def optimize_single(self, lat: float, lon: float, year: int,
                        availability: Optional[float] = None) -> Dict:
        solar_profile = self._get_solar_profile(lat, lon)
        cost, solar_cap, bess_energy, levcost, _ = optimise_bess(
            solar_profile, self.capex_df, year,
            **({'availability': availability} if availability else {})
        )
        return dict(cost=cost, solar_capacity=solar_cap,
                    bess_energy=bess_energy, lcoe=levcost)

    def analyze_multi_year_fixed_capacity(self, base_df: pd.DataFrame,
                                          years: Optional[List[int]] = None) -> pd.DataFrame:
        years = years or self.capex_df['year'].tolist()
        base = base_df[['Country', 'Latitude', 'Longitude', 'Solar_Capacity', 'BESS_Energy']]

        results = []
        for year in years:
            costs = self._get_costs_for_year(year)
            df = base.copy()
            df['Year'] = year
            df['Cost'] = (df['Solar_Capacity'] * costs['solar'] +
                          df['BESS_Energy'] * costs['bess'])
            df['LCOE'] = df['Cost'].apply(self._calc_lcoe)
            results.append(df)
        return pd.concat(results, ignore_index=True)

    def analyze_single_location(self, lat: float, lon: float,
                                years: Optional[List[int]] = None) -> pd.DataFrame:
        years = years or self.capex_df['year'].tolist()
        ref_result = self.optimize_single(lat, lon, years[0])
        solar_cap, bess_energy = ref_result['solar_capacity'], ref_result['bess_energy']

        records = []
        for year in years:
            c = self._get_costs_for_year(year)
            total_cost = solar_cap * c['solar'] + bess_energy * c['bess']
            records.append({
                'Year': year, 'LCOE': self._calc_lcoe(total_cost),
                'Cost': total_cost, 'Solar_Capacity': solar_cap, 'BESS_Energy': bess_energy
            })
        return pd.DataFrame(records)

    def analyze_countries(self, countries: Optional[List[str]] = None,
                          year: int = 2024) -> pd.DataFrame:
        df = (self.country_coords if countries is None
              else self.country_coords[self.country_coords['Country'].isin(countries)])

        results = []
        for _, row in df.iterrows():
            print(f"Processing {row.Country}...")
            res = self.optimize_single(row.Latitude, row.Longitude, year)
            results.append({'Country': row.Country, 'Latitude': row.Latitude,
                            'Longitude': row.Longitude, 'Year': year, **res})
        return pd.DataFrame(results)

    def analyze_availability(self, countries: List[str], availabilities: List[float],
                             year: int = 2024) -> pd.DataFrame:
        rows = []
        for country in countries:
            lat, lon = self.country_coords.query("Country == @country")[['Latitude', 'Longitude']].iloc[0]
            for avail in availabilities:
                res = self.optimize_single(lat, lon, year, availability=avail)
                rows.append({'Country': country, 'Latitude': lat, 'Longitude': lon,
                             'Year': year, 'Availability': avail, **res})
        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame, filename: str,
                     base_path: str = r'C:\Users\barna\OneDrive\Documents\Solar_BESS results'):
        filepath = os.path.join(base_path, filename)
        df.to_csv(filepath, index=False)
        print(f"Results saved to '{filepath}'")
        return filepath
