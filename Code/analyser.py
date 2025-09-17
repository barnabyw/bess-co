import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class OptimizationParams:
    """Configuration for optimization parameters"""
    solar_year: int = 2023
    discount_rate: float = 0.08
    project_lifetime: int = 20
    load_factor: float = 1.0  # Load multiplier
    target_hours: int = 8760


class SolarBESSAnalyzer:
    """
    Unified class for Solar-BESS optimization and analysis across multiple scenarios
    """

    def __init__(self, capex_learning_df: pd.DataFrame,
                 country_coords_df: pd.DataFrame,
                 params: OptimizationParams = None):
        """
        Initialize the analyzer with cost projections and country data

        Args:
            capex_learning_df: DataFrame with columns ['year', 'solar_cost_per_mw', 'bess_energy_cost_per_mwh']
            country_coords_df: DataFrame with columns ['Country', 'Latitude', 'Longitude']
            params: Optimization parameters
        """
        self.capex_df = capex_learning_df
        self.country_coords = country_coords_df
        self.params = params or OptimizationParams()

        # Cache for solar profiles to avoid regenerating
        self._solar_profile_cache = {}

    @lru_cache(maxsize=128)
    def _get_solar_profile(self, lat: float, lon: float) -> np.ndarray:
        """Cached solar profile generation"""
        cache_key = (lat, lon, self.params.solar_year)
        if cache_key not in self._solar_profile_cache:
            self._solar_profile_cache[cache_key] = generate_hourly_solar_profile(
                lat, lon, solar_year=self.params.solar_year
            )
        return self._solar_profile_cache[cache_key]

    def _calculate_lcoe(self, total_cost: float) -> float:
        """Calculate LCOE using consistent parameters"""
        load_total = load * self.params.target_hours * self.params.load_factor
        return 1000 * lcoe(load_total, total_cost, 0,
                           self.params.discount_rate, self.params.project_lifetime)

    def optimize_single_location_year(self, lat: float, lon: float, year: int,
                                      availability: Optional[float] = None) -> Dict:
        """
        Optimize for a single location and year

        Returns:
            Dict with keys: cost, solar_capacity, bess_energy, lcoe
        """
        solar_profile = self._get_solar_profile(lat, lon)

        kwargs = {'availability': availability} if availability is not None else {}
        cost, solar_capacity, bess_energy, levcost = optimise_bess(
            solar_profile, self.capex_df, year, **kwargs
        )

        return {
            'cost': cost,
            'solar_capacity': solar_capacity,
            'bess_energy': bess_energy,
            'lcoe': levcost
        }

    def analyze_multi_year_fixed_capacity(self, base_results_df: pd.DataFrame,
                                          years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Analyze multiple years using fixed capacities from base year but varying costs

        Args:
            base_results_df: DataFrame with optimization results for base year
            years: List of years to analyze (if None, uses all years in capex_df)

        Returns:
            DataFrame with results for all years
        """
        if years is None:
            years = self.capex_df['year'].tolist()

        # Base columns that don't change
        base_columns = ['Country', 'Latitude', 'Longitude', 'Solar_Capacity', 'BESS_Energy']
        base_data = base_results_df[base_columns].copy()

        all_results = []

        for year in years:
            cost_row = self.capex_df[self.capex_df['year'] == year].iloc[0]
            solar_cost_per_mw = cost_row['solar_cost_per_mw']
            bess_cost_per_mwh = cost_row['bess_energy_cost_per_mwh']

            year_data = base_data.copy()
            year_data['Year'] = year

            # Recalculate costs with new prices
            year_data['Cost'] = (year_data['Solar_Capacity'] * solar_cost_per_mw +
                                 year_data['BESS_Energy'] * bess_cost_per_mwh)

            year_data['LCOE'] = year_data['Cost'].apply(self._calculate_lcoe)

            all_results.append(year_data)

        return pd.concat(all_results, ignore_index=True)

    def analyze_single_location_time_series(self, lat: float, lon: float,
                                            years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Analyze a single location across multiple years with re-optimization each year

        Args:
            lat, lon: Location coordinates
            years: Years to analyze

        Returns:
            DataFrame with yearly results
        """
        if years is None:
            years = self.capex_df['year'].tolist()

        results = []

        # Get optimal capacities for one year (used as reference)
        ref_year = years[0] if years else 2025
        ref_result = self.optimize_single_location_year(lat, lon, ref_year)

        for year in years:
            # Get costs for this year
            cost_row = self.capex_df[self.capex_df['year'] == year].iloc[0]
            solar_cost = cost_row['solar_cost_per_mw']
            bess_cost = cost_row['bess_energy_cost_per_mwh']

            # Calculate cost using reference capacities but current year prices
            total_cost = (ref_result['solar_capacity'] * solar_cost +
                          ref_result['bess_energy'] * bess_cost)
            lcoe_val = self._calculate_lcoe(total_cost)

            results.append({
                'Year': year,
                'LCOE': lcoe_val,
                'Cost': total_cost,
                'Solar_Capacity': ref_result['solar_capacity'],
                'BESS_Energy': ref_result['bess_energy']
            })

        return pd.DataFrame(results)

    def analyze_countries_single_year(self, countries: Optional[List[str]] = None,
                                      year: int = 2025) -> pd.DataFrame:
        """
        Analyze multiple countries for a single year

        Args:
            countries: List of country names (if None, uses all countries)
            year: Year to analyze

        Returns:
            DataFrame with country results
        """
        if countries is None:
            country_data = self.country_coords
        else:
            country_data = self.country_coords[self.country_coords['Country'].isin(countries)]

        results = []

        for _, row in country_data.iterrows():
            country = row['Country']
            lat = row['Latitude']
            lon = row['Longitude']

            print(f"Processing {country}...")

            opt_result = self.optimize_single_location_year(lat, lon, year)

            results.append({
                'Country': country,
                'Latitude': lat,
                'Longitude': lon,
                'Year': year,
                **opt_result
            })

        return pd.DataFrame(results)

    def analyze_availability_sensitivity(self, countries: List[str],
                                         availabilities: List[float],
                                         year: int = 2024) -> pd.DataFrame:
        """
        Analyze sensitivity to availability parameter across countries

        Args:
            countries: List of country names
            availabilities: List of availability values to test
            year: Year for analysis

        Returns:
            DataFrame with sensitivity results
        """
        results = []

        for country in countries:
            country_data = self.country_coords[self.country_coords['Country'] == country].iloc[0]
            lat = country_data['Latitude']
            lon = country_data['Longitude']

            print(f"Processing {country}...")

            for availability in availabilities:
                opt_result = self.optimize_single_location_year(lat, lon, year, availability)

                results.append({
                    'Country': country,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Year': year,
                    'Availability': availability,
                    **opt_result
                })

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, filename: str, base_path: str = None):
        """Save results to CSV with consistent formatting"""
        if base_path is None:
            base_path = r'C:\Users\barna\OneDrive\Documents\Solar_BESS results'

        filepath = os.path.join(base_path, filename)
        df.to_csv(filepath, index=False)
        print(f"Results saved to '{filepath}'")
        return filepath
