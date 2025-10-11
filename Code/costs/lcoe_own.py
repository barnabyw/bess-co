import numpy_financial as npf

def lcoe(annual_output, capital_cost, annual_operating_cost, discount_rate, lifetime):
    """Compute levelised cost of electricity

    Arguments
    ---------
    annual_output : float
    capital_cost : float
    annual_operating_cost : float
    discount_rate : float
    lifetime : int

    Returns
    -------
    float
    """
    total_operating_cost = npf.pv(discount_rate, lifetime, -annual_operating_cost, when=1)
    discounted_total_cost = capital_cost + total_operating_cost

    discounted_output = npf.pv(discount_rate, lifetime, -annual_output, when=1)
    print(discounted_output)

    return discounted_total_cost / discounted_output

if __name__ == "__main__":
    pass