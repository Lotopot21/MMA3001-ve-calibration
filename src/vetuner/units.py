"""Unit conversions and fuel chemistry helpers."""

STOICH_AFR_PETROL = 14.7


def afr_to_lambda(afr: float, stoich_afr: float = STOICH_AFR_PETROL) -> float:
    """Convert air-fuel ratio to lambda.

    Parameters
    ----------
    afr : float
        Air-fuel ratio by mass. Must be positive.
    stoich_afr : float, optional
        Stoichiometric AFR for the fuel, by default 14.7 for petrol.

    Returns
    -------
    float
        Lambda, the ratio of actual AFR to stoichiometric AFR.
        Values below 1.0 are rich, above 1.0 are lean.

    Raises
    ------
    ValueError
        If `afr` or `stoich_afr` is not positive.
    """
    if afr <= 0 or stoich_afr <= 0:
        raise ValueError("AFR values must be positive")
    return afr / stoich_afr