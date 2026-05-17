from dataclasses import dataclass

@dataclass(frozen=True)
class HullWhiteModel :
    """
    One-factor Hull-White short-rate model.

    Dynamics under the risk-neutral measure:

        dr_t = [theta(t) - a r_t] dt + sigma dW_t

    Parameters
    ----------
    mean_reversion : float
        Mean-reversion speed a.
    volatility : float
        Short-rate volatility sigma.
    """

    mean_reversion : float
    volatility : float

    def __post_init__(self) -> None :
        if self.mean_reversion <= 0 :
            raise ValueError("mean_reversion must be strictly positive")
    
        if self.volatility <= 0 :
            raise ValueError("volatility must be strictly positive")
        
    @property
    def parameters(self)-> tuple[float, float] :
        return self.mean_reversion, self.volatility