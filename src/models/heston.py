from dataclasses import dataclass

@dataclass(frozen=True)
class HestonModel :
    """
    Heston stochastic volatility model.

    Dynamics under the risk-neutral measure:

        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
        dv_t = kappa(theta - v_t) dt + sigma sqrt(v_t) dW_t^v

        corr(dW_t^S, dW_t^v) = rho

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    sigma : float
        Volatility of variance.
    rho : float
        Correlation between spot and variance.
    v0 : float
        Initial variance.
    """

    kappa : float
    theta : float
    sigma : float
    rho : float
    v0 : float

    def __post_init__(self) -> None :

        if self.kappa <= 0 :
            raise ValueError("kappa must be strictly positive")
        
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        
        if not -1.0 < self.rho < 1.0 : 
            raise ValueError("rho must be between -1 and 1")
        
        if self.v0 <= 0:
            raise ValueError("v0 must be strictly positive")
        
    @property
    def feller_condition(self)-> bool :
        return 2.0 * self.kappa *self.theta > self.sigma ** 2
    
    @property
    def parameters(self)->tuple[float, float, float, float, float] :
        return self.kappa, self.theta, self.sigma, self.rho, self.v0