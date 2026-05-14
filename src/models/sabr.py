from dataclasses import dataclass

@dataclass(frozen=True)
class SABRModel :
    """
     SABR model parameters.

    The SABR dynamics are:

        dF_t = alpha_t * F_t^beta * dW_t
        d alpha_t = nu * alpha_t * dZ_t
        corr(dW_t, dZ_t) = rho

    Parameters
    ----------
    alpha : float
        Initial volatility level.
    beta : float
        Elasticity parameter.
    rho : float
        Correlation between forward and volatility.
    nu : float
        Volatility of volatility.
    """
    alpha : float
    beta : float
    rho : float
    nu : float

    def __post_init__(self) -> None :
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive")
        
        if not 0.0 <= self.beta <= 1.0 : 
            raise ValueError("beta must be between 0 and 1")
        
        if not -1.0 < self.rho < 1.0 : 
            raise ValueError("rho must be between -1 and 1")
        
        if self.nu <= 0:
            raise ValueError("nu must be positive")
    
    @property
    def parameter(self) -> tuple[float, float, float, float] :
        return self.alpha, self.beta, self.rho, self.nu
    
    def with_parameters(
            self,
            alpha : float | None = None, 
            beta : float | None = None, 
            rho : float | None = None,
            nu : float | None = None,
            ) ->  "SABRModel":
        """
        Return a new SABRModel with updated parameters.
        """

        return SABRModel(
            alpha=self.alpha if alpha is None else alpha,
            beta=self.beta if beta is None else beta,
            rho=self.rho if rho is None else rho,
            nu=self.nu if nu is None else nu,
        )