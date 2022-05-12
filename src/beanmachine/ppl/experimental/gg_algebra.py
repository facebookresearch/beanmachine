import numpy as np
import scipy.special as sp

def _eq(a,b):
    return abs(a-b) < 1e-15

def _leq(a,b):
    return a - b < 1e-15

def _geq(a,b):
    return b - a < 1e-15

class GGTail(object):
    
    def __init__(self, nu, sigma, rho):
        #self._check(nu, sigma, rho)
        self.nu    = nu
        self.sigma = sigma
        self.rho   = rho
            
    def __add__(self, other):
        """The addition operation (additive convolution)"""
        
        if not isinstance(other, self.__class__):
            return self._copy()
        
        nu_1, sigma_1, rho_1 = self.params()
        nu_2, sigma_2, rho_2 = other.params()
        
        if _eq(rho_1, rho_2):
            rho = 0.5*(rho_1+rho_2)
            
            if _eq(rho, 1):
                # Exponential case
                return self._rv(nu_1 + nu_2 + 1, 
                                min(sigma_1, sigma_2), 1)
            elif rho > 1:
                # Superexponential case
                sigma  = sigma_1**(-1.0/(rho-1))
                sigma += sigma_2**(-1.0/(rho-1))
                sigma = sigma**(1 - rho)
                return self._rv(nu_1 + nu_2 + 1 - rho_1/2,
                                sigma, rho)
            
            else:
                # Subexponential case
                return max(self, other)._copy()
        
        else:
            return max(self, other)._copy()
        
    def __mul__(self, other):
        """The multiplication operation (multiplicative convolution)"""
        
        if not isinstance(other, self.__class__):
            # Scalar multiplication
            return self._rv(self.nu, 
                            self.sigma / abs(other)**self.rho,
                            self.rho)
        
        nu_1, sigma_1, rho_1 = self.params()
        nu_2, sigma_2, rho_2 = other.params()
        
        if _leq(rho_1,0) and rho_2  > 0:
            return self._reg(abs(nu_1))
        if rho_1 > 0 and _leq(rho_2, 0):
            return self._reg(abs(nu_2))
        if _eq(rho_1,0) and _eq(rho_2,0):
            return self._reg(min(abs(nu_1),abs(nu_2)))
        
        mu = 1.0/abs(rho_1) + 1.0/abs(rho_2)
        sigma  = (sigma_1*abs(rho_1))**(1.0/(mu*abs(rho_1)))
        sigma *= (sigma_2*abs(rho_2))**(1.0/(mu*abs(rho_2)))
        sigma *= mu
        
        if rho_1 < 0:
            nu = nu_1/abs(rho_1) + nu_2/abs(rho_2) + 1/2
            return self._rv(nu/mu, sigma, -1/mu)
        else:
            nu = nu_1/rho_1 + nu_2/rho_2 - 1/2
            return self._rv(nu/mu, sigma, 1/mu)
        
    def __pow__(self, num):
        """The power operation"""
        if (num < 0) and (_eq(self.rho, 0) or _leq((self.nu+1)/self.rho, 0)):
            # Reciprocal by assuming bounded density near zero
            return self._reg(2.0)
        
        return self._rv((self.nu+1)/num - 1, self.sigma, self.rho/num)
    
    def __and__(self, other):
        """Product of densities operation"""
        assert isinstance(other, self.__class__)
        nu_1, sigma_1, rho_1 = self.params()
        nu_2, sigma_2, rho_2 = other.params()
        nu = nu_1 + nu_2
        if _eq(rho_1, rho_2):
            rho = 0.5*(rho_1 + rho_2)
            return self._rv(nu, sigma_1 + sigma_2, rho)
        if rho_1 < rho_2:
            return self._rv(nu, sigma_1, rho_1)
        else:
            return self._rv(nu, sigma_2, rho_2)
        
        
    def __sub__(self, other):
        """The subtraction operation"""
        return self + (-1.0 * other)
    
    def __rsub__(self, other):
        """The subtraction operation"""
        return (-1.0 * self) + other
        
    def __truediv__(self, other):
        """The division operation (involving reciprocal)"""
        return self * (other ** (-1))
    
    def __rtruediv__(self, other):
        """Reciprocal operation"""
        return self ** (-1)
    
    def __abs__(self):
        """Absolute value operation"""
        return self._copy()
    
    def log(self):
        """The logarithm operation"""
        if self.nu < -1 and _leq(self.rho, 0):
            return self._rv(0.0, abs(self.nu)-1, 1.0)
        
        return self._light()
    
    def exp(self):
        """The exponential operation"""
        if _geq(self.rho, 1):
            return self._reg(self.sigma + 1)
        
        return self._heavy()
    
        
    __radd__ = __add__
    __rmul__ = __mul__
    
    ### ORDERING ###
    
    def __ge__(self, other):
        """Comparing heaviness of the tail: does
        self have a heavier tail than other?"""
        nu_1, sigma_1, rho_1 =  self.params()
        nu_2, sigma_2, rho_2 = other.params()
        
        if _leq(rho_1, 0) and _leq(rho_2, 0):
            return (abs(nu_1) < abs(nu_2))
        
        if _eq(rho_1, rho_2):
            if _eq(sigma_1, sigma_2):
                return nu_1 > nu_2
            return sigma_1 < sigma_2
        
        return (rho_1 < rho_2)
        
    def __eq__(self, other):
        return _eq(self.nu,other.nu) and \
                _eq(self.sigma,other.sigma) and \
                _eq(self.rho,other.rho)
    
    def __gt__(self, other):
        return (self.__ge__(other)) and not (self.__eq__(other))
    
    def __ne__(self, other):
        return not (self.__eq__(other))
    
    def __lt__(self, other):
        return not (self.__ge__(other))
    
    def __le__(self, other):
        return other.__ge__(self)
    
    ### STRING REPRESENTATION ###
    
    def __str__(self):
        
        try:
            if _eq(self.nu, -1) and _leq(self.rho, 0):
                return "super heavy tail"
            if np.isinf(self.rho):
                return "super light tail"
        except TypeError:
            pass
        
        tail = "c"
        try:
            nu = '%s' % float('%.4g' % self.nu)
            if not _eq(self.nu, 0):
                if self.nu < 0:
                    tail += " x^({nu})".format(nu=nu)
                else:
                    tail += " x^{nu}".format(nu=nu)
        except TypeError:
            tail += " x^({nu})".format(nu=self.nu)
        try:
            if _eq(self.sigma, 1.0):
                sigma = ""
            else:
                sigma = '%s' % float('%.4g' % self.sigma)
                sigma += " * "
        except TypeError:
            sigma = "({sigma}) * ".format(sigma=str(self.sigma))
        try:
            if _eq(self.rho, 1.0):
                rho = ""
            else:
                rho = '%s' % float('%.4g' % self.rho)
                if self.rho < 0:
                    rho = "^({rho})".format(rho=rho)
                else:
                    rho = "^{rho}".format(rho=rho)
            if not _eq(self.rho, 0):
                tail += " exp(-{sigma}x{rho})".format(
                    sigma=sigma, rho=rho)
        except TypeError:
            tail += " exp(-{sigma}x^({rho}))".format(
                    sigma=sigma, rho=self.rho)
        
        return tail
    
    def __repr__(self):
        return str(self)
    
    ### HELPERS ###
    
    def _check(self, nu, sigma, rho):
        """Check if the parameters are sane"""
        positivity = None
        try:
            positivity = sigma > 0
        except TypeError:
            pass
        if positivity is not None:
            assert positivity
        if _leq(rho, 0):
            assert _leq(nu, 1.0)
            
    def params(self):
        """Return a tuple of the class parameters"""
        return (self.nu, self.sigma, self.rho)
            
    def _copy(self):
        """Return a copy of the current tail object"""
        return self.__class__(self.nu, self.sigma, self.rho)
                                  
    def _rv(self,nu,sigma,rho):
        """Creates a new random variable with given parameters"""
        return self.__class__(nu,sigma,rho)
                
    def _reg(self,nu):
        """Creates a new regularly-varying random variable with
        parameter nu"""
        return self.__class__(-nu,1,0)
    
    def _light(self):
        """Creates a new super-light random variable"""
        return self.__class__(0.0, 1.0, np.inf)
    
    def _heavy(self):
        """Creates a new super-heavy random variable"""
        return self._reg(1.0)
    

RegVar = lambda nu: GGTail(-nu, 1.0, 0.0)
SuperHeavy = lambda: RegVar(1.0)
SuperLight = lambda: GGTail(0.0, 1.0, np.inf)
    
benktander2 = lambda a,b: GGTail(2*b-2, a/b, b)
betaprime   = lambda a,b,*vars: RegVar(b+1)
burr        = lambda c,k: RegVar(c*k+1)
cauchy      = lambda *args: RegVar(2.0)
chi         = lambda k: GGTail(k-1,0.5,2)
chi2        = lambda k: GGTail(k/2-1,0.5,1)
dagum       = lambda a,*args: RegVar(a+1)
davis       = lambda n,b,*args: GGTail(-1-n,b,1)
exponential = lambda lam: GGTail(0,lam,1)
fdistn      = lambda d1,d2: RegVar(d2/2+1)
fisherz     = lambda d1,d2: GGTail(0,d2,1)
frechet     = lambda a,lam,*args: GGTail(-1-a,lam**a,-a)
gamma       = lambda a,b: GGTail(a-1,b,1)
ggompertz   = lambda b,s: GGTail(0,b*s,1)
hyperbolic  = lambda alpha,beta,lam,*args: GGTail(lam-1,alpha-beta,1)
gennormal   = lambda alpha,beta,*args: GGTail(0, alpha**(-beta),beta)
geostable   = lambda alpha: RegVar(alpha+1)
gompertz    = lambda *args: SuperLight()
gumbel      = lambda beta,*args: GGTail(0, 1.0/beta, 1)
gumbel2     = lambda alpha,beta: GGTail(-alpha-1,beta,-alpha)
holtsmark   = lambda: RegVar(5/2)
hypsecant   = lambda: GGTail(0,np.pi/2,1)
invchi2     = lambda k: GGTail(-k/2-1,1/2,-1)
invgamma    = lambda alpha,beta: GGTail(-alpha-1,beta,-1)
invnormal   = lambda *args: GGTail(-2,1/2,-2)
levy        = lambda c,*args: GGTail(-3/2,c/2,-1)
laplace     = lambda lam: GGTail(0,1/lam,1)
logistic    = lambda lam: GGTail(0,1/lam,1)
logcauchy   = lambda *args: SuperHeavy()
loglaplace  = lambda lam,*args: RegVar(1/lam+1)
loglogistic = lambda alpha,beta: RegVar(beta+1)
logt        = lambda *args: SuperHeavy()
lomax       = lambda alpha, *args: RegVar(alpha+1)
maxboltz    = lambda sigma: GGTail(2, 1/(2*sigma**2), 2)
normal      = lambda mu,sigma: GGTail(0, 1/(2*sigma**2), 2)
pareto      = lambda alpha, *args: RegVar(alpha+1)
rayleigh    = lambda sigma: GGTail(1, 1/(2*sigma**2), 2)
rice        = lambda sigma, *args: GGTail(1/2, 1/(2*sigma**2), 2)
skewnormal  = lambda mu,sigma: GGTail(0, 1/(2*sigma**2), 2)
slash       = lambda: GGTail(-2,1/2,2)
stable      = lambda alpha: RegVar(alpha+1)
student     = lambda nu: RegVar(nu+1)
tracywidom  = lambda beta: (-3*beta/4 - 1, 2*beta/3, 3/2)
voigt       = lambda *args: RegVar(2)
weibull     = lambda rho, lam: GGTail(rho-1, 1/lam**rho, rho)
    
############# FUNCTIONS #############
    
class TailFunction(object):
    
    def __init__(self, func, num_args):
        self.func = func
        self.num_args = num_args
        assert num_args > 0
        
    def filter_args(self, args):
        assert len(args) == self.num_args
        list_vars = list(args)
        idx = 0
        for var in list_vars:
            if not isinstance(var, GGTail):
                list_vars.pop(idx)
            else:
                idx += 1
        return list_vars
        
class PowerFunction(TailFunction):
    
    def __init__(self, alpha, func, num_args, const = 1.0):
        super().__init__(func, num_args)
        self.alpha = alpha
        self.const = const
            
    def __call__(self, *args):
        list_vars = self.filter_args(args)
        if len(list_vars) == 0:
            return self.func(*args)
        else:
            return self.const * max(list_vars)**self.alpha
        
class BoundedFunction(TailFunction):
    
    def __init__(self, bound, func, num_args):
        super().__init__(func, num_args)
        self.bound = bound
        
    def __call__(self, *args):
        list_vars = self.filter_args(args)
        if len(list_vars) == 0:
            return self.func(*args)
        else:
            return self.bound
        
class BoundedFunction(TailFunction):
    
    def __init__(self, bound, func, num_args):
        super().__init__(func, num_args)
        self.bound = bound
        
    def __call__(self, *args):
        list_vars = self.filter_args(args)
        if len(list_vars) == 0:
            return self.func(*args)
        else:
            return self.bound
        
class ExpFunction(TailFunction):
    def __init__(self, scale, exponent, func, num_args):
        super().__init__(func, num_args)
        self.scale = scale
        self.exponent = exponent
        
    def __call__(self, *args):
        list_vars = self.filter_args(args)
        if len(list_vars) == 0:
            return self.func(*args)
        else:
            return (max(list_vars)**self.exponent * self.scale).exp()
        
class LogFunction(TailFunction):
    def __init__(self, scale, func, num_args):
        super().__init__(func, num_args)
        self.scale = scale
        
    def __call__(self, *args):
        list_vars = self.filter_args(args)
        if len(list_vars) == 0:
            return self.func(*args)
        else:
            return (max(list_vars).log() * self.scale)
        
LipschitzFunction = lambda func, num_args, const = 1.0: \
                        PowerFunction(1.0, func, num_args, const)
    
exp = ExpFunction(1.0, 1.0, np.exp, 1)
log = LogFunction(1.0, np.log, 1)
relu = LipschitzFunction(lambda x: max([x,0]), 1, 1.0)
sin = BoundedFunction(1.0, np.sin, 1)
cos = BoundedFunction(1.0, np.cos, 1)
sinh = ExpFunction(0.5, 1.0, np.sinh, 1)
cosh = ExpFunction(0.5, 1.0, np.cosh, 1)
tanh = BoundedFunction(np.pi / 2, np.tanh, 1)
arcsinh = LogFunction(1.0, np.arcsinh, 1)
arccosh = LogFunction(1.0, np.arccosh, 1)
sqrt = PowerFunction(0.5, np.sqrt, 1, 1.0)
erf = BoundedFunction(1.0, sp.erf, 1)

############# MATRICES #############

def MatGGTail(size, distn):
    matrix = np.empty(size, dtype=object)
    for idx in range(size[0]):
        for idy in range(size[1]):
            matrix[idx,idy] = GGTail(*distn.params())
    return matrix

gauss_ens = lambda *size: MatGGTail(size, normal(0,1))