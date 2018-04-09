import scipy.stats
import scipy.special

"""Mean-normalized distributions.

We define mean-normalized versions of distribution objects, where the density is
divided by expected density. The normalization factor for  Exponential
distribution is $\frac 1 {2 \cdot scale}$, and for Gamma distribution is
$\frac{\Gamma\left( 2\cdot a-1\right) }{{2}^{2\cdot a - 1} \cdot {{\Gamma\left(
a\right) }^{2}}\cdot \mathit{scale}}$.
"""

class _Normdist(object):
    """Mean-normalized continuous distribution wrapper.
    """
    def __init__(self, dist, norm):
        self.dist = dist
        self.norm = norm
        
    def pdf(self, x):
        d = self.dist.pdf(x)
        return d/self.norm
    
    def __getattr__(self, name):
        return getattr(self.dist, name)
    
def nexpon(scale):
    """Mean-normalized exponential distribution.
    """
    return _Normdist(scipy.stats.expon(scale=scale), 1/(2. * scale))

def ngamma(a, scale):
    """Mean-normalizer gamma distribution.
    """
    return _Normdist(scipy.stats.gamma(a=a, scale=scale),
                     scipy.special.gamma(2 * a   - 1) /
                     (2 ** (2 * a - 1) * scipy.special.gamma(a) ** 2 * scale))
