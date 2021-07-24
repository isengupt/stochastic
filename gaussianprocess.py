import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
np.random.seed(42)

def exponentiated_quadratic(xa, xb):
   
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def GP(X1, y1, X2, kernel_func):
   
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    print(μ2, Σ2)
    return μ2, Σ2  

def GP_noise(X1, y1, X2, kernel_func, σ_noise):


    
    Σ11 = kernel_func(X1, X1) + ((σ_noise ** 2) * np.eye(n1))
    
    Σ12 = kernel_func(X1, X2)
    
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    
    μ2 = solved @ y1
    
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  

class BrownianMotion:
  def __init__(self,total_time, nb_steps, nb_processes, mean):
        self.total_time = total_time
        self.nb_steps = nb_steps
        self.delta_t = total_time / nb_steps
        self.nb_processes = nb_processes 
        self.mean = mean  
        self.stdev = np.sqrt(self.delta_t) 
    
  def sim_1d(self):
      self.distances = np.cumsum(np.random.normal(self.mean, self.stdev, (self.nb_processes, self.nb_steps)), axis=1)

  def graph(self):
    plt.figure(figsize=(6, 4))
     
    t = np.arange(0, self.total_time, self.delta_t)
    for i in range(self.nb_processes):
            plt.plot(t, self.distances[i,:])
    plt.title((
    'Brownian motion process\n '
    'Position over time for 5 independent realizations'))
    plt.xlabel('$t$ (time)', fontsize=13)
    plt.ylabel('$d$ (position)', fontsize=13)
    plt.xlim([-0, 1])
    plt.tight_layout()
    plt.show()

class GaussianProcess:
  def __init__(self,nb_of_samples,number_of_functions):
        self.nb_of_samples = nb_of_samples
        self.number_of_functions = number_of_functions
  
  def collectData(self):
    self.X = np.expand_dims(np.linspace(-4, 4, self.nb_of_samples), 1)
    self.Σ = exponentiated_quadratic(self.X, self.X)  



    self.ys = np.random.multivariate_normal(
        mean=np.zeros(self.nb_of_samples), cov=self.Σ, 
        size=self.number_of_functions)

  def graph(self):
    plt.figure(figsize=(6, 4))
    for i in range(self.number_of_functions):
        plt.plot(self.X, self.ys[i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
        '5 different function realizations at 41 points\n'
        'sampled from a Gaussian process with exponentiated quadratic kernel'))
    plt.xlim([-4, 4])
    plt.show()

  def mean_variance(self, n1, n2,ny, domain):
    f_sin = lambda x: (np.sin(x)).flatten()

    X1 = np.random.uniform(domain[0]+2, domain[1]-2, size=(n1, 1))
    y1 = f_sin(X1)

    X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)

    μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)

    σ2 = np.sqrt(np.diag(Σ2))

    y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)

    fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(6, 6))

    ax1.plot(X2, f_sin(X2), 'b--', label='$sin(x)$')
    ax1.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='red', 
                    alpha=0.15, label='$2 \sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax1.legend()

    ax2.plot(X2, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax2.set_xlim([-6, 6])
    plt.tight_layout()
    plt.show()

  def noise_variance(self, noise):



    y1 = y1 + ((σ_noise ** 2) * np.random.randn(n1))


    μ2, Σ2 = GP_noise(X1, y1, X2, exponentiated_quadratic, σ_noise)

    σ2 = np.sqrt(np.diag(Σ2))


    y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)

   
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6))

    ax1.plot(X2, f_sin(X2), 'b--', label='$sin(x)$')
    ax1.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='red', 
                    alpha=0.15, label='$2\sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax1.legend()

    ax2.plot(X2, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax2.set_xlim([-6, 6])
    plt.tight_layout()
    plt.show()



 

n1 = 8  
n2 = 75  
ny = 5  
domain = (-6, 6)

b1 = BrownianMotion(1,75,5,0.)
b1.sim_1d()
b1.graph()

g1 =GaussianProcess(41,5)
g1.collectData()
g1.graph()

g1.mean_variance(n1,n2,ny,domain)

noise = 1.

g1.noise_variance(noise)