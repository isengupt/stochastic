import numpy as np
import matplotlib.pyplot as plt


class GeometricBrownian():
    def __init__(self,x0):
        self.x0 = float(x0)
        self.nstep = 100
    
    def rand_walk(self, n_step=100):

        #random walk generation

        walk = np.ones(n_step)*self.x0

        for i in range(1, n_step):

            y1 = np.random.choice([1,-1])

            #weiner process
            walk[i] = (walk[i-1] + (y1/np.sqrt(n_step)))

        return walk


    def norm_motion(self,n_step=100):

        w= np.ones(self.n_step) * self.x0

        for i in range(1, self.n_step):

            y1 = np.random.normal()

            w[i] = w[i-1] + (y1/np.sqrt(n_step))

        return w

    def stock_price(
        self,
        s0=100,
        mu=0.2,
        sigma=0.68,
        deltaT=52,
        dt=0.1
    ):

        n_step = int(deltaT/dt)
        t_vector = np.linspace(0,deltaT, num=n_step)

        stock_Var = (mu-(sigma**2/2))*t_vector

        self.x0 = 0

        weiner_prcs = sigma*self.norm_motion(n_step)
        s = s0*(np*exp(stock_var+weiner_prcs))

        return s

b1 = GeometricBrownian(0)

        
for i in range(4):
    plt.plot(b1.rand_walk(1000))
plt.show()




