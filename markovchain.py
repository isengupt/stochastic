import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as plt

def simulate_multinomial(vmultinomial):
  r=np.random.uniform(0.0, 1.0)
  CS=np.cumsum(vmultinomial)
  CS=np.insert(CS,0,0)
  m=(np.where(CS<r))[0]
  nextState=m[len(m)-1]
  return nextState

class MarkovChain:
  def __init__(self,mar_matrix, m_state, distr_hist, stateChangeHist):
      self.mar_matrix = mar_matrix
      self.m_state = m_state
      self.stateHist = m_state
      self.dfStateHist = pd.DataFrame(m_state)
      self.distr_hist = distr_hist
      self.currentState=0
      self.stateChangeHist = stateChangeHist

  def stationary_distribution(self, range_val):
      for x in range(range_val):
        self.m_state = np.dot(self.m_state, self.mar_matrix)
        print(self.m_state)
        self.stateHist = np.append(self.stateHist, self.m_state, axis=0)
        self.dfDistrHist = pd.DataFrame(self.stateHist)
        #self.dfDistrHist.plot()
        
      plt.show()
  




      

mar_matrix = np.array([[0.2, 0.7, 0.1],
              [0.9, 0.0, 0.1],
              [0.2, 0.8, 0.0]])
stateChangeHist= np.array([[0.0,  0.0,  0.0],
                          [0.0, 0.0,  0.0],
                          [0.0, 0.0,  0.0]])
m_state = np.array([[1.0,0.0,0.0]])

distr_hist = [[0,0,0]]
seed(4)
      
m1 = MarkovChain(mar_matrix, m_state, distr_hist, stateChangeHist )

#m1.stationary_distribution(50)
#m1.sim_multinomial(1000)



      


