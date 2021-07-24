import random
import math
import statistics
import matplotlib.pyplot as plt


class PoissonProcess:
  def __init__(self,lambda_, num_events, event_time):
      self.lambda_ = lambda_
      self.num_events = num_events
      self.event_time = event_time
      self.event_num = []
      self.inter_event_times = []
      self.event_times = []
      self.interval_nums = []
      self.num_events_in_interval = []
      self.interval_num = 1
      
      
   

  def simulate(self):
      for i in range(self.num_events):
          self.event_num.append(i)
          n = random.random()

          #inverse of CDF of exponential distribution
          int_arr_time = -math.log(1.0 - n)/self.lambda_
          self.inter_event_times.append(int_arr_time)

          self.event_time = self.event_time + int_arr_time
          self.event_times.append(self.event_time)

          print(str(n)+','+str(int_arr_time)+','+str(self.event_time))
  
  def graphInterEvents(self):
    fig = plt.figure()
    fig.suptitle('Times between consecutive events in a simulated Poisson process')
    plot, = plt.plot(self.event_num, self.inter_event_times, 'bo-', label='Inter-event time')
    plt.legend(handles=[plot])
    plt.xlabel('Index of event')
    plt.ylabel('Time')
    plt.show()

  def graphAbosoluteEvents(self):
    fig = plt.figure()
    fig.suptitle('Absolute times of consecutive events in a simulated Poisson process')
    plot, = plt.plot(self.event_num, self.event_times, 'bo-', label='Absolute time of event')
    plt.legend(handles=[plot])
    plt.xlabel('Index of event')
    plt.ylabel('Time')
    plt.show()

  def consecutiveIntervals(self):
    self.num_events = 0


    for i in range(len(self.event_times)):
	    self.event_time = self.event_times[i]
	    if self.event_time <= self.interval_num:
		    self.num_events += 1
	    else:
		    self.interval_nums.append(self.interval_num)
		    self.num_events_in_interval.append(self.num_events)

		    print(str(self.interval_num) +',' + str(self.num_events))

		    self.interval_num += 1

		    self.num_events = 1


    print(statistics.mean(self.num_events_in_interval))


    fig = plt.figure()
    fig.suptitle('Number of events occurring in consecutive intervals in a simulated Poisson process')
    plt.bar(self.interval_nums, self.num_events_in_interval)
    plt.xlabel('Index of interval')
    plt.ylabel('Number of events')
    plt.show()



  



      

p1 = PoissonProcess( 5,100,0)
p1.simulate()
p1.graphInterEvents()
p1.graphAbosoluteEvents()
p1.consecutiveIntervals()

