# ageny.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the author.
# 
# Author: Ioannis Karamouzas (ioannis@g.clemson.edu)
#

import numpy as np
from abc import ABC, abstractmethod

""" 
    Abstract class for agents
"""
class AbstractAgent(ABC):

    def __init__(self, inputParameters):
        """ 
            Takes an input line from the csv file,  
            and initializes the agent
        """
        self.id = int(inputParameters[0]) # the id of the agent
        self.gid = int(inputParameters[1]) # the group id of the agent
        self.pos = np.array([float(inputParameters[2]), float(inputParameters[3])]) # the position of the agent 
        self.vel = np.zeros(2) # the velocity of the agent
        self.goal = np.array([float(inputParameters[4]), float(inputParameters[5])]) # the goal of the agent
        self.prefspeed = float(inputParameters[6]) # the preferred speed of the agent
        self.gvel = self.goal-self.pos # the goal velocity of the agent
        self.gvel = self.gvel/(np.sqrt(self.gvel.dot(self.gvel )))*self.prefspeed       
        self.maxspeed = float(inputParameters[7]) # the maximum sped of the agent
        self.radius = float(inputParameters[8]) # the radius of the agent
        self.atGoal = False # has the agent reached its goal?
     
    @abstractmethod
    def computeAction(self, neighbors=[]):
        """
            Performs a sense and act simulation step.
        """
        pass

    @abstractmethod    
    def update(self, dt):
        """
            Updates the state of the character, given the time step of the simulation.
        """
        pass

""" 
    Agent class that implements the 2000 Social Force Model by Helbing et al.
    See the paper for details  
"""
class SFMAgent(AbstractAgent):

    def __init__(self, inputParameters, goalRadius=1, dhor = 10, ksi=0.5, A=2000, B=0.08, k=1.2e5, kappa=2.4e5, mass = 80, maxF = 10):
        """ 
           Initializes the agent
        """
        super().__init__(inputParameters)
        self.atGoal = False # has the agent reached its goal?
        self.goalRadiusSq = goalRadius*goalRadius # parameter to determine if agent is close to the goal
        self.dhor = dhor # the sensing radius
        self.ksi = ksi # the relaxation time used to compute the goal force
        self.A = A # scaling constant for the repulsive force
        self.B = B # safe distance that the agent prefers to keep
        self.k = k # scaling constant of the body force in case of a collision
        self.kappa = kappa # scaling constant of the sliding friction foce in case of a collision
        self.F = np.zeros(2) # the total force acting on the agent
        self.maxF = maxF # the maximum force that can be applied to the agent


    def computeAction(self, neighbors=[]):
        """ 
            Your code to compute the forces acting on the agent. 
        """       
        if not self.atGoal:
            self.F = np.zeros(2)
            

    def update(self, dt):
        """ 
            Code to update the velocity and position of the agents.  
            as well as determine the new goal velocity 
        """
        if not self.atGoal:
            self.vel += self.F*dt     # update the velocity
            self.pos += self.vel*dt   #update the position
        
            # compute the goal velocity for the next time step. Do not modify this
            self.gvel = self.goal - self.pos
            distGoalSq = self.gvel.dot(self.gvel)
            if distGoalSq < self.goalRadiusSq: 
                self.atGoal = True  # goal has been reached
            else: 
                self.gvel = self.gvel/np.sqrt(distGoalSq)*self.prefspeed  


""" 
    Agent class that implements the TTC force-based approach  
"""
class TTCAgent(AbstractAgent):

    def __init__(self, inputParameters, goalRadius=1, dhor = 10, ksi=0.5, timehor=5, epsilon=0, maxF = 10):
        """ 
           Initializes the agent
        """
        super().__init__(inputParameters)
        self.atGoal = False # has the agent reached its goal?
        self.goalRadiusSq = goalRadius * goalRadius # parameter to determine if agent is close to the goal
        self.dhor = dhor # the sensing radius
        self.ksi = ksi # the relaxation time used to compute the goal force
        self.timehor = timehor # the time horizon for computing avoidance forces
        self.epsilon = epsilon # the error in sensed velocities
        self.F = np.zeros(2) # the total force acting on the agent
        self.maxF = maxF # the maximum force that can be applied to the agent


    def computeAction(self, neighbors=[]):
        """ 
            Your code to compute the forces acting on the agent. 
        """       
        if not self.atGoal:
            self.F = np.zeros(2)
            

    def update(self, dt):
        """ 
            Code to update the velocity and position of the agents.  
            as well as determine the new goal velocity 
        """
        if not self.atGoal:
            self.vel += self.F*dt     # update the velocity
            self.pos += self.vel*dt   #update the position
        
            # compute the goal velocity for the next time step. Do not modify this
            self.gvel = self.goal - self.pos
            distGoalSq = self.gvel.dot(self.gvel)
            if distGoalSq < self.goalRadiusSq: 
                self.atGoal = True  # goal has been reached
            else: 
                self.gvel = self.gvel/np.sqrt(distGoalSq)*self.prefspeed  


""" 
    Agent class that implements a sampling-based VO approach  
"""
class VOAgent(AbstractAgent):

    def __init__(self, inputParameters, goalRadius=1, dhor = 10, epsilon=0):
        """ 
           Initializes the agent
        """
        super().__init__(inputParameters)
        self.atGoal = False # has the agent reached its goal?
        self.goalRadiusSq = goalRadius*goalRadius # parameter to determine if agent is close to the goal
        self.dhor = dhor # the sensing radius
        self.epsilon = epsilon # the error in sensed velocities
        self.vnew = np.zeros(2) # the new velocity of the agent  
   
    def computeAction(self, neighbors=[]):
        """ 
            Your code to compute the new velocity of the agent. 
            The code should set the vnew of the agent to be one of the sampled admissible one. Please do not directly set here the agent's velocity.   
        """       
        if not self.atGoal:
            self.vnew[:] = self.gvel[:]   # here I just set the new velocity to be the goal velocity   


    def update(self, dt):
        """ 
            Code to update the velocity and position of the agent  
            as well as determine the new goal velocity 
        """
        if not self.atGoal:
            self.vel[:] = self.vnew[:]
            self.pos += self.vel*dt   #update the position
        
            # compute the goal velocity for the next time step. Do not modify this
            self.gvel = self.goal - self.pos
            distGoalSq = self.gvel.dot(self.gvel)
            if distGoalSq < self.goalRadiusSq: 
                self.atGoal = True  # goal has been reached
            else: 
                self.gvel = self.gvel/np.sqrt(distGoalSq)*self.prefspeed  