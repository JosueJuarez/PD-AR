#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import gymnasium as gym

import time
#--BJ class
class BlackJackEnv:
    def __init__(self, episodes, gamma):
        self.EPISODES = episodes # k episodes

        self.rng = np.random.default_rng()
        
        self.env = gym.make('Blackjack-v1', sab=True).env
        
        self.STATES_n0 = self.env.observation_space[0].n #player current sum
        self.STATES_n1 = self.env.observation_space[1].n #dealer's visible card
        self.STATES_n2 = self.env.observation_space[2].n #if player has usable as
        self.ACTIONS_n = self.env.action_space.n #number of actions
    
        self.epsilon = 0.40 # for epsilon greedy search
        self.GAMMA = gamma # discount factor
    
        self.q_func = np.zeros((self.ACTIONS_n, self.STATES_n2, self.STATES_n0, self.STATES_n1)) #start Q-function as zeros

    def _eg_action(self, x, epsilon): #epislon-greedy action
        player_sum, dealer_card, player_as = x
        y = self.rng.random()
        
        if y >= epsilon:
            a = self.env.action_space.sample() # exploration
        else:
            a = self.q_func[:,player_as,player_sum,dealer_card].argmax() # greedy
            
        return a
    
    def _normal_reward(self, system_reward, terminated): #return env reward
        return system_reward

    def _custom_reward(self, system_reward, terminated): #give rewart to hit actions wich dosen't end the game
        if (system_reward == 0.0) and (terminated == False):
            return 0.5
        else:
            return system_reward
        
    #step size functions
    def _lognn(self, n):
        return (np.log(n))/(n)

    def _AB(self, n):
        return 5./(10. + n)
    
    # epsilon functions
    def _cnst_35(self, n):
        return 0.35
    
    def _cnst_50(self, n):
        return 0.50
    
    def _cnst_65(self, n):
        return 0.65
    
    def _epsilon_decay(self, n):
        return 1. - (1./n)
    
    def _epsilon_decay2(self, n):
        return 1./n
    
    def _epsilon_decay3(self, n):
        return 1./(1.+n)


    # Q-learning algorithm
    def q_learn(self, step_func, epsilon_func, reward_func):
        for k in range(self.EPISODES):
            n = 0
            xn, info = self.env.reset() # initial state
            player_sum, dealer_card, player_as = xn
            ep_finished = False
            
            while ep_finished == False:
                n += 1
                step = step_func(n) # alpha_n(xn,an)
                epsilon = epsilon_func(n)
                an = self._eg_action(xn,epsilon)
            
                s, ep_reward, ep_finished, ep_truncation, info = self.env.step(an)
                player_sum_s, dealer_card_s, player_as_s = s
                reward = reward_func(ep_reward,ep_finished)
                    
            
                Qn = self.q_func[an,player_as,player_sum,dealer_card] # Qn(xn,an)
                Wn = self.q_func[:,player_as_s,player_sum_s,dealer_card_s].max() # Wn(s)
                Qit = (1 - step)*Qn + step*(reward + self.GAMMA*Wn) #Q_n+1
            
                self.q_func[an,player_as,player_sum,dealer_card] = Qit
                xn = s
    
    def return_qfunc(self):
        return self.q_func
    
    def return_maxqfunc(self):
        return self.q_func.max(axis = 0)

    def return_drule(self):
        return self.q_func.argmax(axis = 0)
#--end class
    
def normal_reward(system_reward, terminated): #return env reward
    return system_reward

def custom_reward(system_reward, terminated): #give rewart to hit actions wich dosen't end the game
    if (system_reward == 0.0) and (terminated == False):
        return 0.5
    else:
        return system_reward


def env_simulation(actions, gamma, reward_func): # simulate enviorenment with given decision rule
    d_rule = actions
    env = gym.make('Blackjack-v1', sab=True).env

    SIMULATIONS = 10000 #number of simulations to perform
    GAMMA = gamma

    rewards = [] # rewards from simulations
    succes_count = 0
    
    for i in range(SIMULATIONS):
        xn, info = env.reset() #x0
        player_sum, dealer_card, player_as = xn
        
        ep_finished = False

        reward_sim = 0. #reawrd from current simulation
        k = 0

        while ep_finished == False:
            k += 1
            an = d_rule[player_as,player_sum,dealer_card]

            s, system_reward, ep_finished, episode_truncation, info = env.step(an)
            reward = reward_func(system_reward, ep_finished)
            reward_sim += (GAMMA**k)*reward #discounted

            xn = s
                
        rewards.append(reward_sim)
        
        if (ep_finished == True) and (system_reward == 1.0):
            succes_count += 1

    sim_rewards_mean = np.array(rewards).mean()
    sim_succes_rate = succes_count/SIMULATIONS

    return sim_rewards_mean, sim_succes_rate

episodes_list = np.linspace(100, 500000, num=500, endpoint=True, dtype=int)

#for dataframes
df_rewards_mean_nr_ce = [] 
df_succes_rate_nr_ce = []

df_rewards_mean_cr_ce = []
df_succes_rate_cr_ce = []

df_rewards_mean_nr_ed = [] 
df_succes_rate_nr_ed = []

df_rewards_mean_cr_ed = []
df_succes_rate_cr_ed = []

print("-----start learning and simulations-----")

start_time = time.time()

for episode in episodes_list:
    print("current episode: ", episode)

    #normal rewards and constant epsilons (gamma = 1)
    #step = log(n)/n
    blackjack_lognn_cnst_35 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_lognn_cnst_35.q_learn(blackjack_lognn_cnst_35._lognn, 
                                    blackjack_lognn_cnst_35._cnst_35, 
                                    blackjack_lognn_cnst_35._normal_reward) # generate Q-function approximation
    mean_cnst_35, scc_rate_cnst_35 = env_simulation(blackjack_lognn_cnst_35.return_drule(), 1., normal_reward) # simulatte enviorenment following decision rule

    blackjack_lognn_cnst_50 = BlackJackEnv(episode, 1.)
    blackjack_lognn_cnst_50.q_learn(blackjack_lognn_cnst_50._lognn, 
                                    blackjack_lognn_cnst_50._cnst_50, 
                                    blackjack_lognn_cnst_50._normal_reward)
    mean_cnst_50, scc_rate_cnst_50 = env_simulation(blackjack_lognn_cnst_50.return_drule(), 1., normal_reward)

    blackjack_lognn_cnst_65 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_lognn_cnst_65.q_learn(blackjack_lognn_cnst_65._lognn, 
                                    blackjack_lognn_cnst_65._cnst_65, 
                                    blackjack_lognn_cnst_65._normal_reward) # generate Q-function approximation
    mean_cnst_65, scc_rate_cnst_65 = env_simulation(blackjack_lognn_cnst_65.return_drule(), 1., normal_reward)

    df_rewards_mean_nr_ce.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "35" : mean_cnst_35, "50" : mean_cnst_50, "65" : mean_cnst_65})
    df_succes_rate_nr_ce.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "35" : scc_rate_cnst_35, "50" : scc_rate_cnst_50, "65" : scc_rate_cnst_65})

    #step = A/(B+n)
    blackjack_AB_cnst_35 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_AB_cnst_35.q_learn(blackjack_AB_cnst_35._AB, 
                                    blackjack_AB_cnst_35._cnst_35, 
                                    blackjack_AB_cnst_35._normal_reward) # generate Q-function approximation
    mean_cnst_35, scc_rate_cnst_35 = env_simulation(blackjack_AB_cnst_35.return_drule(), 1., normal_reward)

    blackjack_AB_cnst_50 = BlackJackEnv(episode,1.)
    blackjack_AB_cnst_50.q_learn(blackjack_AB_cnst_50._AB, 
                                    blackjack_AB_cnst_50._cnst_50, 
                                    blackjack_AB_cnst_50._normal_reward)
    mean_cnst_50, scc_rate_cnst_50 = env_simulation(blackjack_AB_cnst_50.return_drule(), 1., normal_reward)

    blackjack_AB_cnst_65 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_AB_cnst_65.q_learn(blackjack_AB_cnst_65._AB, 
                                    blackjack_AB_cnst_65._cnst_65, 
                                    blackjack_AB_cnst_65._normal_reward) # generate Q-function approximation
    mean_cnst_65, scc_rate_cnst_65 = env_simulation(blackjack_AB_cnst_65.return_drule(), 1., normal_reward)

    df_rewards_mean_nr_ce.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "35" : mean_cnst_35, "50" : mean_cnst_50, "65" : mean_cnst_65})
    df_succes_rate_nr_ce.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "35" : scc_rate_cnst_35, "50" : scc_rate_cnst_50, "65" : scc_rate_cnst_65})

    #custom rewards and constant epsilons (gamma = 0.95)
    #step = log(n)/n
    blackjack_lognn_cnst_35 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_lognn_cnst_35.q_learn(blackjack_lognn_cnst_35._lognn, 
                                    blackjack_lognn_cnst_35._cnst_35, 
                                    blackjack_lognn_cnst_35._custom_reward) # generate Q-function approximation
    mean_cnst_35, scc_rate_cnst_35 = env_simulation(blackjack_lognn_cnst_35.return_drule(), 0.95, custom_reward)

    blackjack_lognn_cnst_50 = BlackJackEnv(episode, 0.95)
    blackjack_lognn_cnst_50.q_learn(blackjack_lognn_cnst_50._lognn, 
                                    blackjack_lognn_cnst_50._cnst_50, 
                                    blackjack_lognn_cnst_50._custom_reward)
    mean_cnst_50, scc_rate_cnst_50 = env_simulation(blackjack_lognn_cnst_50.return_drule(), 0.95, custom_reward)

    blackjack_lognn_cnst_65 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_lognn_cnst_65.q_learn(blackjack_lognn_cnst_65._lognn, 
                                    blackjack_lognn_cnst_65._cnst_65, 
                                    blackjack_lognn_cnst_65._custom_reward) # generate Q-function approximation
    mean_cnst_65, scc_rate_cnst_65 = env_simulation(blackjack_lognn_cnst_65.return_drule(), 0.95, custom_reward)

    df_rewards_mean_cr_ce.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "35" : mean_cnst_35, "50" : mean_cnst_50, "65" : mean_cnst_65})
    df_succes_rate_cr_ce.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "35" : scc_rate_cnst_35, "50" : scc_rate_cnst_50, "65" : scc_rate_cnst_65})

    #step = A/(B+n)
    blackjack_AB_cnst_35 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_AB_cnst_35.q_learn(blackjack_AB_cnst_35._AB, 
                                    blackjack_AB_cnst_35._cnst_35, 
                                    blackjack_AB_cnst_35._custom_reward) # generate Q-function approximation
    mean_cnst_35, scc_rate_cnst_35 = env_simulation(blackjack_AB_cnst_35.return_drule(), 0.95, custom_reward)

    blackjack_AB_cnst_50 = BlackJackEnv(episode, 0.95)
    blackjack_AB_cnst_50.q_learn(blackjack_AB_cnst_50._AB, 
                                    blackjack_AB_cnst_50._cnst_50, 
                                    blackjack_AB_cnst_50._custom_reward)
    mean_cnst_50, scc_rate_cnst_50 = env_simulation(blackjack_AB_cnst_50.return_drule(), 0.95, custom_reward)

    blackjack_AB_cnst_65 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_AB_cnst_65.q_learn(blackjack_AB_cnst_65._AB, 
                                    blackjack_AB_cnst_65._cnst_65, 
                                    blackjack_AB_cnst_65._custom_reward) # generate Q-function approximation
    mean_cnst_65, scc_rate_cnst_65 = env_simulation(blackjack_AB_cnst_65.return_drule(), 0.95, custom_reward)

    df_rewards_mean_cr_ce.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "35" : mean_cnst_35, "50" : mean_cnst_50, "65" : mean_cnst_65})
    df_succes_rate_cr_ce.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "35" : scc_rate_cnst_35, "50" : scc_rate_cnst_50, "65" : scc_rate_cnst_65})
    
    #normal rewards and epsilon decay (gamma = 1)
    #step = log(n)/n
    blackjack_lognn_ed = BlackJackEnv(episode, 1.) #enviorement
    blackjack_lognn_ed.q_learn(blackjack_lognn_ed._lognn, 
                                    blackjack_lognn_ed._epsilon_decay, 
                                    blackjack_lognn_ed._normal_reward) # generate Q-function approximation
    mean_ed, scc_rate_ed = env_simulation(blackjack_lognn_ed.return_drule(), 1., normal_reward) # simulatte enviorenment following decision rule

    blackjack_lognn_ed2 = BlackJackEnv(episode, 1.)
    blackjack_lognn_ed2.q_learn(blackjack_lognn_ed2._lognn, 
                                    blackjack_lognn_ed2._epsilon_decay2, 
                                    blackjack_lognn_ed2._normal_reward)
    mean_ed2, scc_rate_ed2 = env_simulation(blackjack_lognn_ed2.return_drule(), 1., normal_reward)

    blackjack_lognn_ed3 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_lognn_ed3.q_learn(blackjack_lognn_ed3._lognn, 
                                    blackjack_lognn_ed3._epsilon_decay3, 
                                    blackjack_lognn_ed3._normal_reward) # generate Q-function approximation
    mean_ed3, scc_rate_ed3 = env_simulation(blackjack_lognn_ed3.return_drule(), 1., normal_reward)

    df_rewards_mean_nr_ed.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "ed" : mean_ed, "ed2" : mean_ed2, "ed3" : mean_ed3})
    df_succes_rate_nr_ed.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "ed" : scc_rate_ed, "ed2" : scc_rate_ed2, "ed3" : scc_rate_ed3})

    #step = A/(B+n)
    blackjack_AB_ed = BlackJackEnv(episode, 1.) #enviorement
    blackjack_AB_ed.q_learn(blackjack_AB_ed._AB, 
                                    blackjack_AB_ed._epsilon_decay, 
                                    blackjack_AB_ed._normal_reward) # generate Q-function approximation
    mean_ed, scc_rate_ed = env_simulation(blackjack_AB_ed.return_drule(), 1., normal_reward)

    blackjack_AB_ed2 = BlackJackEnv(episode,1.)
    blackjack_AB_ed2.q_learn(blackjack_AB_ed2._AB, 
                                    blackjack_AB_ed2._epsilon_decay2, 
                                    blackjack_AB_ed2._normal_reward)
    mean_ed2, scc_rate_ed2 = env_simulation(blackjack_AB_ed2.return_drule(), 1., normal_reward)

    blackjack_AB_ed3 = BlackJackEnv(episode, 1.) #enviorement
    blackjack_AB_ed3.q_learn(blackjack_AB_ed3._AB, 
                                    blackjack_AB_ed3._epsilon_decay3, 
                                    blackjack_AB_ed3._normal_reward) # generate Q-function approximation
    mean_ed3, scc_rate_ed3 = env_simulation(blackjack_AB_ed3.return_drule(), 1., normal_reward)

    df_rewards_mean_nr_ed.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "ed" : mean_ed, "ed2" : mean_ed2, "ed3" : mean_ed3})
    df_succes_rate_nr_ed.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "ed" : scc_rate_ed, "ed2" : scc_rate_ed2, "ed3" : scc_rate_ed3})

    #custom rewards and epsilon decay (gamma = 0.95)
    #step = log(n)/n
    blackjack_lognn_ed = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_lognn_ed.q_learn(blackjack_lognn_ed._lognn, 
                                    blackjack_lognn_ed._epsilon_decay, 
                                    blackjack_lognn_ed._custom_reward) # generate Q-function approximation
    mean_ed, scc_rate_ed = env_simulation(blackjack_lognn_ed.return_drule(), 0.95, custom_reward)

    blackjack_lognn_ed2 = BlackJackEnv(episode, 0.95)
    blackjack_lognn_ed2.q_learn(blackjack_lognn_ed2._lognn, 
                                    blackjack_lognn_ed2._epsilon_decay2, 
                                    blackjack_lognn_ed2._custom_reward)
    mean_ed2, scc_rate_ed2 = env_simulation(blackjack_lognn_ed2.return_drule(), 0.95, custom_reward)

    blackjack_lognn_ed3 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_lognn_ed3.q_learn(blackjack_lognn_ed3._lognn, 
                                    blackjack_lognn_ed3._epsilon_decay3, 
                                    blackjack_lognn_ed3._custom_reward) # generate Q-function approximation
    mean_ed3, scc_rate_ed3 = env_simulation(blackjack_lognn_ed3.return_drule(), 0.95, custom_reward)

    df_rewards_mean_cr_ed.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "ed" : mean_ed, "ed2" : mean_ed2, "ed3" : mean_ed3})
    df_succes_rate_cr_ed.append({"Episodio" : episode, "Función paso" : "lognn", 
                                  "ed" : scc_rate_ed, "ed2" : scc_rate_ed2, "ed3" : scc_rate_ed3})

    #step = A/(B+n)
    blackjack_AB_ed = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_AB_ed.q_learn(blackjack_AB_ed._AB, 
                                    blackjack_AB_ed._epsilon_decay, 
                                    blackjack_AB_ed._custom_reward) # generate Q-function approximation
    mean_ed, scc_rate_ed = env_simulation(blackjack_AB_ed.return_drule(), 0.95, custom_reward)

    blackjack_AB_ed2 = BlackJackEnv(episode, 0.95)
    blackjack_AB_ed2.q_learn(blackjack_AB_ed2._AB, 
                                    blackjack_AB_ed2._epsilon_decay2, 
                                    blackjack_AB_ed2._custom_reward)
    mean_ed2, scc_rate_ed2 = env_simulation(blackjack_AB_ed2.return_drule(), 0.95, custom_reward)

    blackjack_AB_ed3 = BlackJackEnv(episode, 0.95) #enviorement
    blackjack_AB_ed3.q_learn(blackjack_AB_ed3._AB, 
                                    blackjack_AB_ed3._epsilon_decay3, 
                                    blackjack_AB_ed3._custom_reward) # generate Q-function approximation
    mean_ed3, scc_rate_ed3 = env_simulation(blackjack_AB_ed3.return_drule(), 0.95, custom_reward)

    df_rewards_mean_cr_ed.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "ed" : mean_ed, "ed2" : mean_ed2, "ed3" : mean_ed3})
    df_succes_rate_cr_ed.append({"Episodio" : episode, "Función paso" : "AB", 
                                  "ed" : scc_rate_ed, "ed2" : scc_rate_ed2, "ed3" : scc_rate_ed3}) #THERE WAS A MISTAKE HERE, DATA ADDED TO WRONG DATAFRAME (nr_ed). ALREADY FIXED

print("time running: ", time.time() - start_time) #time needed for running the script: 177043.26940441132 seg.

print("-----end-----")

df_rewards_mean_nr_ce = pd.DataFrame(df_rewards_mean_nr_ce)
df_succes_rate_nr_ce = pd.DataFrame(df_succes_rate_nr_ce)

df_rewards_mean_cr_ce = pd.DataFrame(df_rewards_mean_cr_ce)
df_succes_rate_cr_ce = pd.DataFrame(df_succes_rate_cr_ce)

df_rewards_mean_nr_ed = pd.DataFrame(df_rewards_mean_nr_ed)
df_succes_rate_nr_ed = pd.DataFrame(df_succes_rate_nr_ed)

df_rewards_mean_cr_ed = pd.DataFrame(df_rewards_mean_cr_ed)
df_succes_rate_cr_ed = pd.DataFrame(df_succes_rate_cr_ed)

df_rewards_mean_nr_ce.to_csv("df_rewards_mean_nr_ce.csv")
df_succes_rate_nr_ce.to_csv("df_succes_rate_nr_ce.csv")

df_rewards_mean_cr_ce.to_csv("df_rewards_mean_cr_ce.csv")
df_succes_rate_cr_ce.to_csv("df_succes_rate_cr_ce.csv")

df_rewards_mean_nr_ed.to_csv("df_rewards_mean_nr_ed.csv")
df_succes_rate_nr_ed.to_csv("df_succes_rate_nr_ed.csv")

df_rewards_mean_cr_ed.to_csv("df_rewards_mean_cr_ed.csv")
df_succes_rate_cr_ed.to_csv("df_succes_rate_cr_ed.csv")