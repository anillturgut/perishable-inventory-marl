import pandas as pd
from environment import CompetitivePerishableInventoryPlanning
from nash_q_learning import nash_q_learning
from utils import combinations, interpolate
import numpy as np
import nashpy as nash
import pickle

# State index generation
m = 2 # product life
max_x = 5  # maximum order quantity
states = []
for comb1 in combinations(m, max_x):
    for comb2 in combinations(m, max_x):
        states.append((comb1, comb2))

# Assign unique identifier to all states
state_index = pd.DataFrame(pd.Series(tuple(states)), columns=["state"])
state_index["inventory_level"] = state_index["state"].apply(lambda x: (sum(x[0]), sum(x[1])))


initial_state = [0, 0]
env = CompetitivePerishableInventoryPlanning(maximum_order_quantity= max_x, planning_horizon = 6, 
                                             product_life=m , initial_state=(initial_state, initial_state))

import warnings
warnings.filterwarnings('ignore')
# Run Nash Q-learning
initial_alpha = 0.5
Q1, Q2, t_info, winner_df = nash_q_learning(env, state_index, initial_alpha=initial_alpha, N=300000)


path = '/Users/anilturgut/Desktop/CompetitiveFixedNQL/FixedTrainedAgents081224M2T6'


winner_df.to_excel(path + "/winner_result.xlsx")