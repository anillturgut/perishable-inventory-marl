import pandas as pd
from environment import CompetitivePerishableInventoryPlanning
from utils import combinations, interpolate
import numpy as np
import nashpy as nash
import utils

m = 3  # product life

# Initialize the environment
initial_state = [0] * m
env_dummy = CompetitivePerishableInventoryPlanning(initial_state=(initial_state, initial_state))

# State index generation
max_x = env_dummy.max_x # maximum order quantity
states = []
for comb1 in combinations(m, max_x):
    for comb2 in combinations(m, max_x):
        states.append((comb1, comb2))

state_index = pd.DataFrame(pd.Series(tuple(states)), columns=["state"])
state_index["inventory_level"] = state_index["state"].apply(lambda x: (sum(x[0]), sum(x[1])))


def negate_nashq_matrices(Q1, Q2):
      BigM = 10**10
      Q1_negated = Q1.copy()
      Q1_negated[Q1_negated == 0] = BigM
      Q2_negated = Q2.copy()
      Q2_negated[Q2_negated == 0] = BigM

      return Q1_negated, Q2_negated

def getNextReplenishmentDecision(Q1, Q2, state, day):
    x1 = 0
    St1, St2 = state
    s_ind = state_index.loc[state_index["state"].apply(lambda x: np.array_equal(x, state))].index[0]
    q_state1 = Q1[s_ind, day, :, :]
    q_state2 = Q2[s_ind, day, :, :]
    game = nash.Game(q_state1,q_state2)
    equilibriums = list(game.support_enumeration())
    best_payoff = -np.Inf
    for eq in equilibriums:
        if len(np.where(eq[0] == 1)[0]) != 0: #The equilibrium needs to be a strict nash equilibrium (no mixed-strategy)
            total_payoff = q_state1[np.where(eq[0]==1)[0][0]][np.where(eq[1]==1)[0][0]] + q_state2[np.where(eq[0]==1)[0][0]][np.where(eq[1]==1)[0][0]]
            if (total_payoff < 0) and (total_payoff >= best_payoff):
                best_payoff = total_payoff
                x1 = np.where(eq[0] == 1)[0][0]
    return x1
