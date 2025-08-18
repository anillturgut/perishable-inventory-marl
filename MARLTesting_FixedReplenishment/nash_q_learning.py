# nash_q_learning.py

import numpy as np
from scipy.optimize import linprog
import nashpy as nash
import datetime
from tqdm import tqdm
import random
import pandas as pd
import os

def negate_nashq_matrices(Q1, Q2):
      BigM = 10**4
      Q1_negated = Q1.copy()
      Q1_negated[Q1_negated == 0] = BigM
      Q2_negated = Q2.copy()
      Q2_negated[Q2_negated == 0] = BigM

      return -Q1_negated, -Q2_negated

def find_nash_equilibrium(Q1, Q2):
    Q1_negated, Q2_negated = negate_nashq_matrices(Q1, Q2)
    game = nash.Game(Q1_negated, Q2_negated)
    equilibria = list(game.support_enumeration())
    equilibrium_values = []
    min_payoff = -np.inf
    if equilibria:
        selected_eq = random.choice(equilibria)
        for eq in equilibria:
            payoff =  game[eq][0] + game[eq][1]
            if (payoff >= min_payoff):
                selected_eq = eq
                min_payoff = payoff
                equilibrium_values = game[eq]
        return selected_eq, equilibrium_values
    else: 
        return None, [0, 0]

def update_winner_info(df, epoch, cost_1, cost_2, winner):
    # Create a new row with the game information
    new_row = pd.DataFrame({
        'Epoch': [epoch],
        'Cost_Pharmacy1': [cost_1],
        'Cost_Pharmacy2': [cost_2],
        'Winner_Pharmacy': [winner]
    })

    df = pd.concat([df, new_row], ignore_index=True)
    
    return df

def nash_q_learning(env, state_index, initial_alpha=0.5, gamma=0.7, N=10, epsilon=0.5, strategy="greedy", method="q-learning", mode="debug"):
    Q1 = np.zeros([len(state_index), env.T + 1, env.action_space.n, env.action_space.n])
    Q2 = np.zeros([len(state_index), env.T + 1, env.action_space.n, env.action_space.n])
    Q1[:, env.T, :, :] = 1000
    Q2[:, env.T, :, :] = 1000
    
    decay_rate = 0.999994

    winner_df = pd.DataFrame(columns=['Epoch', 'Cost_Pharmacy1', 'Cost_Pharmacy2', 'Winner_Pharmacy'])

    t = 0
    t_info = []
    start_time = datetime.datetime.now()
    print(initial_alpha)
    save_dir = 'FixedTrainedAgents_05Greedy_010825M2T6'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in tqdm(range(N)):
        #print(f"Epoch : {(i + 1)}")
        env._reset()
        done = False
        alpha = initial_alpha * (decay_rate ** i)
        # print("Alpha", str(alpha))
        while not done:
            if i == 0 and env.t == 0:
                s = env.initial_state
                s_ind = state_index.loc[state_index["state"].apply(lambda x: np.array_equal(x, s))].index[0]
                x1 = np.random.randint(0, env.action_space.n)
                x2 = 3
            else:
                if t == env.T - 1:
                    s_ind = s1_ind
                    t = 0
                else:
                    s_ind = s1_ind
                    t = env.t

            if strategy == 'random':
                x1 = np.random.randint(0, env.action_space.n) #Random move
                x2 = 3
            if strategy == 'greedy':
                greedy_matrix1 = Q1[s_ind, t + 1, :, 3].reshape(-1, 1)  # Fix x2 = 3 for Agent 2
                greedy_matrix2 = Q2[s_ind, t + 1, :, 3].reshape(-1, 1)
                greedy_matrix1_negated, greedy_matrix2_negated = negate_nashq_matrices(greedy_matrix1, greedy_matrix2)
                greedy_game =  nash.Game(greedy_matrix1_negated,greedy_matrix2_negated)
                equilibriums = list(greedy_game.support_enumeration())
                greedy_equilibrium = equilibriums[random.randrange(len(equilibriums))] #One random equilibrium
                if len(np.where(greedy_equilibrium[0] == 1)[0]) == 0: #No strict equilibrium found
                        x1 = np.random.randint(0, env.action_space.n) #Random move
                        x2 = 3
                else:  #Select the movements corresponding to the nash equilibrium
                        x1 = np.where(greedy_equilibrium[0] == 1)[0][0]
                        x2 = 3    
            if strategy == 'epsilon-greedy':
                random_number = np.random.uniform(0,1)
                if random_number >= epsilon: #greedy
                    #greedy_matrix1 = Q1[s_ind, t + 1, :, :]
                    #greedy_matrix2 = Q2[s_ind, t + 1, :, :]
                    greedy_matrix1 = Q1[s_ind, t + 1, :, 3].reshape(-1, 1)  # Fix x2 = 3 for Agent 2
                    greedy_matrix2 = Q2[s_ind, t + 1, :, 3].reshape(-1, 1)
                    greedy_matrix1_negated, greedy_matrix2_negated = negate_nashq_matrices(greedy_matrix1, greedy_matrix2)
                    greedy_game =  nash.Game(greedy_matrix1_negated,greedy_matrix2_negated)
                    equilibriums = list(greedy_game.support_enumeration())
                    greedy_equilibrium = equilibriums[random.randrange(len(equilibriums))] #One random equilibrium
                    if len(np.where(greedy_equilibrium[0] == 1)[0]) == 0: #No strict equilibrium found
                        x1 = np.random.randint(0, env.action_space.n) #Random move
                        x2 = 3
                    else:  #Select the movements corresponding to the nash equilibrium
                        x1 = np.where(greedy_equilibrium[0] == 1)[0][0]
                        x2 = 3        
                else: #random
                    x1 = np.random.randint(0, env.action_space.n) #Random move
                    x2 = 3

            if strategy == "limiting-greedy":
                exploration = 1-epsilon/(i+1)
                rand = np.random.uniform(0, 1)
                if rand < exploration:
                    x1 = np.argmin(Q1[s_ind,t,:,x2])
                else:
                    x1 = np.random.randint(0, env.action_space.n-1)
                x2 = 3  # Fixed action for inventory 2
                exploration = 1-exploration

            t, s, s1, r, done, info = env._step(x1, x2)
            s1_ind = state_index.loc[state_index["state"].apply(lambda x: np.array_equal(x, s1))].index[0]

            # nash_eq_matrix1 = Q1[s1_ind, t + 1, :, :]
            # nash_eq_matrix2 = Q2[s1_ind, t + 1, :, :]

            nash_eq_matrix1 = Q1[s1_ind, t + 1, :, 3].reshape(-1, 1)
            nash_eq_matrix2 = Q2[s1_ind, t + 1, :, 3].reshape(-1, 1)

            equilibrium, equilibrium_values = find_nash_equilibrium(nash_eq_matrix1, nash_eq_matrix2)
            if equilibrium:
                sigma_1, sigma_2 = equilibrium
                V1 = sum(sigma_1[a] * Q1[s1_ind, t + 1, a, b] for a in range(env.action_space.n) for b in range(env.action_space.n))
                # V2 = sum(sigma_2[b] * Q2[s1_ind, t + 1, a, b] for a in range(env.action_space.n) for b in range(env.action_space.n))
                V2 = Q2[s1_ind, t + 1, :, 3].min() 
            else:
                V1 = Q1[s1_ind, t + 1].min()
                V2 = Q2[s1_ind, t + 1].min()

            #print("x1,x2:",x1,x2)
            Q1[s_ind, t, x1, x2] = (1 - alpha) * Q1[s_ind, t, x1, x2] + alpha * (r[0] + gamma * -equilibrium_values[0])
            Q2[s_ind, t, x1, x2] = (1 - alpha) * Q2[s_ind, t, x1, x2] + alpha * (r[1] + gamma * -equilibrium_values[1])

            if mode == "debug":
                info["episode"] = i + 1
                info["s_ind"] = s_ind
                info["s1_ind"] = s1_ind
                info["q1_value"] = Q1[s_ind, t, x1, x2]
                info["q2_value"] = Q2[s_ind, t, x1, x2]
                info["q1_value1"] = Q1[s1_ind, t + 1, :, :]
                info["q2_value1"] = Q2[s1_ind, t + 1, :, :]
                info["reward"] = r
                t_info.append(info)
            #print(info)

        winner_pharmacy, c1, c2 = env.get_winner()
        winner_df = update_winner_info(winner_df, (i+1), c1, c2, winner_pharmacy)

        if (i + 1) % 5000 == 0:
            level = int((i + 1) / 1000)
            # Save the Q1 and Q2 agents
            Q1_filename = os.path.join(save_dir, f"Q1_new_max_{level}K_X{env.max_x}.pkl")
            Q2_filename = os.path.join(save_dir, f"Q2_new_max_{level}K_X{env.max_x}.pkl")
            
            pd.to_pickle(Q1, Q1_filename)
            pd.to_pickle(Q2, Q2_filename)
            
            print(f"Agents saved at epoch {i+1}: {Q1_filename}, {Q2_filename}")
        #print("Epoch:",str(i+1),"--Winner:",winner_pharmacy,"--C1",str(c1),"--C2",str(c2))

    total_time = datetime.datetime.now() - start_time
    print(total_time)

    return Q1, Q2, t_info, winner_df