import random
import math
import numpy as np
from gym import Env, spaces
from gym.utils import seeding
import utils as utils

class CompetitivePerishableInventoryPlanning(Env):
    def __init__(self, planning_horizon=6, product_life=2, maximum_order_quantity=5,
                 ordering_cost=1, holding_cost=2, shortage_cost=12, deterioration_cost=15,
                 initial_state=None, lambd=3, max_x=5):
        self.T = planning_horizon
        self.X = maximum_order_quantity
        self.m = product_life
        self.co = ordering_cost
        self.ch = holding_cost
        self.cs = shortage_cost
        self.cd = deterioration_cost
        self.l = lambd
        self.max_x = max_x
        self.chosen_inventory = 0
        self.C1 = 0
        self.C2 = 0

        self.action_space = spaces.Discrete(maximum_order_quantity + 1)
        self.t = 0
        self.initial_state = initial_state
        self.current_state = initial_state

        self._seed()
        self._reset()

    def demand(self):
        return min(self.max_x, np.random.poisson(self.l))

    def transition(self, fulfillment_metric, St1, St2, xt1, xt2, Dt):
        selected_pharmacy_index = utils.choosePharmacy(fulfillment_metric)
        if selected_pharmacy_index == 1:
            St1_1 = self.transition_single(St1, xt1, Dt)
            St2_1 = self.transition_single(St2, xt2, 0)
        else:
            St1_1 = self.transition_single(St1, xt1, 0)
            St2_1 = self.transition_single(St2, xt2, Dt)
        return St1_1, St2_1

    def transition_single(self, St, xt, Dt):
        yt_begin = sum(St)
        St_1 = np.zeros(self.m)
        remaining_demand = Dt

        for j in range(self.m):
            if remaining_demand >= St[j]:
                St_1[j] = 0
                remaining_demand -= St[j]
            else:
                St_1[j] = St[j] - remaining_demand
                remaining_demand = 0

        for j in range(1, self.m):
            St_1[j - 1] = St_1[j]
        St_1[-1] = max(0, xt - max(0, Dt - yt_begin))
        return St_1

    def reward(self, fulfillment_metric, St1, St2, xt1, xt2, Dt):
        selected_pharmacy_index = utils.choosePharmacy(fulfillment_metric)
        if selected_pharmacy_index == 1:
            r1 = self.reward_single(St1, xt1, Dt, True)
            r2 = self.reward_single(St2, xt2, Dt, False)
            self.chosen_inventory = "Agent"
        else:
            r1 = self.reward_single(St1, xt1, Dt, False)
            r2 = self.reward_single(St2, xt2, Dt, True)
            self.chosen_inventory = "User"
        return r1, r2

    def reward_single(self, St, xt, Dt, is_chosen):
        yt = sum(St) + xt
        if is_chosen:
            r = self.co * xt + self.ch * max(0, yt - Dt) + self.cs * max(0, Dt - yt) + self.cd * max(0, St[0])
        else:
            r = self.co * xt + self.ch * max(0, yt) + self.cs * max(0, Dt) + self.cd * max(0, St[0])
        return r

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, x1, x2):
        St1, St2 = self.current_state
        Dt = self.demand()
        fulfillment_metric = utils.calculate_fulfillment_metric(St1, St2)
        reward1, reward2 = self.reward(fulfillment_metric, St1, St2, x1, x2, Dt)
        c1, c2 = self.updateAgentCost(reward1, reward2)
        St1_1, St2_1 = self.transition(fulfillment_metric, St1, St2, x1, x2, Dt)
        yt1_begin = sum(St1)
        yt2_begin = sum(St2)
        yt1 = sum(St1) + x1
        yt2 = sum(St2) + x2
        yt1_end = sum(St1_1)
        yt2_end = sum(St2_1)

        info = {
            "t": self.t,
            "St1": St1,
            "St1_1": St1_1,
            "xt1": x1,
            "Dt": Dt,
            "C(St1,xt1,Dt)": reward1,
            "yt1_begin": yt1_begin,
            "yt1": yt1,
            "yt1_end": yt1_end,
            "St2": St2,
            "St2_1": St2_1,
            "xt2": x2,
            "C(St2,xt2,Dt)": reward2,
            "yt2_begin": yt2_begin,
            "yt2": yt2,
            "yt2_end": yt2_end,
            "chosen_inventory": self.chosen_inventory
        }
        for i in range(self.m):
            info[f"yt1_{i}"] = St1[i]
            info[f"yt2_{i}"] = St2[i]

        self.current_state = (St1_1, St2_1)
        self.t += 1
        return self.t - 1, (St1, St2), (St1_1, St2_1), (reward1, reward2), self._terminate(self.t), info

    def _render(self):
        pass

    def _terminate(self, t):
        return t == self.T
    
    def get_winner(self):

        if abs(self.C1) < abs(self.C2):
            return "Pharmacy 1", self.C1, self.C2
        elif abs(self.C1) > abs(self.C2):
            return "Pharmacy 2", self.C1, self.C2
        else:
            return "Tie", self.C1, self.C2

    def updateAgentCost(self, r1, r2):
        self.C1 += r1
        self.C2 += r2

        return self.C1, self.C2
    
    def _reset(self):
        self.t = 0
        self.C1 = 0
        self.C2 = 0
        self.current_state = self.initial_state
