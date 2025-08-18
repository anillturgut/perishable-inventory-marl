# /api/app.py
from flask import Flask, jsonify, render_template, request
import pandas as pd
from environment import CompetitivePerishableInventoryPlanning
from nash_qlearning import getNextReplenishmentDecision, negate_nashq_matrices
import utils
import numpy as np
import platform

app = Flask(__name__, template_folder='../templates', static_folder='../static')
moq = 5
m = 3
T = 10
# Initialize the environment
initial_state = [0] * m
env = CompetitivePerishableInventoryPlanning(product_life=m, maximum_order_quantity=moq, planning_horizon=10, 
                                             max_x=moq, initial_state=(initial_state, initial_state))

if platform.system() == "Darwin":
    file1_name = "TrainedAgents//Q1_m3_x"
    file2_name = "TrainedAgents//Q2_m3_x"
else:
    file1_name = "TrainedAgents\Q1_x"
    file2_name = "TrainedAgents\Q2_x"
# Load Q-matrices
Q1 = pd.read_pickle(file1_name + str(env.max_x) + ".pkl")
Q2 = pd.read_pickle(file2_name+ str(env.max_x) + ".pkl")
Q1_negated, Q2_negated = negate_nashq_matrices(Q1, Q2)


total_agent_cost = 0
total_user_cost = 0
current_day = 1
game_done = False

def reset_game():
    global total_agent_cost, total_user_cost, current_day, game_done
    total_agent_cost = 0
    total_user_cost = 0
    current_day = 1
    game_done = False

@app.route('/')
def index():
    print(env.max_x)
    max_replenishment = env.max_x
    product_lifetime = env.m
    return render_template('index.html', max_replenishment=max_replenishment, 
                           planning_horizon=env.T, 
                           product_life=env.m, 
                           maximum_order_quantity=env.max_x,
                           ordering_cost=env.co,
                           holding_cost=env.ch,
                           shortage_cost=env.cs,
                           deterioration_cost=env.cd,
                           lambd=env.l,
                           product_lifetime=product_lifetime)

@app.route('/api/make-decision', methods=['POST'])
def make_decision():
    global total_agent_cost, game_number, total_user_cost, current_day, game_done

    data = request.json
    day = data.get('day')
    user_replenishment = data.get('userReplenishment')
    current_agent_state = data.get('currentAgentState')
    current_user_state = data.get('currentUserState')

    game_done = True if (day - 1) == env.T else False

    current_state = (current_agent_state, current_user_state)
    index_day = day - 1

    # Agent decision
    agent_replenishment = getNextReplenishmentDecision(-Q1_negated, -Q2_negated, current_state, index_day)

    Dt = env.demand()
    fulfillment_metric = utils.calculate_fulfillment_metric(current_agent_state, current_user_state)
    agent_cost, user_cost = env.reward(fulfillment_metric, current_agent_state, current_user_state,
                                   agent_replenishment, user_replenishment, Dt)
    new_agent_state, new_user_state = env.transition(fulfillment_metric, current_agent_state, current_user_state,
                                   agent_replenishment, user_replenishment, Dt)

    total_agent_cost += agent_cost
    total_user_cost += user_cost
    # Send response back to the frontend
    response = {
        'day': int(day + 1),  # Convert NumPy int64 to Python int
        'indexDay': int(index_day + 1),
        'agentReplenishment': int(agent_replenishment),
        'agentCost': float(agent_cost),  # Convert NumPy float64 to Python float
        'userCost': float(user_cost),  # Convert NumPy float64 to Python float
        'totalAgentCost': float(total_agent_cost),  
        'totalUserCost': float(total_user_cost),
        'newAgentState': [int(state) for state in new_agent_state],  # Ensure all elements are Python ints
        'newUserState': [int(state) for state in new_user_state],  # Ensure all elements are Python ints,
        'demand': int(Dt),
        'chosenInventory': env.chosen_inventory,
        'fulfillmentMetric': float(round(fulfillment_metric, 3)),
        'done': game_done,

    }

    if game_done: 
        reset_game()

    return jsonify(response)

@app.route('/api/reset-game', methods=['POST'])
def reset_game_endpoint():
    reset_game()
    return jsonify({'status': 'Game reset successful'})

if __name__ == '__main__':
    app.run(debug=True)
