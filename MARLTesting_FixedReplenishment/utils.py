import random
import numpy as np
def interpolate(start, stop, step):
    if step == 1:
        return [start]
    return [start + (stop - start) / step * i for i in range(step + 1)]

def combinations(k, n):
    if k == 1:
        for i in range(n + 1):
            yield [i]
    else:
        for i in range(n + 1):
            for combination in combinations(k - 1, n):
                yield [i] + combination

def g(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1
    

def calculate_fulfillment_metric(St, wn=1.0, wv=1.0):

    N = len(St)
    m = len(St[0])

    # Age-weighted freshness g_jt for each player
    G_raw = [sum((i+1) * St[j][i] for i in range(m)) for j in range(N)]
    G_max = max(G_raw) if max(G_raw) != 0 else 1  # avoid division by zero

    # Normalized freshness: G_jt
    G = [(G_raw[j] - G_max) / G_max if G_max != 0 else 0 for j in range(N)]

    # Total inventory Y_jt for each player
    Y = [sum(St[j]) for j in range(N)]
    Y_max = max(Y) if max(Y) != 0 else 1  # avoid division by zero

    # Normalized inventory difference: NID_jt
    NID = [(Y[j] - Y_max) / Y_max if Y_max != 0 else 0 for j in range(N)]

    # Fulfillment metric F_jt = wn * NID + wv * G
    F = [1 * NID[j] + 1 * G[j] for j in range(N)]

    return F



def choosePharmacy(fulfillment_metric):
    
    max_val = max(fulfillment_metric)
    max_indices = [i for i, val in enumerate(fulfillment_metric) if val == max_val]
    return random.choice(max_indices) + 1


def update_initial_inventory_states(St, x):
    new_St = []
    for j, inv in enumerate(St):
        updated = inv.copy()
        updated[-1] = x[j]
        new_St.append(updated)
    return tuple(new_St)