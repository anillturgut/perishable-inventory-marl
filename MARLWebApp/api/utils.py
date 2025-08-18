import random
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

def calculate_fulfillment_metric(St1, St2, w1=1, w2=1):

    I1 = sum(St1)
    I2 = sum(St2)
    V1 = 2 * St1[-1] + St1[0]
    V2 = 2 * St2[-1] + St2[0]

    if max(I1, I2) == 0:
        normalized_inventory_difference = 0
    else:
        normalized_inventory_difference = (g(I1 - I2) * abs(I1 - I2)) / max(I1, I2)
    return w1 * normalized_inventory_difference + w2 * (V1 - V2)

def choosePharmacy(fulfillment_metric):
    if fulfillment_metric > 0:
        return 1
    elif fulfillment_metric < 0:
        return 2
    else:
        return random.choice([1,2])
    

