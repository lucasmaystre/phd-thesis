import numpy as np
import random
import sys


def ranks(vals):
    """Compute the rank of each item."""
    items = list(range(len(vals)))
    random.shuffle(items)  # Randomizes things in case of ties.
    ordered = sorted(items, key=lambda x: vals[x], reverse=True)
    ranks = np.zeros(len(vals), dtype=int)
    for rank, item in enumerate(ordered, start=1):
        ranks[item] = rank
    return ranks

    
def displacement(vals1, vals2):
    """Compute the rank displacement."""
    ranks1 = ranks(vals1)
    ranks2 = ranks(vals2)
    total = 0
    for i, j in zip(ranks1, ranks2):
        total += abs(i - j)
    return total


def weighted_quantiles(data, weights, fractions=[0.5]):
    """Calculate weighted quantiles.

    Basically, this function uses a trick that avoids duplicating the same
    measurements multiple times.
    """
    # Preprocessing (mostly sorting), takes the most time.
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    ind = np.argsort(data)
    sdata = np.array(data[ind])
    sweights = np.array(weights[ind])
    cum_weight = np.cumsum(sweights)
    # Actual work.
    quantiles = list()
    for fraction in fractions:
        point = fraction * np.sum(weights)
        below_idx = np.where(cum_weight <= point)[0][-1]
        if cum_weight[below_idx] - point < sys.float_info.epsilon:
            quantiles.append(sdata[below_idx])
            continue
        quantiles.append(sdata[below_idx+1])
    return np.array(quantiles)


def qtod(quantiles):
    """Generate a synthetic dataset with prescribed quantiles."""
    data = np.zeros(101)
    nxt = 0
    for q, val in sorted(quantiles):
        for i in range(nxt, int(100 * q) + 1):
            data[i] = val
        nxt = int(100 * q) + 1
    for i in range(nxt, 101):
        data[i] = val
    return data
