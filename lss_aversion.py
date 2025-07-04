import numpy as np

def homesky_salience_weight(x, alpha=0.88, beta=0.88, lambda_=2.25, gamma=0.1):
    ret_value = gamma
    if x > 0:
        ret_value += x**alpha
    elif x < 0:
        ret_value += ( (1 + lambda_) * ((-x)**beta) ) # non-zero fixed weight for 0
    return ret_value


# Step 3: Sample n keys according to normalized weights
def sample_keys(weighted_list, n):
    keys = [key for key, _ in weighted_list]
    probabilities = [weight for _, weight in weighted_list]
    return np.random.choice(keys, size=n, replace=True, p=probabilities)