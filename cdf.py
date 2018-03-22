import numpy as np

def get_cdf(prob):
    cdf = [np.sum(prob[:i+1]) for i in range(prob.shape[0])]
    rand = np.random.uniform(0,1)
    val = 0
    for ite in range(prob.shape[0]):
        if rand < cdf[ite]:
            val = ite
            break
    return val

