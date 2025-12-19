import numpy as np


def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
