import numpy as np
from openrec.utils.evaluators import Evaluator

class Rank(Evaluator):

    def __init__(self, name='Rank'):
        
        super(Rank, self).__init__(etype='visual', name=name)

    def compute(self, pos_samples, predictions):
        ind2rank = {ind: rank for rank, ind in enumerate(np.argsort(predictions)[::-1])}
        pos_ranks = [ind2rank[i] for i in pos_samples]
        neg_ranks = list(set(range(len(predictions))) - set(pos_ranks))
        return np.array([np.mean(pos_ranks), np.mean(neg_ranks)])
