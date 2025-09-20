import numpy as np
import pandas as pd
import math
import copy

class Binomial:
    def __init__(self, increase_factor: float, decrease_factor: float, risk_free_rate: float, volatility: float) -> None:

        if increase_factor <= 1:
            raise ValueError("Wrong increase_factor provided. Inserted value must satisfy: increase_factor > 1 ")
        else:
            self.u = increase_factor

        if decrease_factor >= 1 or decrease_factor <= 0:
            raise ValueError("Wrong decrease_factor provided. Inserted value must satisfy: 0 < decrease_factor < 1 ")
        else:
            self.d = decrease_factor

        self.r = risk_free_rate
        self.sigma = volatility
        # self.delta = 0.5

        # self.p = lambda t: (np.exp(self.r*self.delta*t) - self.d) / (self.u - self.d)

    def binomial_price_tree(self, S: float, T: int, delta: float) -> float | np.ndarray:

        # number of steps in tree simulation
        steps = int(T/delta)
        # matrix creation
        tree_matrix = np.array([np.array([0] * (steps+1)) for _ in range(steps+1)])
        # initial value at time 0
        tree_matrix[steps][0] = S

        for i in range(steps+1):            # row iteration
            for j in range(i, steps+1):     # column iteration
                tree_matrix[steps-i][j] = S * self.u**i * self.d**(j-i)

        return tree_matrix

    def option_pricing(self, K: float, delta: float, tree_matrix: float | np.ndarray, is_call: bool=True) -> float | np.ndarray:

        p = (np.exp(self.r*delta) - self.d) / (self.u - self.d)
        size = len(tree_matrix)
        option_payoff_matrix = np.maximum(0, (-1)**(1-is_call) * (tree_matrix - K))

        # european

        # american
        for i in range(1, size):
            for j in range(i, size):
                print(f'i:{i}, j:{j}')
                option_payoff_matrix[j][size-i] = np.exp(-self.r * delta) * (p*option_payoff_matrix[j-1][size-i] + (1-p)*option_payoff_matrix[j][size-i])

        return option_payoff_matrix


# TESTS ################################################################################################################
Binomial_Tree = Binomial(1.1, 0.9, 0.01, 0.15)
asset_price_tree = Binomial_Tree.binomial_price_tree(S=100, T=5, delta=1)

print(np.maximum(0, asset_price_tree - 100))
asset_price_tree_df = pd.DataFrame(asset_price_tree)
print(asset_price_tree_df)

option_value_tree = Binomial_Tree.option_pricing(K=100, delta=1, tree_matrix=asset_price_tree)
option_value_tree_df = pd.DataFrame(option_value_tree)
print(option_value_tree_df)

