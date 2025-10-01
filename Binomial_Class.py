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
        # self.delta_t = 0.5

        # self.p = lambda t: (np.exp(self.r*self.delta_t*t) - self.d) / (self.u - self.d)

    def binomial_price_tree(self, S: float, T: int, delta_t: float) -> float | np.ndarray:

        # number of steps in tree simulation
        steps = int(T/delta_t)
        # matrix creation
        tree_matrix = np.array([np.array([0] * (steps+1)) for _ in range(steps+1)], dtype=float)
        # initial value at time 0
        tree_matrix[steps][0] = S

        for i in range(steps+1):            # row iteration
            for j in range(i, steps+1):     # column iteration
                tree_matrix[steps-i][j] = S * self.u**i * self.d**(j-i)

        return tree_matrix

    def european_option_pricing(self, K: float, delta_t: float, tree_matrix: float | np.ndarray, is_call: bool=True) -> float:

        p = (np.exp(self.r * delta_t) - self.d) / (self.u - self.d)
        size = len(tree_matrix)
        # coefficients
        pascal_triangle = np.zeros(size)
        # vector for powers of p | (1-p)
        p_vector = np.zeros(size)

        for k in range(size):
            pascal_triangle[k] = math.factorial(size-1) / (math.factorial(size-1-k) * math.factorial(k))
            p_vector[k] = p**(size-1-k) * (1-p)**k

        option_payoffs = np.maximum(0, (-1)**(1-is_call) * (tree_matrix - K))
        option_value = np.exp(-self.r * int(delta_t * (size-1))) * np.sum(p_vector * pascal_triangle * option_payoffs[:, -1])

        return np.round(option_value, 2)

    def american_option_pricing(self, K: float, delta_t: float, tree_matrix: float | np.ndarray, is_call: bool=True) -> float | np.ndarray:

        p = (np.exp(self.r*delta_t) - self.d) / (self.u - self.d)
        size = len(tree_matrix)
        # matrix with payoffs
        option_payoff_matrix = np.maximum(0, (-1)**(1-is_call) * (tree_matrix - K))
        # matrix with option values (at different time points
        option_value_matrix = copy.deepcopy(option_payoff_matrix)
        # matrix execution moments
        option_exec_matrix = copy.deepcopy(option_payoff_matrix)
        # matrix execution moments - last column valuation
        option_exec_matrix[:, size-1] = option_payoff_matrix[:, size-1] > 0

        for i in range(1, size):
            for j in range(i, size):
                # print(f'i:{i}, j:{j}')
                risk_neutral_valuation = np.exp(-self.r * delta_t) * (p*option_value_matrix[j-1, size-i] + (1-p)*option_value_matrix[j, size-i])
                option_value_matrix[j, size-i-1] = np.maximum(risk_neutral_valuation, option_payoff_matrix[j, size-i-1])
                option_exec_matrix[j, size-i-1] = risk_neutral_valuation < option_payoff_matrix[j, size-i-1]

        return np.round(option_value_matrix, 2)


# TESTS ################################################################################################################
Binomial_Tree = Binomial(1.1, 0.9, 0.05, 0.15)
asset_price_tree = Binomial_Tree.binomial_price_tree(S=100, T=5, delta_t=1)

print("Payoffs matrix")
print(np.maximum(0, (asset_price_tree - 100)))
print("\n")

eu_option_value = Binomial_Tree.european_option_pricing(K=100, delta_t=1, tree_matrix=asset_price_tree)
print(f'European option price: {eu_option_value}')
print("\n")
# print(asset_price_tree)
# print(np.maximum(asset_price_tree - 100, 0))

option_value_tree = Binomial_Tree.american_option_pricing(K=100, delta_t=1, tree_matrix=asset_price_tree)
print("American option price matrix")
print(option_value_tree)
# option_value_tree_df = pd.DataFrame(option_value_tree)
# print(option_value_tree_df)

