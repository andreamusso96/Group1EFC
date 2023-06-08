import pandas as pd

from Data import Data
import numpy as np


class Algorithms:
    @staticmethod
    def compute_fitness_complexity(data: Data, rta_threshold: float = 1):
        mcp = Algorithms._mcp(data=data, threshold=rta_threshold)
        fitness, complexity = Algorithms._fitness_complexity_algorithm(mcp=mcp)
        fitness_df = pd.DataFrame(data=fitness, index=data.data.index, columns=['fitness'])
        complexity_df = pd.DataFrame(data=complexity, index=data.data.columns, columns=['complexity'])
        return fitness_df, complexity_df

    @staticmethod
    def compute_product_space(data: Data, rta_threshold: float = 1, method: str = 'proximity'):
        mcp = Algorithms._mcp(data=data, threshold=rta_threshold)
        if method == 'proximity':
            product_space = Algorithms._product_space_proximity_algorithm(mcp=mcp)
        elif method == 'taxonomy':
            product_space = Algorithms._product_space_taxonomy_algorithm(mcp=mcp)
        else:
            raise ValueError(f'Method {method} not supported')
        return product_space

    @staticmethod
    def _rta(data: Data):
        df = data.data
        numerator = (df.T / df.sum(axis=1)).T
        denominator = df.sum(axis=0) / df.sum().sum()
        rta = numerator / denominator
        rta.fillna(0, inplace=True)
        return rta

    @staticmethod
    def _mcp(data: Data, threshold: float = 1):
        rta = Algorithms._rta(data=data)
        mcp = np.where(rta.values > threshold, 1, 0)  # Binarization of the matrix
        return mcp

    @staticmethod
    def _fitness_complexity_algorithm(mcp, maximal_iterations=100):
        fitness = np.sum(mcp, axis=1) / np.sum(mcp)  # Share of patents produced by each country (Diversification rescaled)
        complexity = np.sum(mcp, axis=0).astype(np.float64)  # Ubiquity of each technology (number of countries that produce that technology code)

        one_fit = np.ones_like(fitness)
        one_com = np.ones_like(complexity)
        np.divide(one_com, complexity, out=complexity, where=complexity != 0)

        complexity /= complexity.sum()
        mpc = np.transpose(mcp)
        inverse_fitness = np.zeros_like(fitness)

        # loop over the iterations
        for iteration in range(maximal_iterations):
            # compute the inverse fitness
            np.divide(one_fit, fitness, out=inverse_fitness, where=fitness != 0)

            # update the fitness
            fitness = mcp.dot(complexity)
            fitness /= fitness.sum()

            # update the complexity
            complexity = mpc.dot(inverse_fitness)
            np.divide(one_com, complexity, out=complexity, where=complexity != 0)
            complexity /= complexity.sum()

        return fitness / fitness.mean(), complexity / complexity.mean()

    @staticmethod
    def _product_space_proximity_algorithm(mcp):
        correlation = np.matmul(np.transpose(mcp), mcp)
        max_ubiquity_matrix = Algorithms._max_ubiquity_matrix(mcp=mcp)
        proximity_matrix = np.multiply(correlation, max_ubiquity_matrix)
        return proximity_matrix

    @staticmethod
    def _product_space_taxonomy_algorithm(mcp):
        diversification = mcp.sum(1)
        diversification_matrix = np.transpose(np.tile(diversification, [mcp.shape[1], 1]))
        mcp_divided_by_diversification = np.divide(mcp, diversification_matrix, where=diversification_matrix != 0)
        A = np.matmul(np.transpose(mcp), mcp_divided_by_diversification)
        max_ubiquity_matrix = Algorithms._max_ubiquity_matrix(mcp=mcp)
        taxonomy_matrix = np.multiply(A, max_ubiquity_matrix)
        return taxonomy_matrix

    @staticmethod
    def _max_ubiquity_matrix(mcp):
        ubiquity = mcp.sum(axis=0)
        ubiquity_matrix = np.tile(ubiquity, [mcp.shape[1], 1])
        max_ubiquity_matrix = np.maximum(ubiquity_matrix, np.transpose(ubiquity_matrix)).astype(float)
        max_ubiquity_matrix = np.divide(np.ones_like(max_ubiquity_matrix, dtype=float), max_ubiquity_matrix, where=max_ubiquity_matrix != 0)
        return max_ubiquity_matrix
    


if __name__ == '__main__':
    from Data import FileType, GeographicScale
    data = Data(year=2018, geo_scale=GeographicScale.REGION, file_type=FileType.Y)
    f, c = Algorithms.compute_fitness_complexity(data=data)
    product_space = Algorithms.compute_product_space(data=data)



