import pandas as pd
from Cluster import *


log_K = []
log_OR = []


def load_datasets(path):
    df = pd.read_csv(path)
    y = df.Label
    X = df.drop('Label', axis=1)
    return X, y


def compute_or(outliers, shape):
    return ((1 / shape[0]) * outliers[0]) / ((1 / shape[1]) * outliers[1])


def do_loop(K, alpha):
    model = Cluster(n_cluster=K)

    # load CICIDS-2017
    X, y = load_datasets('../data/Domain-CICIDS-2017-Norm.csv')
    model.Train(X)

    y_c = model.Test(X)
    distances = model.MeanD(X, y_c)

    T = model.ComputeT(distances, alpha)

    # 标记离群点
    outliers = np.where(distances > T)[0]

    # load CSE-CICIDS-2018
    _X, _y = load_datasets('../data/Domain-CICIDS-2018-Norm.csv')
    _y_c = model.Test(_X)

    _distances = model.MeanD(_X, _y_c)

    # 标记离群点
    _outliers = np.where(_distances > T)[0]

    OR = compute_or([_outliers.shape[0], outliers.shape[0]], [_X.shape[0], X.shape[0]])

    log_K.append(K)
    log_OR.append(OR)

    print(f'## K = {K}, Alpha = {alpha}, T = {T}, {outliers.shape[0]}/{X.shape[0]}, {_outliers.shape[0]}/{_X.shape[0]}, OR = {OR}')
    return OR, outliers.shape[0]


if __name__ == '__main__':
    for K in range(9, 101):
        alpha = 1.0
        OR, outliers = do_loop(K, alpha)
        with open(f'txt/k_or/{alpha}/output_k.txt', 'a') as f:
            f.write(f"{K},")

        with open(f'txt/k_or/{alpha}/output_or.txt', 'a') as f:
            f.write(f"{OR},")

        with open(f'txt/k_or/{alpha}/output_outliers.txt', 'a') as f:
            f.write(f"{outliers},")

    # for i in range(101):
    #     alpha = round(0.0 + i * 0.1, 1)
    #     OR, outliers = do_loop(53, alpha)

    #     with open(f'txt/alpha_or/output_alpha.txt', 'a') as f:
    #         f.write(f"{alpha},")

    #     with open(f'txt/alpha_or/output_or.txt', 'a') as f:
    #         f.write(f"{OR},")

    #     with open(f'txt/alpha_or/output_outliers.txt', 'a') as f:
    #         f.write(f"{outliers},")

