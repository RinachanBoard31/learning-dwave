import numpy as np
from neal import SimulatedAnnealingSampler

# 第1項: ルートの重複を調べる
def get_traffic_qubo(cars_size, roots_size, K):
    qubo_size = cars_size * roots_size
    traffic_qubo = np.zeros((qubo_size, qubo_size))
    indices = [(u, v, i, j) for u in range(cars_size) for v in range(cars_size) for i in range(roots_size) for j in range(roots_size)]
    for u, v, i, j in indices:
        ui = u * roots_size + i
        vj = v * roots_size + j
        if ui > vj:
            continue
        if ui == vj:
            traffic_qubo[ui][vj] -= K  # 自分自身のペナルティを強化
        if u == v and i != j:
            traffic_qubo[ui][vj] += 4 * K  # 同じ車が複数のルートを選ばないように強制
    return traffic_qubo

# 第2項: 各車は1つしかルートを選べない
def get_traffic_cost_qubo(cost_matrix):
    traffic_cost_qubo = np.zeros((len(cost_matrix[0]), len(cost_matrix[0])))
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix[i])):
            traffic_cost_qubo += np.outer(cost_matrix[i], cost_matrix[i])  # コスト行列
    return traffic_cost_qubo

# 観測
def get_traffic_optimisation(traffic_qubo, traffic_cost_qubo):
    qubo = traffic_qubo + traffic_cost_qubo

    # Simulated Annealing Sampler を利用して QUBO を最適化
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo)
    print(response)

    # プロットの代替手段
    try:
        import matplotlib.pyplot as plt
        plt.bar(range(len(response.record)), response.record['sample'][0])
        plt.show()
    except ImportError:
        print("Matplotlibがインストールされていないため、プロットできませんでした。")
    return response

# パラメータの設定
pu_cost_matrix = [
    [1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1]
]
pu_cars_size = 2
pu_roots_size = int(len(pu_cost_matrix[0]) / pu_cars_size)
pu_K = 50  # ペナルティを強化

# 交通量のQUBO行列を取得(第1項)
traffic_qubo = get_traffic_qubo(pu_cars_size, pu_roots_size, pu_K)

# 交通コストのQUBO行列を取得(第2項)
traffic_cost_qubo = get_traffic_cost_qubo(pu_cost_matrix)

# 交通最適化問題を解く
q = get_traffic_optimisation(traffic_qubo, traffic_cost_qubo)
