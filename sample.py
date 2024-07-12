import numpy as np

import itertools

# 10ビットのすべての組み合わせを生成
combinations = list(itertools.product([0, 1], repeat=10))

# 各組み合わせをnumpy配列に変換し、リストに格納
s_k_list = [np.array([list(comb)]) for comb in combinations]

# 確認のために最初のいくつかを出力
for item in s_k_list[:5]:
    print(item)
