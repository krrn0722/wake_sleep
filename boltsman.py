import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



x_neuron = np.zeros(4)
# 3番目のニューロンの値は常に1
x_neuron[3] = 1




# 横行目から縦列目への結合の重み
w = np.random.rand(4,4)

iter = 10000
T = 1

x_memory = x_neuron
print(x_memory,"最初")

for t in range(iter):

    # どのニューロンを変更するかランダムに決定
    changeIndex = np.random.randint(0,3)

    # 重み付けの和を計算
    sum = w[:,changeIndex] @ x_neuron
    u = sigmoid(sum)
    p = 1/ (1 + np.exp(-u)/T)

    # 確率に従ってサンプリング
    x_neuron[changeIndex] = np.random.binomial(1, p)

    
    x_memory = np.vstack([x_memory, x_neuron])

print(x_memory)