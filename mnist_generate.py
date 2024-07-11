import numpy as np
import matplotlib.pyplot as plt
from mnist_extract import x_train_normalized
from input_time import saved_time

def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))


_filepath_w_g_single = './weights/' + saved_time +  '_w_g_single.txt'
_filepath_w_g_kj = './weights/' + saved_time +  '_w_g_kj.txt'
_filepath_w_g_ji = './weights/' + saved_time +  '_w_g_ji.txt'

w_g_single = np.loadtxt(_filepath_w_g_single)
w_g_kj = np.loadtxt(_filepath_w_g_kj)
w_g_ji = np.loadtxt(_filepath_w_g_ji)

mnist_list = []

for i in range(20): 
    ## トップダウンで生成
    # generative single biasを使用
    p_k = sigmoid(w_g_single)
    # 確率に従ってサンプリング
    s_k = np.random.binomial(1, p_k)
    # s_k = np.array([[0,0,0,0]])
    # s_k = np.array([[0,0,0,1]])
    # s_k = np.array([[0,0,1,0]])
    # s_k = np.array([[0,0,1,1]])
    # s_k = np.array([[0,1,0,0]])
    # s_k = np.array([[0,1,0,1]])
    # s_k = np.array([[0,1,1,0]])
    # s_k = np.array([[0,1,1,1]])
    # s_k = np.array([[1,0,0,0]])
    # s_k = np.array([[1,0,0,1]])
    # s_k = np.array([[1,0,1,0]])
    # s_k = np.array([[1,0,1,1]])
    # s_k = np.array([[1,1,0,0]])
    # s_k = np.array([[1,1,0,1]])
    # s_k = np.array([[1,1,1,0]])
    # s_k = np.array([[1,1,1,1]])

    # バイアスを追加
    s_k_appended = np.append(s_k, 1)
    # 重みづけの和を計算
    p_j = s_k_appended @ w_g_kj
    p_j = sigmoid(p_j)
    # 確率に従ってサンプリング
    s_j = np.random.binomial(1, p_j)

    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    # 重みづけの和を計算
    p_i = s_j_appended @ w_g_ji
    p_i = sigmoid(p_i)
    
    # 確率に従ってサンプリング
    # s_i = np.random.binomial(1, p_i)
    
    #interp関数により、0-1の値を0-255に変換
    scaled_values_interp = np.interp(p_i, (0, 1), (0, 255)).astype(int)
    mnist_list.append(scaled_values_interp.reshape(28,28))


# ２値画像にしたよ
# for i in range(5): 
#     ## トップダウンで生成
#     # generative single biasを使用
#     p_k = sigmoid(w_g_single)
#     # 確率に従ってサンプリング
#     s_k = np.random.binomial(1, p_k)

#     # バイアスを追加
#     s_k_appended = np.append(s_k, 1)
#     # 重みづけの和を計算
#     p_j = s_k_appended @ w_g_kj
#     p_j = sigmoid(p_j)
#     # 確率に従ってサンプリング
#     s_j = np.random.binomial(1, p_j)

#     # バイアスを追加s
#     s_j_appended = np.append(s_j, 1)
#     # 重みづけの和を計算
#     p_i = s_j_appended @ w_g_ji
#     p_i = sigmoid(p_i)

#     # 確率に従ってサンプリング
#     s_i = np.random.binomial(1, p_i)

#     #interp関数により、0-1の値を0-255に変換
#     scaled_values_interp = np.interp(s_i, (0, 1), (0, 255)).astype(int)
#     mnist_list.append(scaled_values_interp.reshape(28,28))

# 抽出された画像をいくつか表示
plt.figure(figsize=(20, 10))
for i in range(len(mnist_list)):
    plt.subplot(4, 5, i+1)
    plt.imshow(mnist_list[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()