import numpy as np
from create_data import create_input_data
from create_data import create_input_data_all

def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def relu(x):
    return np.maximum(0, x)



# 重みを初期化
k_num = 2 #1
j_num = 8 #8
i_num = 16 #16

# 学習率 epsilon
lr = 0.1

# w_r_ij = np.random.rand(i_num+1, j_num)
# w_r_jk = np.random.rand(j_num+1, k_num)


# w_g_kj = np.random.rand(k_num+1, j_num)
# w_g_ji = np.random.rand(j_num+1, i_num)

# w_g_single = np.random.rand(1, k_num)

w_r_ij = np.zeros((i_num+1, j_num))
w_r_jk = np.zeros((j_num+1, k_num))


w_g_kj = np.zeros((k_num+1, j_num))
w_g_ji = np.zeros((j_num+1, i_num))

w_g_single = np.zeros((1, k_num))


s_k = np.zeros(k_num)
s_j = np.zeros(j_num)
s_i = np.zeros(i_num)

iter = 500000

# 入力dを作成
d_list = create_input_data_all()

for i in range(iter):
    
    # ランダムに選択
    d= d_list[np.random.randint(0, len(d_list))]
    # 順番に選択
    # d = d_list[i % len(d_list)]
    
    
    ##### wakeフェーズ
    ### ボトムアップ

    # バイアスを追加
    d_appended = np.append(d, 1)

    # 重みづけの和を計算
    s_j = d_appended @ w_r_ij
    s_j = sigmoid(s_j)
    # 確率に従ってサンプリング
    s_j = np.random.binomial(1, s_j)

    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    # 重みづけの和を計算
    s_k = s_j_appended @ w_r_jk
    s_k = sigmoid(s_k)
    # 確率に従ってサンプリング
    s_k = np.random.binomial(1, s_k)


    ### トップダウン
    p_k = np.zeros(k_num)
    p_j = np.zeros(j_num)
    p_i = np.zeros(i_num)

    # generative single biasを使用
    p_k = sigmoid(w_g_single)

    # バイアスを追加
    s_k_appended = np.append(s_k, 1)
    # 重みづけの和を計算
    p_j = s_k_appended @ w_g_kj
    p_j = sigmoid(p_j)

    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    # 重みづけの和を計算
    p_i = s_j_appended @ w_g_ji
    p_i = sigmoid(p_i)


    ## 生成重みの更新
    # シングル生成バイアスの更新
    delta_w_g_single = s_k - p_k
    w_g_single = w_g_single + lr * delta_w_g_single

    # 2,3層生成重みの更新
    # バイアスを追加
    s_k_appended = np.append(s_k, 1)
    # 形を整える
    s_k_reshape = s_k_appended.reshape(k_num+1,1)
    s_j_reshape = s_j.reshape(1,j_num)
    p_j_reshape = p_j.reshape(1,j_num)
    delta_w_g_kj = s_k_reshape @ (s_j_reshape - p_j_reshape)
    w_g_kj = w_g_kj + lr * delta_w_g_kj
    

    # 1,2層生成重みの更新
    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    # 形を整える
    s_j_reshape = s_j_appended.reshape(j_num+1,1)
    s_i_reshape = d.reshape(1,i_num)
    p_i_reshape = p_i.reshape(1,i_num)
    delta_w_g_ji = s_j_reshape @ (s_i_reshape - p_i_reshape)
    
    
    w_g_ji = w_g_ji + lr * delta_w_g_ji
    
    ##### sleepフェーズ

    ## トップダウン
    s_k = sigmoid(w_g_single)
    # 確率に従ってサンプリング
    s_k = np.random.binomial(1, s_k)

    # バイアスを追加
    s_k_appended = np.append(s_k, 1)
    # 重みづけの和を計算
    s_j = s_k_appended @ w_g_kj
    s_j = sigmoid(s_j)
    # 確率に従ってサンプリング
    s_j = np.random.binomial(1, s_j)

    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    s_i = s_j_appended @ w_g_ji
    s_i = sigmoid(s_i)
    # 確率に従ってサンプリング
    s_i = np.random.binomial(1, s_i)


    ## ボトムアップ
    # これiはいらないよね
    q_j = np.zeros(j_num)
    q_k = np.zeros(k_num)

    # バイアスを追加
    s_i_appended = np.append(s_i, 1)
    q_j = s_i_appended @ w_r_ij
    q_j = sigmoid(q_j)

    # バイアスを追加
    s_j_appended = np.append(s_j, 1)
    q_k = s_j_appended @ w_r_jk
    q_k = sigmoid(q_k)

    ## 認識重みの更新
    # 1,2層認識重みの更新
    s_i_reshape = s_i_appended.reshape(i_num+1,1)
    s_j_reshape = s_j.reshape(1,j_num)
    q_j_reshape = q_j.reshape(1,j_num)
    delta_w_r_ij = s_i_reshape @ (s_j_reshape - q_j_reshape)
    w_r_ij = w_r_ij + lr * delta_w_r_ij

    # 2,3層認識重みの更新
    s_j_reshape = s_j_appended.reshape(j_num+1,1)
    s_k_reshape = s_k.reshape(1,k_num)
    q_k_reshape = q_k.reshape(1,k_num)
    delta_w_r_jk = s_j_reshape @ (s_k_reshape - q_k_reshape)
    w_r_jk = w_r_jk + lr * delta_w_r_jk

w_g_ji_reshape = w_g_ji.reshape(j_num+1, 4,4)
print("w_g_ji_reshape",w_g_ji_reshape)
# print("g_single \n",w_g_single) 
# print("g_kj \n",w_g_kj)

# print("g_ji",w_g_ji)



for i in range(3): 
    ## トップダウンで生成
    # generative single biasを使用
    p_k = sigmoid(w_g_single)
    # 確率に従ってサンプリング
    s_k = np.random.binomial(1, p_k)

    # バイアスを追加
    s_k_appended = np.append(s_k, 1)
    # 重みづけの和を計算
    p_j = s_k_appended @ w_g_kj
    p_j = sigmoid(p_j)
    # 確率に従ってサンプリング
    s_j = np.random.binomial(1, p_j)

    # バイアスを追加s
    s_j_appended = np.append(s_j, 1)
    # 重みづけの和を計算
    p_i = s_j_appended @ w_g_ji
    p_i = sigmoid(p_i)
    # 確率に従ってサンプリング
    s_i = np.random.binomial(1, p_i)

    print(s_i.reshape(4,4))
