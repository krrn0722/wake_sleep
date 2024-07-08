import numpy as np
from create_data import create_input_data
from create_data import create_input_data_all
from generate_define_weight import w_g_single, w_g_kj, w_g_ji

def check(arr):
    ver_check = False
    hor_check = False
    # 配列が4x4であることを確認
    if arr.shape != (4, 4):
        raise ValueError("配列は4x4でなければなりません")
    
    # 各列で1が4つ揃っているかチェック
    for col in range(4):
        if np.all(arr[:, col] == 1):
            ver_check = True
            break
    for row in range(4):
        if np.all(arr[row, :] == 1):
            hor_check = True
            break
    return ver_check , hor_check

def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))



generated_list = []

for i in range(100):
    ## トップダウンで生成
    # generative single biasを使用
    p_k = sigmoid(w_g_single)
    # 確率に従ってサンプリング
    s_k = np.random.binomial(1, p_k)
    print(s_k.shape)
    s_k = np.array([[0, 0]])
    # s_k = np.array([[0, 1]])
    # s_k = np.array([[1, 0]])
    # s_k = np.array([[1, 1]])

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

    generated_list.append(s_i.reshape(4,4))


results = list(map(check, generated_list))

both_t_count = 0
ver_count = 0
hor_count = 0
both_f_count = 0

for result in results:
    if(result[0] and result[1]):
        both_t_count += 1
    elif(result[0]):
        ver_count += 1
    elif(result[1]):
        hor_count += 1
    else:
        both_f_count += 1

print("どっちも:",both_t_count,"\n","縦:", ver_count, "\n" ,"横:", hor_count,"\n" ,"なし:",both_f_count)

with open('generated_list.txt', 'w') as f:
    for i in range(len(generated_list)):
        f.write(str(generated_list[i]) + '\n'+'\n')