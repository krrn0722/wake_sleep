import numpy as np

ver_check = False
hor_check = False

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

# テスト用の配列
test_arr = np.array([
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1]
])

both_t_count = 0
ver_count = 0
hor_count = 0
both_f_count = 0

ver_check, hor_check = check(test_arr)
if(ver_check and hor_check):
    both_t_count += 1
elif(ver_check):
    ver_count += 1
elif(hor_check):
    hor_count += 1
else:
    both_f_count += 1
    
print("どっちも:",both_t_count,"\n","縦:", ver_count, "\n" ,"横:", hor_count,"\n" ,"なし:",both_f_count)