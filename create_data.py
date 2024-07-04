import numpy as np

def create_input_data():
    d = np.zeros((4, 4))
    
    #0なら横、1なら縦
    ver_or_hor = np.random.randint(0,2)
    if(ver_or_hor == 0):
        #横
        for i in range(4):
            on_off = np.random.randint(0,2)
            if(on_off == 1):
                #i行目の要素を全て1にする
                d[i] = 1
    else:
        #縦
        for i in range(4):
            on_off = np.random.randint(0,2)
            if(on_off == 1):
                #i行目の要素を全て1にする
                d[:,i] = 1
    
    # 全ての要素が1の場合は再帰
    if(np.all(d == 1)):
        d = create_input_data()
        
    d = d.reshape(16)
    return d


def create_input_data_all():
    d_list = []
    tmp_d = np.zeros((4, 4))
    # 全て0
    d_list.append(tmp_d.reshape(16))
    
    # 横
    #一本の場合
    for i in range(4):
        tmp_d = np.zeros((4, 4))
        tmp_d[i] = 1
        d_list.append(tmp_d.reshape(16))        
    
    #二本の場合
    for i in range(4):
        for j in range(4):
            tmp_d = np.zeros((4, 4))
            tmp_d[i] = 1
            if(i < j):
                tmp_d[j] = 1
                d_list.append(tmp_d.reshape(16))
    #三本の場合
    for i in range(4):
        tmp_d = np.ones((4, 4))
        tmp_d[i] = 0
        d_list.append(tmp_d.reshape(16))
    
    # 縦
    #一本の場合
    for i in range(4):
        tmp_d = np.zeros((4, 4))
        d_list.append(tmp_d.reshape(16))
        tmp_d[:,i] = 1
    #二本の場合
    for i in range(4):
        for j in range(4):
            tmp_d = np.zeros((4, 4))
            tmp_d[:,i] = 1
            if(i < j):
                tmp_d[:,j] = 1
                d_list.append(tmp_d.reshape(16))
    #三本の場合
    for i in range(4):
        tmp_d = np.ones((4, 4))
        tmp_d[:,i] = 0
        d_list.append(tmp_d.reshape(16))
        
    return d_list