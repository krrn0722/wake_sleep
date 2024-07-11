from datetime import datetime
import numpy as np
import pytz

def output_weight(w_g_single, w_g_kj, w_g_ji):
    # 日本時間のタイムゾーンを取得
    tz = pytz.timezone('Asia/Tokyo')

    # 現在の日時を取得し、日本時間に変換
    now = datetime.now(tz)

    # 日付と時刻を指定の形式で整形
    formatted_now = now.strftime("%Y-%m-%d,%H:%M:%S")

    print("現在の日付と時刻:", formatted_now)

    # w_g_ji_reshape = w_g_ji.reshape(j_num+1, 28,28)
    # print("w_g_ji_reshape",w_g_ji_reshape)
    
    # print("g_single \n",w_g_single)
    # print("g_kj \n",w_g_kj)
    # print("g_ji",w_g_ji)
    
    # ファイルパス
    _filepath_w_g_single = './weights/' + formatted_now +  '_w_g_single.txt'
    _filepath_w_g_kj = './weights/' + formatted_now +  '_w_g_kj.txt'
    _filepath_w_g_ji = './weights/' + formatted_now +  '_w_g_ji.txt'
    
    np.savetxt(_filepath_w_g_single, w_g_single)
    np.savetxt(_filepath_w_g_kj, w_g_kj)
    np.savetxt(_filepath_w_g_ji, w_g_ji)
    
    print("ファイル出力完了 : ", formatted_now)