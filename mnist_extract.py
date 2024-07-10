import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNISTデータセットをロード
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def extract_digits(x, y, digits):
    """指定された数字のみを抽出する関数"""
    mask = np.isin(y, digits)
    return x[mask], y[mask]

# 数字の3と7を抽出
# digits_to_extract = [3, 7]
digits_to_extract = [5,7]
x_train_extracted, y_train_extracted = extract_digits(x_train, y_train, digits_to_extract)
x_test_extracted, y_test_extracted = extract_digits(x_test, y_test, digits_to_extract)

# # 抽出されたデータの分布を確認
# unique, counts = np.unique(y_train_extracted, return_counts=True)
# print("\n訓練データの分布:")
# for digit, count in zip(unique, counts):
#     print(f"数字 {digit}: {count}枚")

# # 抽出された画像をいくつか表示
# plt.figure(figsize=(10, 5))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(x_train_extracted[i], cmap='gray')
#     plt.title(f"Label: {y_train_extracted[i]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# データの正規化（オプション）
x_train_normalized = x_train_extracted.astype('float32') / 255
x_test_normalized = x_test_extracted.astype('float32') / 255

# print("\n正規化後のデータ範囲:")
# print(f"訓練データ: {x_train_normalized.min()} to {x_train_normalized.max()}")
# print(f"テストデータ: {x_test_normalized.min()} to {x_test_normalized.max()}")