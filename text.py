import re

# 変換対象の文字列
input_string = """
 [[ -0.26144732  -6.25823983  -1.7932414    0.28334412  -6.13055345
    0.19124223  -1.90881357  -9.6388685 ]
 [ -0.19476153   6.59676074 -14.88079773  16.68429364   6.1584592
    0.1678363    3.76250841  -7.4896625 ]
 [  0.68961213  -0.2382437    6.4494803   -8.44414947   0.20734814
    0.13888404  -0.70994735   8.53983402]]
"""

# 数値の間にカンマを追加
output_string = re.sub(r'(\d)\s+(-?\d)', r'\1, \2', input_string)

# 角括弧の間にカンマを追加
output_string = re.sub(r'\]\s*\[', r'], [', output_string)

# 変換結果を表示
print(output_string)