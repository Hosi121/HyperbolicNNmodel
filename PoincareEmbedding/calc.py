from math import comb, pow

def calculate_next_element(a, n, j):
    a_next = a[1]
    for i in range(1, n+1):
        for k in range(1, j+1):
            term = comb(j, k) * pow(a[i], (j-k)/j)
            a_next += term
    return a_next

# 初期設定
a = [0] * 11  # 例として10個の要素を計算します
a[1] = 1
j = 4  # jの値を3とします

# 数列の要素を計算
for i in range(1, 10):
    a[i+1] = calculate_next_element(a, i, j)

# 結果の出力
for index, value in enumerate(a[1:], 1):
    print(f"a[{index}] = {value}")

