import pandas as pd

# XLSXファイルを読み込む
xlsx_path = 'emotion.xlsx'
df = pd.read_excel(xlsx_path)

# TSVファイルとして保存する
tsv_path = 'emotion.tsv'
df.to_csv(tsv_path, sep='\t', index=False)

