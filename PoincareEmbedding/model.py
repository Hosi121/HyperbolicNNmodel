import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareModel
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

class EmotionModel:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.model = None

    def load_data(self):
        # Excelファイルを開く
        with pd.ExcelFile(self.file_path) as xls:
            # 指定されたシートをDataFrameとして読み込む
            df = pd.read_excel(xls, self.sheet_name)
        # 'Word' と 'Emotion' 列を使用してタプルのリストを作成
        self.relations = list(df[['Word', 'Emotion']].itertuples(index=False, name=None))

    def train_model(self, size=2, negative=2, epochs=5):
        # モデルをインスタンス化
        self.model = PoincareModel(self.relations, size=size, negative=negative)
        # モデルを訓練
        self.model.train(epochs=epochs)

    def get_vectors(self):
        # 訓練後の各単語のベクトルを取得して表示
        if self.model:
            for word in self.model.kv.index_to_key:
                print(f"{word}: {self.model.kv.get_vector(word)}")
        else:
            print("Model is not trained yet.")

    def plot_vectors(self, save_path='emotion_plot.png'):
        # ベクトルデータをプロット
        if self.model:
            # フォントプロパティの設定
            font_path = r'C:\Users\takuy\AppData\Local\Microsoft\Windows\Fonts\ipaexg.ttf'
            prop = FontProperties(fname=font_path)

            vectors = [self.model.kv.get_vector(word) for word in self.model.kv.index_to_key]
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(vectors)
        
            plt.figure(figsize=(8, 6))
            for i, word in enumerate(self.model.kv.index_to_key):
                plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], marker='o')
                plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=9, fontproperties=prop)
            plt.title('Poincare Embedding Visualization (2D Projection)', fontproperties=prop)
            plt.xlabel('PCA 1', fontproperties=prop)
            plt.ylabel('PCA 2', fontproperties=prop)
            plt.grid(True)
            plt.savefig(save_path)  # プロットを指定したパスに保存
            plt.show()
        else:
            print("Model is not trained yet or there are no vectors to plot.")


# 使用例
if __name__ == "__main__":
    emotion_model = EmotionModel('data/emotion.xlsx', 'C')
    emotion_model.load_data()
    emotion_model.train_model(size=5)  # ここで埋め込み次元を5に設定
    emotion_model.get_vectors()
    emotion_model.plot_vectors(save_path='emotion_plot.png')