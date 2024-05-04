from gensim.models.poincare import PoincareModel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

class EmotionModel:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.model = None
        self.vector_history = []

    def load_data(self):
        with pd.ExcelFile(self.file_path) as xls:
            df = pd.read_excel(xls, self.sheet_name)
        self.relations = list(df[['Word', 'Emotion']].itertuples(index=False, name=None))

    def train_model(self, size=5, negative=2, epochs=5):
        self.model = PoincareModel(self.relations, size=size, negative=negative)
        for epoch in range(epochs):
            self.model.train(epochs=1)
            vectors = [self.model.kv.get_vector(word) for word in self.model.kv.index_to_key]
            self.vector_history.append(vectors)

    def plot_animation(self):
        font_path = r'C:\Users\takuy\AppData\Local\Microsoft\Windows\Fonts\ipaexg.ttf'
        prop = FontProperties(fname=font_path)
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            pca = PCA(n_components=2)  # PCAをフレームごとに実行
            transformed_vectors = pca.fit_transform(self.vector_history[frame])  # このエポックのデータでPCA
            ax.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1])
            for i, word in enumerate(self.model.kv.index_to_key):
                ax.text(transformed_vectors[i, 0], transformed_vectors[i, 1], word, fontproperties=prop)
            ax.set_title(f'Epoch {frame + 1}', fontproperties=prop)
            ax.grid(True)

        ani = FuncAnimation(fig, update, frames=len(self.vector_history), repeat=False)
        ani.save('poincare_training.gif', writer='imagemagick')

if __name__ == "__main__":
    emotion_model = EmotionModel('data/emotion.xlsx', 'C')
    emotion_model.load_data()
    emotion_model.train_model(epochs=10)
    emotion_model.plot_animation()
