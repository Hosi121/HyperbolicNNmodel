from gensim.test.utils import datapath
# Read the sample relations file and train the model
filepath = datapath('../data/emotion.tsv')
relations = PoincareRelations(filepath)
model = PoincareModel(train_data=relations)
model.train(epochs=50)
# Which words are most similar to 'kangaroo'?
model.kv.most_similar('kangaroo.n.01', topn=2)