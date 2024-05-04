from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
relations = PoincareRelations(file_path=datapath('/emotion1.tsv'))
model = PoincareModel(train_data=relations, size=2)
model.train(epochs=50)

import gensim.viz.poincare
import ployly.offline

plotly.offline.init_notebook_mode(connected=False)
prefecutre_map = gensim.viz.poincare.poincare_2d_visualization(model=model,
                                                               tree=relations,
                                                               figure_title="可視化",
                                                               num_nodes=10,
                                                               show_node_labels=model.kv.vocab.keys())
plotly.offline.iplot(prefecutre_map)
