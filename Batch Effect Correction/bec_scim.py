# Environment settings
import scanpy as sc
sc.set_figure_params(dpi=100)

import warnings
warnings.filterwarnings('ignore')

from src.scimilarity.utils import lognorm_counts
from src.scimilarity import CellAnnotation, align_dataset, CellQuery

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report

# import CellQuery object
annotation_path = './models/annotation_model_v1'
query_path = './models/query_model_v1'
cq = CellQuery(model_path=annotation_path,
               cellsearch_path=query_path)

# Load the tutorial data.
# Replace data_path with your local file path.
data_path = "/gpfs/gibbs/pi/zhao/tl688/largedataset/singlecellimmune_covid.h5ad"
adams = sc.read(data_path)
adams.layers['counts'] = adams.X
# adams.var_names = [i.upper() for i in adams.var_names]

adams.var_names = list(adams.var['feature_name'])

adams = align_dataset(adams, cq.gene_order)

adams = lognorm_counts(adams)

adams.obsm['X_scimilarity'] = cq.get_embeddings(adams.X)

adams.write_h5ad("/gpfs/gibbs/pi/zhao/tl688/scimilarity/bec_result/singlecellimmune_covid_sim.h5ad")