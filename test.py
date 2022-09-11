from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing
from river import tree
from tree.ORTO import ORTO
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")


"""
#############
Bikes dataset
#############
"""

# data_Bikes = datasets.Bikes()

# catag_dict = {'station':[],'description':[]}

# for x,y in data_Bikes:
#     catag_dict['station'].append(x['station'])
#     catag_dict['description'].append(x['description'])

# oe = OrdinalEncoder()
# catag_encoded = oe.fit_transform(pd.DataFrame.from_dict(catag_dict))  # array of shape (nrows,2)

# dataset = []
# i=0
# for x,y in data_Bikes:
#     del x['moment']
#     x['station'] = catag_encoded[i,0]
#     x['description'] = catag_encoded[i,1]
#     dataset.append((x,y))
#     i+=1



"""
#############
TrumpApproval dataset
#############
"""

dataset = datasets.TrumpApproval()



"""
#############
Hoeffding tree regressor 
#############
"""

model = (
    preprocessing.StandardScaler() |
    tree.HoeffdingTreeRegressor(
    grace_period=100,
    leaf_prediction='mean',
    min_samples_split=20
    #model_selector_decay=0.9
    )
)



"""
#############
ORTO
#############
"""


# model = (
#     preprocessing.StandardScaler() |
#     ORTO(
#     grace_period=100,
#     leaf_prediction='mean',
#     min_samples_split=20
#     #model_selector_decay=0.9
#     )
# )



metric = metrics.MAE()

evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric,print_every=100)
