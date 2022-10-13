import numpy as np
import pandas as pd 

from dataset_specifications.real_dataset import RealDataset
from sklearn import preprocessing as skpp

class RealRetailSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "real_retail"
        self.requires_path = False

        self.x_dim = 10
        self.support = (0.,1.)

        self.val_percent = 0.20
        self.test_percent = 0.20

        # California housing dataset
        # Full dataset is available at http://lib.stat.cmu.edu/datasets/houses.zip
        # Here it is loaded from sci-kit learn librbary, so no file has to
        # be manually downloaded

    def preprocess(self, file_path): 
        data = pd.read_csv("../../data/retail_dataset.csv")

        num_feats = data.select_dtypes(['float64']).columns
        data[num_feats] = (data[num_feats] - data[num_feats].mean()) / data[num_feats].std()
        
        x = data.drop(columns="target").values
        y = data["target"].values
        print(x.shape, y.shape)
        
        loaded_data = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)

        # Standardize all data
        # loaded_data = skpp.StandardScaler().fit(loaded_data).transform(loaded_data)

        return loaded_data

