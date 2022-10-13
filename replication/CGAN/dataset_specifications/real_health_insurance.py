import numpy as np
import pandas as pd 

from dataset_specifications.real_dataset import RealDataset
from sklearn import preprocessing as skpp

class RealHealthInsuranceSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "real_health_insurance"
        self.requires_path = False

        self.x_dim = 8
        self.support = (0.,1.)

        self.val_percent = 0.20
        self.test_percent = 0.20

        # California housing dataset
        # Full dataset is available at http://lib.stat.cmu.edu/datasets/houses.zip
        # Here it is loaded from sci-kit learn librbary, so no file has to
        # be manually downloaded

    def preprocess(self, file_path): 
        data = pd.read_csv("../../data/insurance.csv")

        num_feats = data.select_dtypes(['float64']).columns
        data[num_feats] = (data[num_feats] - data[num_feats].mean()) / data[num_feats].std()
        
        # df = df.sample(n=50000, random_state=43)
                
        data["sex"] = (data["sex"]=="male").astype(np.int64)
        data["smoker"] = (data["smoker"]=="yes").astype(np.int64)

        data["region_1"] = (data["region"]=="southwest").astype(np.int64)
        data["region_2"] = (data["region"]=="southeast").astype(np.int64)
        data["region_3"] = (data["region"]=="northeast").astype(np.int64)
        del data["region"]

        
        x = data.drop(columns="expenses").values
        y = data["expenses"].values
        print(x.shape, y.shape)
        
        loaded_data = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)

        # Standardize all data
        # loaded_data = skpp.StandardScaler().fit(loaded_data).transform(loaded_data)

        return loaded_data

