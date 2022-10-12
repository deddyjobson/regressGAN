import numpy as np

from dataset_specifications.real_dataset import RealDataset
from sklearn import preprocessing as skpp
from sklearn.datasets import fetch_openml

from sklearn.preprocessing import OneHotEncoder

class RealInsuranceSet(RealDataset):
    def __init__(self):
        super().__init__()
        self.name = "real_insurance"
        self.requires_path = False

        self.x_dim = 8
        self.support = (0.,1.)

        self.val_percent = 0.10
        self.test_percent = 0.10

        # California housing dataset
        # Full dataset is available at http://lib.stat.cmu.edu/datasets/houses.zip
        # Here it is loaded from sci-kit learn librbary, so no file has to
        # be manually downloaded

    def preprocess(self, file_path): 
        df = fetch_openml(data_id=41214, as_frame=True)["data"].dropna()
        
        for feat in df.select_dtypes(['category']).columns:
            df[feat] = df[feat].values.codes
        
        for feat in df.columns:
            num_new_feats = len(df[feat].unique())
            if df[feat].dtype != "float64":
                new_feats = OneHotEncoder(sparse=False).fit_transform(df[feat].values.reshape(-1,1))
                for idx in range(num_new_feats):
                    df[f"{feat}_{idx}"] = new_feats[:,idx]
                    df[f"{feat}_{idx}"] = df[f"{feat}_{idx}"].astype("float")
                del df[feat]

        print(df.dtypes)
        x = df.drop(columns="ClaimNb").values
        y = df["ClaimNb"].values
        
        loaded_data = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)

        # Standardize all data
        loaded_data = skpp.StandardScaler().fit(loaded_data).transform(loaded_data)

        return loaded_data

