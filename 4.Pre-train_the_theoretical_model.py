import os.path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def MAE(target, predict):
    return (abs(target-predict)).mean()


def RMSE(target,predict):
    return np.sqrt(((predict - target) ** 2).mean())

class PreTrainModel:

    def __init__(self, path1: str, path2: str):
        self.MixSpectraPath = path1
        self.ModelPath = path2

    def process3component(self):
        mix_dft_spec = pd.read_csv(self.MixSpectraPath, sep='\t', encoding='utf-8')
        print("len(mix_dft_spec):\t",len(mix_dft_spec)) 
        mix_dft_spec['ratio'] = mix_dft_spec['ratio'].map(eval)
        mix_dft_spec['mix_dft_spec_smooth'] = mix_dft_spec['mix_dft_spec_smooth'].map(eval)
        train_df, test_df = train_test_split(mix_dft_spec, test_size=0.1, random_state=1, shuffle=True) 
        # print('test_df["ratio"].values.tolist():', test_df["ratio"].values.tolist())
        train_x, train_y = np.array(train_df["mix_dft_spec_smooth"].values.tolist()), \
            np.array(train_df["ratio"].values.tolist())
        test_x, test_y = np.array(test_df['mix_dft_spec_smooth'].values.tolist()), \
            np.array(test_df["ratio"].values.tolist())
        print("len(test_y):\t", len(test_y)) 
        
        """train and save model"""
        random_seed = 1
        model = RFR(n_estimators=300, max_depth=6, random_state=random_seed)
        model_y = model.fit(train_x, train_y)
        torch.save(model_y, self.ModelPath)
        pred_y = model_y.predict(test_x)
        """end for train & save model"""

        mae_dict = {}
        rmse_dict = {}
        ratio_labels = ['B', 'C', 'D']
        for i, str in enumerate(ratio_labels): 
            real_ratio = [item[i] for item in test_y]
            pred_ratio = [item[i] for item in pred_y]
            print(f"R2_{str}: %.5f" % r2_score(real_ratio, pred_ratio))
            mae = MAE(np.array(real_ratio), np.array(pred_ratio))
            rmse = RMSE(np.array(real_ratio), np.array(pred_ratio))
            mae_dict[str] = mae
            rmse_dict[str] = rmse

        print('mae_dict:\t', mae_dict)
        print('rmse_dict:\t', rmse_dict)

    def process2component(self):
        mix_dft_spec = pd.read_csv(self.MixSpectraPath, sep='\t', encoding='utf-8')
        print("len(mix_dft_spec):\t",len(mix_dft_spec))   
        mix_dft_spec['ratio'] = mix_dft_spec['ratio'].map(eval)
        mix_dft_spec['mix_dft_spec_smooth'] = mix_dft_spec['mix_dft_spec_smooth'].map(eval)
        train_df, test_df = train_test_split(mix_dft_spec, test_size=0.1, random_state=1, shuffle=True) 
        # print('test_df["ratio"].values.tolist():', test_df["ratio"].values.tolist())
        train_x, train_y = np.array(train_df["mix_dft_spec_smooth"].values.tolist()), \
            np.array(train_df["ratio"].values.tolist())
        test_x, test_y = np.array(test_df['mix_dft_spec_smooth'].values.tolist()), \
            np.array(test_df["ratio"].values.tolist())
        print("len(test_y):\t", len(test_y))  
        

        """train and save model"""
        random_seed = 1
        model = RFR(n_estimators=300, max_depth=6, random_state=random_seed)
        model_y = model.fit(train_x, train_y)
        torch.save(model_y, self.ModelPath)
        pred_y = model_y.predict(test_x)
        """end for train & save model"""

        mae_dict = {}
        rmse_dict = {}
        ratio_labels = ['B2', 'C']
        for i, str in enumerate(ratio_labels): 
            real_ratio = [item[i] for item in test_y]
            pred_ratio = [item[i] for item in pred_y]
            print(f"R2_{str}: %.5f" % r2_score(real_ratio, pred_ratio))
            mae = MAE(np.array(real_ratio), np.array(pred_ratio))
            rmse = RMSE(np.array(real_ratio), np.array(pred_ratio))
            mae_dict[str] = mae
            rmse_dict[str] = rmse
            plt.figure(dpi=300)  
            plt.scatter(real_ratio, pred_ratio, s=5)  
            plt.xlabel(f"real r({str})") 
            plt.ylabel(f"pred r({str})")
            name = f'CP_model fitting result(ration_{str}).png'
            save_path = os.path.join(self.ImageSavePath, name)
            plt.savefig(save_path)
            plt.show()

        print('mae_dict:\t', mae_dict)
        print('rmse_dict:\t', rmse_dict)


if __name__ == "__main__":
    testmodel: PreTrainModel = PreTrainModel(
        "./data/two-component_dataset-BC/Pre-train/Pre-train_model_dataset.tsv", # PATH1: dataset to train CP_model
        './data/two-component_dataset-BC/Pre-train/Pre-trained_model.pth',   # PATH2: Save the trained CP_model
        )
    print('Begin predict...')
    # testmodel.process2component()
    testmodel.process3component()
    print("process complete")



