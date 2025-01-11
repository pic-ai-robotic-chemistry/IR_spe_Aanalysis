# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import torch
import logging 
import joblib  # 用于保存和加载scikit-learn模型
# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SP: 
    def __init__(self, path1, path2,  align_image_save_dir, align_pretrain_image_save_dir, 
                 model_save_dir, predictions_save_dir, repeats=5):
        self.MixSpectraPath = path1
        self.PreTrainModelPath = path2
        self.AlignImageSaveDir = align_image_save_dir
        self.AlignPretrainImageSaveDir = align_pretrain_image_save_dir
        self.ModelSaveDir = model_save_dir
        self.PredictionsSaveDir = predictions_save_dir
        self.repeats = repeats

        os.makedirs(self.AlignImageSaveDir, exist_ok=True)
        os.makedirs(self.AlignPretrainImageSaveDir, exist_ok=True)
        os.makedirs(self.ModelSaveDir, exist_ok=True)
        os.makedirs(self.PredictionsSaveDir, exist_ok=True) 

    def load_data(self):
        # load data
        try:
            mix_df = pd.read_csv(self.MixSpectraPath, sep='\t', encoding='utf-8')
        except Exception as e:
            print(f"erro：{e}")
            return None, None, None, None
        
        mix_df['spec_max_smooth_10'] = mix_df['spec_max_smooth_10'].map(eval)  # or import ast  .map(ast.literal_eval)
        mix_df['dft_spec_max_smooth_10'] = mix_df['dft_spec_max_smooth_10'].map(eval)
        
        # split data
        X = np.array(mix_df['spec_max_smooth_10'].tolist())
        y = np.array(mix_df['dft_spec_max_smooth_10'].tolist())

        ratio_1 = mix_df['ratio_1'].values.tolist()
        ratio_2 = mix_df['ratio_2'].values.tolist()
        ratios = [np.array([r1, r2]) for (r1, r2) in zip(ratio_1, ratio_2)]
        P = np.array([item / item.sum() for item in ratios])

        indices = mix_df.index.to_numpy()
        return mix_df, X, y, P, indices  # get exp_data theo_data propotions
    
    
    def run_experiments_and_save_results(self, X, y, P, indices):
        train_ratio = 0.2  # 20%-train_validate_Align-model，80%-test_Fused-model

        # load Pre-trained-model
        if os.path.exists(self.PreTrainModelPath):
            PreTrainModel = torch.load(self.PreTrainModelPath)
            logging.info(f"successful load：{self.PreTrainModelPath}")
        else:
            logging.error(f"{self.PreTrainModelPath} not exist, can't load.")
            return

        for rep in range(1, self.repeats + 1):
            logging.info(f'======================= repate process:{rep}/{self.repeats} =======================')
            X_train_full, X_test, y_train_full, y_test, P_train_full, P_test, idx_train_full, idx_test = train_test_split(
                X, y, P, indices, train_size=train_ratio, shuffle=True, random_state=None
            )
            logging.info(f"len(train_valid_dataset)：{len(X_train_full)}, len(test_dataset)：{len(X_test)}")

            X_train, X_eval, y_train, y_eval, idx_train, idx_eval = train_test_split(
                X_train_full, y_train_full, idx_train_full, train_size=0.8, shuffle=True, random_state=1
            )
            logging.info(f"len(Align_train_dataset)：{len(X_train)}，len(Align_valid_dataset)：{len(X_eval)}")
            
            try:
                # AlignModel: 
                AlignModel = RFR(n_estimators=200, max_depth=4, random_state=1, n_jobs=-1) 
                AlignModel.fit(X_train, y_train)
                logging.info(f"AlignModel training complete")

                model_save_path = os.path.join(self.ModelSaveDir, f"AlignModel_repeat_{rep}.joblib")
                joblib.dump(AlignModel, model_save_path)
                logging.info(f"save AlignModel to {model_save_path}")

                y_pred_eval = AlignModel.predict(X_eval)
                r2_align = r2_score(y_eval, y_pred_eval)
                logging.info(f"R²_Aligned：{r2_align:.4f}")

                logging.info(f"True y shape: {y_eval.shape}")
                logging.info(f"Predicted y shape: {y_pred_eval.shape}")
                
                eval_true_df = pd.DataFrame(y_eval, columns=[f"Feature_{i+1}" for i in range(y_eval.shape[1])])
                eval_true_df.insert(0, 'idx', idx_eval)
                eval_true_df.insert(1, 'Type', 'True')

                eval_pred_df = pd.DataFrame(y_pred_eval, columns=[f"Feature_{i+1}" for i in range(y_pred_eval.shape[1])])
                eval_pred_df.insert(0, 'idx', idx_eval)
                eval_pred_df.insert(1, 'Type', 'Predicted')

                align_eval_results = pd.concat([eval_true_df, eval_pred_df], ignore_index=True)
                eval_csv_path = os.path.join(self.PredictionsSaveDir, f"align_model_eval_predictions_repeat_{rep}.csv")
                align_eval_results.to_csv(eval_csv_path, index=False)
                logging.info(f"save Align_validate value to: {eval_csv_path}")

                
                # plot Aligned-validate results
                plt.figure(figsize=(8, 6))
                plt.scatter(y_eval, y_pred_eval, alpha=0.6, s=2, edgecolors='none') 
                plt.grid(False)
                min_val = min(y_eval.min(), y_pred_eval.min())
                max_val = max(y_eval.max(), y_pred_eval.max())
                range_val = max_val - min_val
                plt.xlim(min_val - 0.05*range_val, max_val + 0.05*range_val)
                plt.ylim(min_val - 0.05*range_val, max_val + 0.05*range_val)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
                plt.xlabel("True y")
                plt.ylabel("Predicted y")
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_visible(True)
                plt.tight_layout()
                align_image_path = os.path.join(self.AlignImageSaveDir, f"align_true_vs_pred_repeat_{rep}.png")
                plt.savefig(align_image_path, dpi=300)
                plt.close()
                logging.info(f"save AlignModel plot to:{align_image_path}")

                # Fused model in test_data
                y_pred_test = AlignModel.predict(X_test)
                P_pred = PreTrainModel.predict(y_pred_test)
                r2_align_pretrain = r2_score(P_test, P_pred)
                logging.info(f"R²_Fused ：{r2_align_pretrain:.4f}")

                test_df = pd.DataFrame({
                    'idx': idx_test,
                    'true_p1': P_test[:, 0],
                    'pred_p1': P_pred[:, 0],
                    'true_p2': P_test[:, 1],
                    'pred_p2': P_pred[:, 1]
                })
                test_csv_path = os.path.join(self.PredictionsSaveDir, f"Fused_test_predictions_repeat_{rep}.csv")
                test_df.to_csv(test_csv_path, index=False)
                logging.info(f"save Fused_test value to: {test_csv_path}")

                
                # plot Fused-test results
                plt.figure(figsize=(8, 6))
                all_values = np.concatenate([P_test, P_pred])
                p_min, p_max = all_values.min(), all_values.max()
                p_range = p_max - p_min
                plt.scatter(P_test[:, 0], P_pred[:, 0], alpha=0.8, s=5, edgecolors='none', label='Ratio 1')
                plt.scatter(P_test[:, 1], P_pred[:, 1], alpha=0.8, s=5, edgecolors='none', label='Ratio 2')
                plt.grid(False)
                plt.xlim(p_min - 0.05*p_range, p_max + 0.05*p_range)
                plt.ylim(p_min - 0.05*p_range, p_max + 0.05*p_range)
                plt.plot([p_min, p_max], [p_min, p_max], 'r--', linewidth=1)
                plt.xlabel("True P")
                plt.ylabel("Predicted P")
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_visible(True)
                plt.legend()
                plt.tight_layout()
                align_pretrain_image_path = os.path.join(self.AlignPretrainImageSaveDir, f"align_pretrain_true_vs_pred_repeat_{rep}.png")
                plt.savefig(align_pretrain_image_path, dpi=300)
                plt.close()
                logging.info(f"save Fused plot to: {align_pretrain_image_path}")

            except Exception as e:
                logging.error(f"repeat {rep}-erro: {e}")
                continue
    
    def compute_learning_curve(self, X, y, P, indices):
        self.run_experiments_and_save_results(X, y, P, indices)


if __name__ == "__main__":
    path = './data/two-component_dataset-BC'
    predictor = SP(
        path1=os.path.join(path, "all_exp_data.csv"),
        path2=os.path.join(path, "Pre-train/Pre-trained_model-B1-rfr-1000.pth"),
        align_image_save_dir=os.path.join(path, "Fused/align_plots"),
        align_pretrain_image_save_dir=os.path.join(path, "Fused/Fused_plots"),
        model_save_dir=os.path.join(path, "Fused/align_models"),
        predictions_save_dir=os.path.join(path, "PredictionsSaveDir"),
        repeats=5      
    ) 
    
    start_time = time.time()
    mix_df, X, y, P, indices = predictor.load_data()
    if X is not None:
        logging.info(f"len(X): {len(X)}")
        logging.info(f"len(y): {len(y)}")
        logging.info(f"len(P): {len(P)}")
        logging.info(f"len(indices): {len(indices)}")
        predictor.compute_learning_curve(X, y, P, indices)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Predict complete in {elapsed_time:.2f} seconds")
        print("Predict complete!")
    else:
        print("----LOAD DATA ERRO!-----")
    


