# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExpProcess:
    def __init__(self, dft_data_path, exp_data_path, save_path):
        self.dft_data_path = dft_data_path  # './two-component/dft_info-B-1.tsv' ；'./two-component/dft_info-B-2.tsv';'./three-component/dft_info-B-1.tsv'
        self.exp_data_path = exp_data_path  # "./two_component/exp_spec_B1"；"./two_component/exp_spec_B2";"./three_component/exp_spec_B1"  
        self.save_path = save_path

    def load_data(self):
        exp_csv_files = os.listdir(self.exp_data_path)
        for i,csv in enumerate(exp_csv_files):
            csv_path = os.path.join(self.exp_data_path, csv)
            csv_data = pd.read_csv(csv_path, header=None)
            list_0 = csv_data.iloc[:, 1].values[0]
            list_1 = csv_data.iloc[:, 1].values[-1]
            if list_1 != 0:
                print(f"{csv}对应光谱的最后一个强度值不为0") 
            if list_0 == 0:
                print(f"{csv}对应的光谱第一个强度值为0") 
        dft_broden_spec = pd.read_csv(self.dft_data_path, sep='\t', encoding='utf-8')
        dft_spectras = []
        for _, row in dft_broden_spec.iterrows():
        #     print(row['log_path'])
        #     print(row['log_path'].split('\\')[-1])
            info = row['log_path'].split('\\')[-1]
            if info == "B.log" or info == "C.log":   # two component B\C
            # if info == "B_acid.log" or info == "C.log":   # two component B2\C
            # if info == "B.log" or info == "C.log" or info == "D.log":   # three component B\C\D
                dft_spectras.append(eval(row["spectra"]))  # eval()    [target_spec, reactant_spec]
                print(f"add {info}-sepc")
        return exp_csv_files, dft_spectras
    
    #  ================== process data code ======================
    def cal(freqs, intens):
        calc = 0
        for i in range(len(freqs)-1):
            calc += intens[i] * (freqs[i + 1] - freqs[i])
        return calc

    def spec_smooth_10(spec):
        spec = np.array(spec)
        length = len(spec)
        avg_spec = []
        for i in range(5, length-5): 
            avg_list = spec[i-5:i+6]  
            avg_item = np.sum(avg_list) / 11
            avg_spec.append(avg_item)
        avg_spec.extend(spec[-5:])
        avg_spec = list(spec[:5]) + avg_spec
        return avg_spec
    #  ================== process data code end======================

    def process_data(self):
        exp_csv_files, dft_spectras = self.load_data()
        # exp_data = pd.DataFrame(columns=['file_name', 'ratio_1', 'ratio_2', 'ratios', 'exp_freq', 'exp_spec', 'spec_max', 'spec_calc', 
        #                                  'spec_max_smooth_10',  'dft_spec_max', 'dft_spec_max_smooth_10'])
        exp_data = pd.DataFrame(columns=['file_name', 'ratio_1', 'ratio_2', 'ratio_3', 'ratios', 'exp_freq', 'exp_spec', 'spec_max', 'spec_calc', 
                                 'spec_max_smooth_10', 'dft_spec_max', 'dft_spec_max_smooth_10'])
        for i,csv in enumerate(exp_csv_files):
            csv_path = os.path.join(self.exp_data_path, csv)
            csv_data = pd.read_csv(csv_path, header=None)
            splites = csv.rsplit(".", 1)[0].split('-')
            
            # two-component ratios info
            ratio_1 = int(splites[0])   # B
            ratio_2 = int(splites[1])   # C
            ratios = [ratio_1/50, ratio_2/50]

            # three-component ratios info
            # ratio_1 = int(splites[0])   # B
            # ratio_2 = int(splites[1])   # C
            # ratio_3 = float(splites[2])   # D
            # sum = ratio_1 + ratio_2 + ratio_3
            # ratios = [ratio_1/sum, ratio_2/sum, ratio_3/sum]  # [B, C, D] 比例

            freqs = csv_data.iloc[:, 0].values[:-1].tolist()
            initial_spec = csv_data.iloc[:, 1].values[:-1] 
        #     print(len(freqs))
        #     print(dft_ratios)
            spec = initial_spec / 100
            spec = np.log10((1 / spec))
            spec_max = spec/spec.max() 
            spec_max_smooth_10 = self.spec_smooth_10(spec_max)
            spec_calc = spec / self.cal(freqs, spec)
            dft_spec = np.dot(ratios, dft_spectras)  # dft_ratio[C, B] * spec[C, B_acid]
            dft_spec_max = dft_spec / dft_spec.max()
            dft_spec_max_smooth_10 = self.spec_smooth_10(dft_spec_max)
            exp_data = exp_data.append({'file_name': csv,  'ratio_1': ratio_1, 'ratio_2':ratio_2, 'ratios':ratios, 
                                        # 'ratio_3':ratio_3,  # add in three-component
                                        'exp_freq': freqs, 'exp_spec':initial_spec.tolist(), 'spec_max': spec_max, 'spec_calc': spec_calc,
                                        'spec_max_smooth_10':spec_max_smooth_10, 'dft_spec_max': dft_spec_max, 
                                        'dft_spec_max_smooth_10': dft_spec_max_smooth_10}, ignore_index=True)
        
        exp_data_path = os.path.join(self.save_path, 'all_exp_data.csv')
        exp_data.to_csv(exp_data_path, sep="\t", encoding="utf-8", index=False)
        print("-------------------------- Process and Save data complete! -----------------------------")
        

if __name__ == "__main__":
    dft_data_path = './data/two-component_dataset-BC/dft_info-B-1.csv'
    exp_data_path = './data/two-component_dataset-BC/exp_spec_B1'
    save_path = './data/two-component_dataset-BC/'
    start_time = time.time()
    exp_process = ExpProcess(dft_data_path, exp_data_path, save_path)
    exp_process.process_data()
    end_time = time.time()
    logging.info(f"任务完成，耗时：{end_time - start_time:.2f}秒")
