import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # solve: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度方向设置内向
plt.rcParams['ytick.direction'] = 'in'

np.random.seed(1)

class MixSpectraGenerator:

    def __init__(self, path1: str, path2: str):
        self.DftInfoSavePath = path1
        self.MixSpectra = path2
    
    def MixProcess_2_Component(self):
        mix_dft_spec_df = pd.DataFrame(columns=["ratio", 'mix_dft_spec_smooth'])
        n = 2000
        ratios = []
        with tqdm(total=n) as pbar:
            for i in range(n):
                pbar.update(1)
                rands = np.random.randint(low=1, high=49, size=2) # 原来的high=50

                # 防止生成NAN值
                if np.count_nonzero(rands) == 0:  # 如果非零元素个数为0， 即生成的数值全为0
                    continue
                ratio = rands / rands.sum()
                ratios.append(ratio)
                dft_specs = self.Get2ComponentSpec()
                print(len(dft_specs))  
                mix_dft_spec = np.dot(ratio, dft_specs)
                print(mix_dft_spec.shape) 

                if len(ratios) == 1001:
                    break

                mix_dft_spec_max = mix_dft_spec / mix_dft_spec.max()  # 先混合再统一尺度到0-1
                print(mix_dft_spec_max.shape) # (3320,)
                mix_dft_spec_max_smooth_10 = self.spec_smooth_10(mix_dft_spec_max)
                mix_dft_spec_df = mix_dft_spec_df.append(
                    {"ratio": ratio.tolist(), "mix_dft_spec_smooth": mix_dft_spec_max_smooth_10},
                    ignore_index=True)
        
        mix_dft_spec_df.to_csv(self.MixSpectra, sep="\t", encoding="utf-8", index=False)


    def Get2ComponentSpec(self):
        dft_broden_spec = pd.read_csv(self.DftInfoSavePath, sep='\t', encoding='utf-8')
        dft_spectras = []
        for _, row in dft_broden_spec.iterrows():
        #     print(row['log_path'])
        #     print(row['log_path'].split('\\')[-1])
            info = row['log_path'].split('\\')[-1]
            if info == "B.log" or info == "C.log":  # 只加入B\C
            # if info == "B_acid.log" or info == "C.log":   # 只加入B\C
            # if info == "B.log" or info == "C.log" or info == "D.log":   # 加入B\C\D
                dft_spectras.append(eval(row["spectra"])) 
                print(f"已加入{info}对应的光谱")   # 顺序：已加入B_acid.log对应的光谱;已加入C.log对应的光谱
        
        return dft_spectras


    def GetCombinatios(self):
        dft_info = pd.read_csv(self.DftInfoSavePath, sep='\t', encoding='utf-8')
        dft_info['spectra'] = np.array(dft_info['spectra'].map(eval))
        all_list = list(range(5))  # 5种组分编号[0, 1, 2, 3, 4]
        combis_list = []
        combis = []
        for i in range(2, 6):  # 混合2-5种组分
            for j in combinations(all_list, i):
                combis_list.append(list(j))   # 只是用于dft_info的提取
                combis.append(np.array(j))
        dft_spectras = []
        for i, combi in enumerate(combis_list):
            dft_spec = dft_info.loc[combi, 'spectra'].values.tolist()
            dft_spectras.append(np.array(dft_spec))   # 构建一个numpy.array格式的类型数据，更加方便模型进行矩阵乘法
        return combis, dft_spectras

    def cal(freqs, intens):
        calc = 0
        for i in range(len(intens)-1):
            calc += intens[i] * (freqs[i + 1] - freqs[i])
        return calc

    def show(self, x, y, y_scale_calc, label1, label2, ratio):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, y, label=f"{label1}")
        ax.plot(x, y_scale_calc, label=f'{label2}')
        plt.xlabel("freqs")
        plt.ylabel("intens")
        plt.title(f"混合物IR：{ratio}")
        plt.legend(loc='upper right')
        plt.show()

    def spec_smooth_10(self, spec):
        spec = np.array(spec)
        length = len(spec)
        avg_spec = []
        for i in range(5, length - 5):  # 从第6个数据开始取平均，到倒数第6个
            avg_list = spec[i - 5:i + 6]  # 因为左闭右开，因此要+6
            avg_item = sum(avg_list) / 11
            avg_spec.append(avg_item)
        avg_spec.extend(spec[-5:])
        avg_spec = list(spec[:5]) + avg_spec
        return avg_spec

    def GetDftFromES(espath: str):  # ES: extract_spectra
        es_df = pd.read_csv(espath, sep='\t', encoding='utf-8')
        spectras = []
        for _, row in es_df.iterrows():
            # print("row['spectra']:\t", row["spectra"])
            # print("eval(row['spectra']):\t", eval(row["spectra"]))
            spectras.append(eval(row["spectra"]))  # eval()    [[h-1], [h-2], [h-3]]
        return np.array(spectras)

    def GetSyInfo(SyInPath: str):   # 最初小龙师兄的9个实验光谱
        sy_spectra_df = pd.read_csv(SyInPath)
        # freqs = sy_spectra_df.iloc[:, 0].values[208:3216]  # 600-3500
        freqs = sy_spectra_df.iloc[:, 0].values
        # reac_spectra = sy_spectra_df['0-10'].values[208:3216]  # 600-3500
        reac_spectra = sy_spectra_df['0-10'].values
        reac_spectra = reac_spectra / 100
        reac_spectra = np.log10((1 / reac_spectra))  # 透射率转换为吸光度
        # pro_spectra = sy_spectra_df['10-0'].values[208:3216]
        pro_spectra = sy_spectra_df['10-0'].values
        pro_spectra = pro_spectra / 100
        pro_spectra = np.log10((1 / pro_spectra))
        sy_spectra = np.array([pro_spectra, reac_spectra])
        return freqs, sy_spectra


if __name__ == "__main__":
    MSG = MixSpectraGenerator("./data/gaussian_output/dft_info.tsv", 
                              "./data/two-component_dataset-BC/Pre-train/Pre-train_model_dataset.tsv"
    )
    MSG.MixProcess_2_Component()
    print("process complete!")



