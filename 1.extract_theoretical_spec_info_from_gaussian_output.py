import linecache
import os
import pandas as pd
import torch
from tqdm import tqdm
import re
import random
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rc('axes', unicode_minus=False)

class ExtractSpectraInfo:

    def __init__(self, path1: str, path2: str):
        self.SpectraInfoPath = path1  # gaussian result path
        self.SpectraLogInfoSavePath = path2   # extract freq & intensity info and save to a csv
        # self.SpectraTxtInfoSavePath = path3   # extract x y info from gaussian-generated-txt to a csv

    def SaveLogInfoProcess(self):
        """从log文件中获取信息"""
        logs = []
        for file in os.listdir(self.SpectraInfoPath):
            if os.path.splitext(file)[1] == ".log":
                # if os.path.splitext(file)[0] == "B_acid":  # 进行品频哪醇的信息提取
                if os.path.splitext(file)[0] == "B":  # 进行苯硼酸的信息提取
                    continue
                else:
                    logs.append(file)
                
        print("logs: \t", logs)
        log_paths = [os.path.join(self.SpectraInfoPath, log) for log in logs]  # 获取每个文件的路径
        print(("log_path:\t", log_paths))
        spectra_df = pd.DataFrame(columns=["log_path", "freqs", "spectra"])  # 构建一个保存展宽后光谱信息的df
        with tqdm(total=len(log_paths)) as pbar:  # 2 是样本的个数
            for log_path in log_paths:
                pbar.update(1)
                # print("log_path:\t", log_path)
                freqs, intens = ExtractSpectraInfo.ExtractLogSpectra(log_path)  # 第一步完成log信息提取
                spectra_x, spectra_y = ExtractSpectraInfo.lor_process(freqs, intens)  # 第二步进行展宽处理
                plt.plot(spectra_x, spectra_y)
                plt.legend({log_path})
                plt.show()
                # print("log_path:\t", log_path)
                # print("spectra_y:\t", spectra_y)
                spectra_df = spectra_df.append({"log_path": log_path, "freqs": spectra_x, "spectra": spectra_y},
                                               ignore_index=True)
                # 保存x信息
                print("spectra_df_log: \t", spectra_df)
        spectra_df.to_csv(self.SpectraLogInfoSavePath, sep="\t", encoding="utf-8", index=False)

    @staticmethod
    def ExtractLogSpectra(log_path: str):  # 提取高斯文件中的光谱信息，对其中一个文件的处理
        freqs = []
        intens = []
        pattern_space = r"\s+"
        lines = linecache.getlines(log_path)
        meet_ir = False
        for line in lines:
            line = line.strip()
            if line.startswith('Harmonic frequencies'):
                meet_ir = True
            if not meet_ir:
                continue
            if line.startswith("Frequencies --"):
                for freq in re.split(pattern_space, line)[2:]:
                    # print(freq)
                    # print(float(freq))
                    freqs.append(float(freq))
            if line.startswith("IR Inten    --"):
                for inten in re.split(pattern_space, line)[3:]:
                    intens.append(float(inten))
        return freqs, intens

    @staticmethod
    def lor_process(fs: [float], ins: [float], sigma: float = 10.0):
        fs = torch.tensor(fs, dtype=torch.float)
        print("fs.shape:\t", fs.shape)
        ins = torch.tensor(ins, dtype=torch.float)
        print(("ins.shape；\t", ins.shape))
        # new_x = torch.linspace(400.1568, 3999.641, 3734) # 400-4000
        # new_x = torch.linspace(400, 4000, 3600)
        # new_x = torch.linspace(600.7174, 3500.168, 3008)  # 600-3500
        # new_x = torch.linspace(400.1568, 2500.258, 2179)     # 400——2500
        # new_x = torch.linspace(2500.258, 3999.641, 1556)     # 2500-4000
        # new_x = torch.linspace(600.7174, 3999.641, 3526)  # 600-4000
        new_x = torch.linspace(799.3496, 3999.641, 3320)  # 800-4000  对应实验数据，去除最后一个为0的点，共3329个点
        lx = fs[:, None] - new_x[None, :]
        print("lx.shape；\t", lx.shape)
        ly = sigma / (lx ** 2 + sigma ** 2)
        print("ly.shape:\t", ly.shape)
        maxm = torch.max(ly)
        print("torch.max(ly)；\t", maxm)
        new_y = torch.sum(ins[:, None] * ly / maxm, dim=0)
        new_y = new_y.tolist()
        new_x = new_x.tolist()
        return new_x, new_y


if __name__ == "__main__":
    ESI = ExtractSpectraInfo("./gaussian_output", "./gaussian_data/dft_info.tsv")
    ESI.SaveLogInfoProcess()
    print('process complete')







