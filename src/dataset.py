import os
from pathlib import Path
import pandas as pd

#  TODO - need to make sure that the Pp class is only loading in the data for one participant at a time
#  TODO - need to work on class dataset


class Dataset:

    def __init__(self):
        pass

    def get_subject_list(self):
        pass

    def load_subject_data(self):
        pass

    def preprocess_subject_data(self):
        pass

    def save_preprocessed_data(self):
        pass


class Pp:

    def __init__(self):
        self.pp = None
        self.data = None
        self.filepath = None
        self.ch0_HE = []
        self.ch1_lhe = []
        self.ch2_rh3 = []
        self.ch3_LE = []
        self.ch4_A2 = []
        self.ch5_MiPf = []
        self.ch6_LLPf = []
        self.ch7_RLPf = []
        self.ch8_LMPf = []
        self.ch9_RMPf = []
        self.ch10_LDFr = []
        self.ch11_RDFr = []
        self.ch12_LMFr = []
        self.ch13_RMFr = []
        self.ch14_LLFr = []
        self.ch15_RLFr = []
        self.ch16_LMCe = []
        self.ch17_RMCe = []
        self.ch18_LDCe = []
        self.ch19_RDCe = []
        self.ch20_MiCe = []
        self.ch21_MiPa = []
        self.ch22_LLTe = []
        self.ch23_RLTe = []
        self.ch24_LDPa = []
        self.ch25_RDPa = []
        self.ch26_LLOc = []
        self.ch27_RLOc = []
        self.ch28_LMOc = []
        self.ch29_RMOc = []
        self.ch30_MiOc = []
        self.event_list = []
        self.chan_list = [self.ch0_HE, self.ch1_lhe, self.ch2_rh3, self.ch3_LE, self.ch4_A2, self.ch5_MiPf,
                          self.ch6_LLPf, self.ch7_RLPf, self.ch8_LMPf, self.ch9_RMPf, self.ch10_LDFr, self.ch11_RDFr,
                          self.ch12_LMFr, self.ch13_RMFr, self.ch14_LLFr, self.ch15_RLFr, self.ch16_LMCe,
                          self.ch17_RMCe, self.ch18_LDCe, self.ch19_RDCe, self.ch20_MiCe, self.ch21_MiPa,
                          self.ch22_LLTe, self.ch23_RLTe, self.ch24_LDPa, self.ch25_RDPa, self.ch26_LLOc,
                          self.ch27_RLOc, self.ch28_LMOc, self.ch29_RMOc, self.ch30_MiOc, self.event_list]

    def load_raw_data_file(self):
        pass
        # absolute_path = Path(__file__).parent.absolute()
        # tmp_path = absolute_path.strip('src')
        # path = '{}{}'.format(tmp_path, 'data/raw')
        # files = os.listdir(path)
        # for file in files:
        #     self.filepath = os.path.join(path, file)
        #     self.pp = file.strip('ec_normalized.txt')
        #     if file[0] != '_':
        #         self.file_list.append(self.filepath)

    def preprocess_raw_data(self):
        print('Reformatting data for Pp {}'.format(self.pp))
        with open(self.filepath, 'r') as f:
            for line in f.readlines():
                sample_list = line.split()

                for i in range(len(self.chan_list)):
                    uv = float(sample_list[i])
                    rounded_uv = "{:.6f}".format(uv)
                    self.chan_list[i].append(rounded_uv)

    def save_preprocessed_data(self):
        self.data = pd.DataFrame({'ch0_HE': self.ch0_HE, 'ch1_lhe': self.ch1_lhe, 'ch2_rh3': self.ch2_rh3,
                                  'ch3_LE': self.ch3_LE, 'ch4_A2': self.ch4_A2, 'ch5_MiPf': self.ch5_MiPf,
                                  'ch6_LLPf': self.ch6_LLPf, 'ch7_RLPf': self.ch7_RLPf,
                                  'ch8_LMPf': self.ch8_LMPf, 'ch9_RMPf': self.ch9_RMPf, 'ch10_LDF': self.ch10_LDFr,
                                  'ch11_RDFr': self.ch11_RDFr, 'ch12_LMFr': self.ch12_LMFr,
                                  'ch13_RMFr': self.ch13_RMFr, 'ch14_LLFr': self.ch14_LLFr,
                                  'ch15_RLFr': self.ch15_RLFr, 'ch16_LMCe': self.ch16_LMCe,
                                  'ch17_RMCe': self.ch17_RMCe, 'ch18_LDCe': self.ch18_LDCe,
                                  'ch19_RDCe': self.ch19_RDCe, 'ch20_MiCe': self.ch20_MiCe,
                                  'ch21_MiPa': self.ch21_MiPa, 'ch22_LLTe': self.ch22_LLTe,
                                  'ch23_RLTe': self.ch23_RLTe, 'ch24_LDPa': self.ch24_LDPa,
                                  'ch25_RDPa': self.ch25_RDPa, 'ch26_LLOc': self.ch26_LLOc,
                                  'ch27_RLOc': self.ch27_RLOc, 'ch28_LMOc': self.ch28_LMOc,
                                  'ch29_RMOc': self.ch29_RMOc, 'ch30_MiOc': self.ch30_MiOc,
                                  'event_list': self.event_list})

        print("Finished reformatting for {}".format(self.pp))


def get_data_list():
    file_list = []
    absolute_path = Path(__file__).parent.absolute()
    tmp_path = absolute_path.strip('src')
    path = '{}{}'.format(tmp_path, 'data/raw')
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        if file[0] != '_':
            file_list.append(filepath)


def main():
    data = Pp()
    data.load_raw_data_file()
    data.preprocess_raw_data()
    data.save_preprocessed_data()


main()

