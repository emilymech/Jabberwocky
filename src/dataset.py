import os
import re
from pathlib import Path
import pandas as pd


class Participant:

    def __init__(self):
        self.pp = None
        self.data = None
        self.filepath = None
        self.sample_list = []
        self.ch0_HE = []
        self.ch1_lhe = []
        self.ch2_rhe = []
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
        self.chan_list = [self.ch0_HE, self.ch1_lhe, self.ch2_rhe, self.ch3_LE, self.ch4_A2, self.ch5_MiPf,
                          self.ch6_LLPf, self.ch7_RLPf, self.ch8_LMPf, self.ch9_RMPf, self.ch10_LDFr, self.ch11_RDFr,
                          self.ch12_LMFr, self.ch13_RMFr, self.ch14_LLFr, self.ch15_RLFr, self.ch16_LMCe,
                          self.ch17_RMCe, self.ch18_LDCe, self.ch19_RDCe, self.ch20_MiCe, self.ch21_MiPa,
                          self.ch22_LLTe, self.ch23_RLTe, self.ch24_LDPa, self.ch25_RDPa, self.ch26_LLOc,
                          self.ch27_RLOc, self.ch28_LMOc, self.ch29_RMOc, self.ch30_MiOc, self.event_list]

    def load_raw_data_file(self, participant):
        pp_num = re.search(r'\d+', participant)
        if pp_num:
            self.pp = pp_num.group()
        print("Loading raw data file for Pp {}...".format(self.pp))
        if participant[0] != '_':
            self.filepath = participant
            with open(self.filepath, 'r') as f:
                for line in f.readlines():
                    self.sample_list.append(line.split())

    def preprocess_raw_data(self):
        print('Reformatting data for Pp {}...'.format(self.pp))
        k = 0
        for i in range(len(self.sample_list)):
            for j in range(len(self.sample_list[i])):
                uv = float(self.sample_list[i][j])
                rounded_uv = "{:.6f}".format(uv)
                self.chan_list[k].append(rounded_uv)
                k += 1
                if k == 32:
                    k = 0

    def save_preprocessed_data(self):
        print("Saving preprocessed data for Pp {}...".format(self.pp))

        self.data = pd.DataFrame({'ch0_HE': self.ch0_HE, 'ch1_lhe': self.ch1_lhe, 'ch2_rhe': self.ch2_rhe,
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
    print("Getting data list...")
    file_list = []
    absolute_path = Path(__file__).parent.absolute()
    tmp_path = str(absolute_path).strip('src')
    path = '{}{}'.format(tmp_path, 'data/raw')
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        if file[0] != '_':
            file_list.append(filepath)
    return file_list

