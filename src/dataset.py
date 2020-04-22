import re


class Participant:

    def __init__(self):
        self.pp = None
        self.data = None
        self.filepath = None
        self.pp_list = []
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
        for i in range(len(self.sample_list)):
            for j in range(len(self.sample_list[i])):
                uv = float(self.sample_list[i][j])
                rounded_uv = "{:.6f}".format(uv)
                float_uv = float(rounded_uv)
                self.chan_list[j].append(float_uv)

