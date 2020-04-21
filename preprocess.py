import os
import numpy as np
import mne
import pickle as pkl
from pathlib import Path


#  Sampling Rate
SR = 250


class Preprocess:
    def __init__(self):
        self.data = None

    def load_pickle(self):
        absolute_path = Path(__file__).parent.absolute()
        filename = '{}{}{}'.format(absolute_path, 'data/reformatted/', 'reformatted_dataset.pkl')
        with open(filename, 'rb') as pickle_file:
            self.data = pkl.load(pickle_file)

    def get_raw(self):
        for Pp, data in self.data.items():
            col_names = data.index[0]
            no_header = data.iloc[:, -0]
            print(col_names)
            #  Get a list of Channels
            ch_names = [col_names]
            #  Create the info structure needed by MNE
            info = mne.create_info(ch_names, SR)
            #  Finally, create the Raw object
            raw = mne.io.RawArray(no_header, info)

            #  Plot it!
            raw.plot()


def main():
    process = Preprocess()
    process.load_pickle()
    process.get_raw()


main()