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

    def get_raw(self):
        for Pp, data in self.data.items():
            col_names = data.index[0]
            no_header = data.iloc[:, -0]
            print(col_names)
            ch_names = [col_names]
            info = mne.create_info(ch_names, SR)  # Create the info structure needed by MNE
            raw = mne.io.RawArray(no_header, info)  # create the Raw object
            raw.plot()


def main():
    data = Preprocess()
    data.get_raw()


main()
