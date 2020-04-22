import numpy as np
from pathlib import Path
import sqlite3
import mne

# from src.dataset import load_raw_data_file

ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'


#  Sampling Rate
SR = 250


class Preprocess:

    def __init__(self):
        self.data = None
        self.db_header = None
        self.chan_list = None
        self.connection = sqlite3.connect(DB)

    def get_chan_list(self):
        cursor = self.connection.execute('select * from data_table')
        self.db_header = list(map(lambda x: x[0], cursor.description))
        self.chan_list = self.db_header[1:32]

    def get_raw_data(self):
        cursor = self.connection.execute('select * from data_table where pp_list=?', ('04',))
        print('getting rows')
        rows = cursor.fetchall()
        data_array = np.array(rows)
        data = data_array.transpose()
        data = data[1:32]
        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        raw = mne.io.RawArray(data, info=info)  # create the Raw object
        # raw.plot()


def main():
    data = Preprocess()
    data.get_chan_list()
    data.get_raw_data()


main()
