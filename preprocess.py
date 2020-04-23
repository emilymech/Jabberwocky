import numpy as np
from pathlib import Path
import sqlite3
import mne


ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'


SR = 250  # Sampling Rate in Hz


class Preprocess:

    def __init__(self):
        self.data = None
        self.db_header = None
        self.chan_list = None
        self.participant_list = None
        self.event_list = []
        self.raw = None
        self.connection = sqlite3.connect(DB)

    def get_chan_pp_lists(self):
        cursor = self.connection.execute('select * from data_table')
        self.db_header = list(map(lambda x: x[0], cursor.description))
        self.chan_list = self.db_header[1:32]
        self.participant_list = self.db_header[0]

    def get_raw_data(self, participant):
        cursor = self.connection.execute('select * from data_table where pp_list=?', (participant,))
        print(f'Getting Pp {participant} from database...')
        rows = cursor.fetchall()
        print(f'Finished getting Pp {participant} from database')
        data_array = np.array(rows)
        data = data_array.transpose()
        data = data[1:32]
        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        self.raw = mne.io.RawArray(data, info=info)  # create the Raw object
        cur = self.connection.execute('select event_list from data_table')
        for event in cur.fetchall():
            self.event_list.append(str(event[0]))
        print(self.event_list)

    def epoch_data(self):
        events = mne.find_events(self.raw, stim_channel=self.event_list)
        print(events[:5])


def main():
    data = Preprocess()
    data.get_chan_pp_lists()
    data.get_raw_data('04')
    data.epoch_data()


main()
