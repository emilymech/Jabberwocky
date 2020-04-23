import numpy as np
from pathlib import Path
import sqlite3
import mne

# TODO - Need to figure out a way to allow multiple triggers to be epoched in epoch_data

ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'


SR = 250  # Sampling Rate in Hz


class PpPreprocess:

    def __init__(self):
        self.data = None
        self.db_header = None
        self.chan_list = None
        self.participant_list = None
        self.events = None
        self.event_dict = None
        self.raw = None
        self.reject_criteria = None
        self.epochs = None
        self.connection = sqlite3.connect(DB)

    def get_chan_pp_lists(self):
        cursor = self.connection.execute('select * from data_table')
        self.db_header = list(map(lambda x: x[0], cursor.description))
        self.chan_list = self.db_header[1:33]
        self.participant_list = self.db_header[0]

    def get_raw_data(self, participant):
        cursor = self.connection.execute('select * from data_table where pp_list=?', (participant,))
        print(f'Getting Pp {participant} from database...')
        rows = cursor.fetchall()
        print(f'Finished getting Pp {participant} from database')
        data_array = np.array(rows)
        data = data_array.transpose()
        data = data[1:33]
        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        self.raw = mne.io.RawArray(data, info=info)  # create the Raw object

    def get_events(self):
        print('Getting events...')
        self.events = mne.find_events(self.raw, stim_channel='event_list')
        self.event_dict = {'Congruent Unambiguous Words': [71, 72], 'Congruent Ambiguous Words': [73, 74],
                           'Jabberwocky Unambiguous Words': [81, 82], 'Jabberwocky Ambiguous Words': [83, 84],
                           'Random Unambiguous Words': [91, 92], 'Random Ambiguous Words': [93, 94],
                           'Congruent Unambiguous Nouns': 71, 'Congruent Unambiguous Verbs': 72,
                           'Congruent Ambiguous Nouns': 73, 'Congruent Ambiguous Verbs': 74,
                           'Jabberwocky Unambiguous Nouns': 81, 'Jabberwocky Unambiguous Verbs': 82,
                           'Jabberwocky Ambiguous Nouns': 83, 'Jabberwocky Ambiguous Verbs': 84,
                           'Random Unambiguous Nouns': 91, 'Random Unambiguous Verbs': 92,
                           'Random Ambiguous Nouns': 93, 'Random Ambiguous Verbs': 94,
                           'all unambiguous words': [71, 72, 81, 82, 91, 92],
                           'all ambiguous words': [73, 74, 83, 84, 93, 94], 'Congruent Words': [71, 72, 73, 74],
                           'Jabberwocky Words': [81, 82, 83, 84], 'Random Words': [91, 92, 93, 94]}

    def epoch_data(self):
        print('Getting epochs...')
        self.reject_criteria = dict(eeg=150e-6)
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_dict, tmin=-0.2, tmax=1.0,
                                 reject=self.reject_criteria, preload=True)
        print(self.epochs)


def main():
    data = PpPreprocess()
    data.get_chan_pp_lists()
    data.get_raw_data('04')
    data.get_events()
    data.epoch_data()


main()
