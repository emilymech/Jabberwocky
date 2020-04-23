import numpy as np
from pathlib import Path
import sqlite3
import mne

# TODO - Need to figure out a way to allow multiple triggers to be epoched in epoch_data
# TODO - Figure out what is wrong with the loaded raw data
# TODO - Add reject artifacts options: ICA, annotation of raw data
# TODO - Add plotting capabilities for each processing step

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
        self.all_events_dict = None
        self.event_dict = {}
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
        self.all_events_dict = {'Congruent Unambiguous Words': [71, 72], 'Congruent Ambiguous Words': [73, 74],
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

        self.event_dict = {'Congruent Unambiguous Nouns': 71, 'Congruent Unambiguous Verbs': 72,
                           'Congruent Ambiguous Nouns': 73, 'Congruent Ambiguous Verbs': 74,
                           'Jabberwocky Unambiguous Nouns': 81, 'Jabberwocky Unambiguous Verbs': 82,
                           'Jabberwocky Ambiguous Nouns': 83, 'Jabberwocky Ambiguous Verbs': 84,
                           'Random Unambiguous Nouns': 91, 'Random Unambiguous Verbs': 92,
                           'Random Ambiguous Nouns': 93, 'Random Ambiguous Verbs': 94}

    def epoch_data(self):
        print('Getting epochs...')
        self.reject_criteria = dict(eeg=150e-6)
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_dict, tmin=-0.2, tmax=1.0,
                                 reject=self.reject_criteria, preload=True)

    def annotate_data(self):
        pass


def main():
    data = PpPreprocess()
    data.get_chan_pp_lists()
    data.get_raw_data('04')
    data.get_events()
    data.epoch_data()


main()
