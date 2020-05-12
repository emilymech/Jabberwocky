import os
from pathlib import Path
import numpy as np
import sqlite3
import mne

#  TODO - Process all participants by hand
#  TODO - Save all logs for each participant
#  TODO - save participants with too many artifacts to "bad_participants"


ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'
SAVE_PATH = f'{ABS_PATH}/data/processed'

# The total set of participants in the dataset: {4, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 20, 21, 22, 23,
#                                               24, 25, 26, 28, 30, 32, 34, 35, 37}

SAVE = False
Pp = '7'
OVERWRITE = False


SR = 250  # Sampling Rate in Hz


class PpPreprocess:

    def __init__(self):
        self.data = None
        self.db_header = None
        self.chan_list = None
        self.participant_list = None
        self.events = None
        self.event_list = None
        self.all_events_dict = None
        self.event_dict = {}
        self.raw = None
        self.reject_criteria = None
        self.stronger_reject_criteria = None
        self.flat_criteria = None
        self.epochs = None
        self.connection = sqlite3.connect(DB)

    def get_chan_pp_lists(self):
        cursor = self.connection.execute('select * from data_table')
        self.db_header = list(map(lambda x: x[0], cursor.description))
        self.chan_list = self.db_header[1:33]

    def get_raw_data(self, participant):
        cursor = self.connection.execute('select * from data_table where pp_list=?', (participant,))
        print(f'Getting Pp {participant} from database...')
        rows = cursor.fetchall()
        print(f'Finished getting Pp {participant} from database')

        data_array = np.array(rows)
        data = data_array.transpose()
        data = data[1:33, :]
        self.event_list = data[31, :]
        data[31, :] = abs(data[31, :])

        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        self.raw = mne.io.RawArray(data, info=info)  # create the Raw object
        self.raw.pick_types(meg=False, eeg=True, eog=True)
        self.raw.set_channel_types(mapping={'ch0_HE': 'eog', 'ch1_lhe': 'eog', 'ch2_rhe': 'eog', 'ch3_LE': 'eog',
                                            'event_list': 'stim'})

    def plot_raw(self):
        self.raw.plot(block=True, scalings=dict(eeg=50, eog=50),
                      n_channels=31, title="Raw Data")

    def re_reference(self):
        # tutorial: https://github.com/mne-tools/mne-python/blob/master/tutorials/preprocessing/
        #           plot_55_setting_eeg_reference.py

        # get rid of the default average reference
        raw_no_ref, _ = mne.set_eeg_reference(self.raw, [])

        # add new reference channel (all zero) (because A1 is not saved in the recording - this will be flat)
        raw_new_ref = mne.add_reference_channels(self.raw, ref_channels=['ch99_A1'])
        # raw_new_ref.plot(block=True, scalings=dict(eeg=50, grad=1e13, mag=1e15, eog=50), n_channels=31)

        # set reference to average of A1 and A2
        raw_new_ref.set_eeg_reference(ref_channels=['ch4_A2'])
        # raw_new_ref.plot(block=True, scalings=dict(eeg=50e-6, grad=1e13, mag=1e15, eog=50e-6), n_channels=31)

    def filter(self):
        self.raw.filter(1, 30., fir_design='firwin')
        self.raw.plot(block=True, scalings=dict(eeg=50, eog=50), n_channels=31,
                      title="Filtered Data")

    def artifact_reject(self):
        eog_events = mne.preprocessing.find_eog_events(self.raw, reject_by_annotation=True)
        onsets = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ['bad blink'] * len(eog_events)
        blink_annot = mne.Annotations(onsets, durations, descriptions,
                                      orig_time=self.raw.info['meas_date'])
        self.raw.set_annotations(blink_annot)
        print("Auto annotations:", self.raw.annotations)
        print(len(self.raw.annotations))

        eeg_picks = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=True, exclude='bads')

        fig = self.raw.plot(block=True, scalings=dict(eeg=50, eog=50), n_channels=31,
                            events=eog_events, order=eeg_picks, title="Auto-Rejected Data")
        fig.canvas.key_press_event('a')

        interactive_annot = self.raw.annotations
        self.raw.set_annotations(interactive_annot)
        print("Auto and hand annotations:", self.raw.annotations)
        print(len(self.raw.annotations))

        # possible arguments for Epochs if want to do additional quick and dirty reject in get events
        self.reject_criteria = dict(eeg=100e-6,  # 100 µV
                                    eog=200e-6)  # 200 µV

        self.stronger_reject_criteria = dict(eeg=100e-6,  # 100 µV
                                             eog=100e-6)  # 100 µV

        self.flat_criteria = dict(eeg=1e-6)  # 1 µV

    def get_events(self):
        print('Getting events...')
        self.events = mne.find_events(self.raw, stim_channel='event_list', shortest_event=1, verbose=False)
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
                                'all ambiguous words': [73, 74, 83, 84, 93, 94],
                                'Congruent Words': [71, 72, 73, 74],
                                'Jabberwocky Words': [81, 82, 83, 84], 'Random Words': [91, 92, 93, 94]}

        self.event_dict = {'Congruent Unambiguous Nouns': 71, 'Congruent Unambiguous Verbs': 72,
                           'Congruent Ambiguous Nouns': 73, 'Congruent Ambiguous Verbs': 74,
                           'Jabberwocky Unambiguous Nouns': 81, 'Jabberwocky Unambiguous Verbs': 82,
                           'Jabberwocky Ambiguous Nouns': 83, 'Jabberwocky Ambiguous Verbs': 84,
                           'Random Unambiguous Nouns': 91, 'Random Unambiguous Verbs': 92,
                           'Random Ambiguous Nouns': 93, 'Random Ambiguous Verbs': 94}

    def epoch_data(self):
        print('Getting epochs...')
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_dict, preload=True, tmin=-0.2, tmax=1.0,
                                 reject_by_annotation=True,)
        self.epochs.drop_bad()  # if needed can add second pass stronger reject

        self.epochs.plot_drop_log()
        self.epochs.plot(block=True, scalings=dict(eeg=50, eog=50), n_channels=31, title="Epochs")

    def save_data(self, save, participant, save_path):
        if save:
            self.epochs.save(f'{save_path}/{participant}_preprocessed-epo.fif', overwrite=OVERWRITE)


def main():
    data = PpPreprocess()
    data.get_chan_pp_lists()
    data.get_raw_data(Pp)
    data.plot_raw()
    data.re_reference()
    data.filter()
    data.artifact_reject()
    data.get_events()
    data.epoch_data()
    data.save_data(SAVE, Pp, SAVE_PATH)


main()
