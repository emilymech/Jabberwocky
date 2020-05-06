import numpy as np
from pathlib import Path
import sqlite3
import mne


# TODO - Need to figure out a way to allow multiple triggers to be epoched in epoch_data
# TODO - Check on filter type
# TODO - need to add grand average functionality (save Pp to processed data)
# TODO - figure out what's up with the scale on the plots

ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'
single_participant_demo = False


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
        self.ecg_inds = None
        self.scores = None
        self.N400_electrodes = None
        self.noun_condition = None
        self.verb_condition = None
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

        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        self.raw = mne.io.RawArray(data, info=info)  # create the Raw object
        self.raw.pick_types(meg=False, eeg=True, eog=True)
        self.raw.set_channel_types(mapping={'ch0_HE': 'eog', 'ch1_lhe': 'eog', 'ch2_rhe': 'eog', 'ch3_LE': 'eog'})

    def plot_raw(self):
        self.raw.plot(block=True, scalings='auto', n_channels=31, title="Raw Data")  # lowpass=30
        # loc_file = f'{ABS_PATH}/{"data/info_files"}{"oldsystem_locs.loc"}'
        # montage = mne.channels.read_montage(loc_file, ch_names=self.chan_list, path=None, unit='m', transform=False)
        # print(montage)

    def re_reference(self):
        # tutorial: https://github.com/mne-tools/mne-python/blob/master/tutorials/preprocessing/
        #           plot_55_setting_eeg_reference.py

        # get rid of the default average reference
        raw_no_ref, _ = mne.set_eeg_reference(self.raw, [])

        # add new reference channel (all zero) (because A1 is not saved in the recording - this will be flat)
        raw_new_ref = mne.add_reference_channels(self.raw, ref_channels=['ch99_A1'])
        # raw_new_ref.plot(block=True, scalings='auto', n_channels=31)

        # set reference to average of A1 and A2
        raw_new_ref.set_eeg_reference(ref_channels=['ch4_A2'])
        # raw_new_ref.plot(block=True, scalings='auto', n_channels=31)

    def filter(self):
        self.raw.filter(1, 30., fir_design='firwin')
        self.raw.plot(block=True, scalings='auto', n_channels=31, title="Filtered Data")

    def artifact_reject(self):
        eog_events = mne.preprocessing.find_eog_events(self.raw, reject_by_annotation=True)
        onsets = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ['bad blink'] * len(eog_events)
        blink_annot = mne.Annotations(onsets, durations, descriptions,
                                      orig_time=self.raw.info['meas_date'])
        self.raw.set_annotations(blink_annot)

        eeg_picks = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=True, exclude='bads')

        fig = self.raw.plot(block=True, scalings='auto', n_channels=31, events=eog_events,
                            order=eeg_picks, title="Auto-Rejected Data")
        fig.canvas.key_press_event('a')

        interactive_annot = self.raw.annotations

        self.raw.set_annotations(blink_annot + interactive_annot)

        average_eog = mne.preprocessing.create_eog_epochs(self.raw).average()
        print('We found %i EOG events' % average_eog.nave)
        average_eog.plot()

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
                                 flat=self.flat_criteria, reject_by_annotation=True,)
        self.epochs.plot_drop_log()
        self.epochs.drop_bad()
        self.epochs.apply_baseline((-.2, 0))  # baseline correct

    def average(self):
        self.N400_electrodes = ['ch12_LMFr', 'ch13_RMFr', 'ch16_LMCe', 'ch17_RMCe',
                                'ch20_MiCe', 'ch21_MiPa', 'ch24_LDPa', 'ch25_RDPa']
        self.noun_condition = ['Congruent Unambiguous Nouns', 'Congruent Ambiguous Nouns',
                               'Jabberwocky Unambiguous Nouns', 'Jabberwocky Ambiguous Nouns',
                               'Random Unambiguous Nouns', 'Random Ambiguous Nouns']
        self.verb_condition = ['Congruent Unambiguous Verbs', 'Congruent Ambiguous Verbs',
                               'Jabberwocky Unambiguous Verbs', 'Jabberwocky Ambiguous Verbs',
                               'Random Unambiguous Verbs', 'Random Ambiguous Verbs']

        all_evokeds = dict((cond, self.epochs[cond].average()) for cond in self.event_dict)

        # noun_epochs = self.epochs[self.noun_condition].average(picks=self.N400_electrodes)
        # verb_epochs = self.epochs[self.verb_condition].average(picks=self.N400_electrodes)
        # noun_epochs.plot_image(group_by=self.N400_electrodes)
        # verb_epochs.plot_image(group_by=self.N400_electrodes)
        mne.viz.plot_compare_evokeds(all_evokeds, colors=['lightcoral', 'indianred', 'maroon',
                                                          'honeydew', 'palegreen', 'darkseagreen',
                                                          'lightcyan', 'paleturquoise', 'darkslategray',
                                                          'lavenderblush', 'deeppink', 'mediumvioletred'],
                                     split_legend=True, picks=self.N400_electrodes)


def main():
    if single_participant_demo:
        data = PpPreprocess()
        data.get_chan_pp_lists()
        data.get_raw_data('11')
        data.plot_raw()
        data.re_reference()
        data.filter()
        data.artifact_reject()
        data.get_events()
        data.epoch_data()
        data.average()
    else:
        connection = sqlite3.connect(DB)  # connect to your DB
        cursor = connection.cursor()  # get a cursor
        participant_list = [participant[0] for participant in cursor.execute("SELECT pp_list FROM data_table")]
        participant_set = set(participant_list)
        print(participant_set)

        for participant in participant_set:
            data = PpPreprocess()
            data.get_chan_pp_lists()
            data.get_raw_data(participant)
            data.plot_raw()
            data.re_reference()
            data.filter()
            data.artifact_reject()
            data.get_events()
            data.epoch_data()
            data.average()


main()
