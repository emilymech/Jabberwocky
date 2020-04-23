import numpy as np
from pathlib import Path
import sqlite3
import mne
# import os
# from datetime import timedelta


# TODO - Need to figure out a way to allow multiple triggers to be epoched in epoch_data
# TODO - Figure out what is wrong with the loaded raw data
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
        self.stronger_reject_criteria = None
        self.flat_criteria = None
        self.epochs = None
        self.ecg_inds = None
        self.scores = None
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
        data = data[1:33, :]

        info = mne.create_info(self.chan_list, SR, ch_types='eeg')  # Create the info structure needed by MNE
        self.raw = mne.io.RawArray(data, info=info)  # create the Raw object
        self.raw.pick_types(meg=False, eeg=True, eog=True)
        self.raw.set_channel_types(mapping={'ch0_HE': 'eog', 'ch1_lhe': 'eog', 'ch2_rhe': 'eog', 'ch3_LE': 'eog'})

    def plot_raw(self):
        self.raw.plot(block=True, scalings='auto', n_channels=31)  # lowpass=30
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
        self.epochs = mne.Epochs(self.raw, self.events, preload=True)
        self.epochs.plot_drop_log()
        self.epochs.drop_bad()
        print(self.epochs.drop_log)

    def eog(self):
        fig = self.raw.plot(block=True, scalings='auto', n_channels=31)
        fig.canvas.key_press_event('a')

        eog_events = mne.preprocessing.find_eog_events(self.raw, ch_name='ch1_lhe', reject_by_annotation=False)
        onsets = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ['bad blink'] * len(eog_events)
        blink_annot = mne.Annotations(onsets, durations, descriptions,
                                      orig_time=self.raw.info['meas_date'])
        self.raw.set_annotations(blink_annot)

        eeg_picks = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=True)

        self.raw.plot(block=True, scalings='auto', n_channels=31, events=eog_events, order=eeg_picks)

        self.reject_criteria = dict(eeg=100e-6,  # 100 µV
                                    eog=200e-6)  # 200 µV

        self.stronger_reject_criteria = dict(eeg=100e-6,  # 100 µV
                                             eog=100e-6)  # 100 µV

        self.flat_criteria = dict(eeg=1e-6)  # 1 µV

    def sixty_hertz(self):
        fig = self.raw.plot_psd(tmax=np.inf, fmax=250, average=True)
        # add some arrows at 60 Hz and its harmonics:
        for ax in fig.axes[:2]:
            freqs = ax.lines[-1].get_xdata()
            psds = ax.lines[-1].get_ydata()
            for freq in (60, 120, 180, 240):
                idx = np.searchsorted(freqs, freq)
                ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                         width=0.1, head_width=3, length_includes_head=True)

    def heartbeat(self):
        ecg_epochs = mne.preprocessing.create_ecg_epochs(self.raw)
        ecg_epochs.plot_image(combine='mean')
        avg_ecg_epochs = ecg_epochs.average()
        avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
        avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])

    def interpolate(self):
        evoked = mne.read_evokeds(self.raw, condition='Congruent Unambiguous Nouns',
                                  baseline=(None, 0))

        # plot with bads
        evoked.plot(exclude=[], time_unit='s')

        # compute interpolation (also works with Raw and Epochs objects)
        evoked.interpolate_bads(reset_bads=False, verbose=False)

        # plot interpolated (previous bads)
        evoked.plot(exclude=[], time_unit='s')

    def ica(self):
        ica = ICA(n_components=0.95, method='fastica').fit(self.epochs)

        ecg_epochs = create_ecg_epochs(self.raw, tmin=-.5, tmax=.5)
        self.ecg_inds, self.scores = ica.find_bads_ecg(ecg_epochs)

        ica.plot_components(self.ecg_inds)
        ica.plot_properties(self.epochs, picks=self.ecg_inds)

    def filter(self):
        self.raw.filter_data(self.raw, sfreq=SR, l_freq =1, h_freq=30, fir_design='firwin')
        self.raw.plot()


def main():
    data = PpPreprocess()
    data.get_chan_pp_lists()
    data.get_raw_data('04')
    data.plot_raw()
    data.re_reference()
    # data.interpolate()
    data.eog()
    # data.ica()
    # data.heartbeat()
    data.filter()
    # data.sixty_hertz()
    data.get_events()
    data.epoch_data()


main()
