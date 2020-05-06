import numpy as np
from pathlib import Path
import sqlite3
import mne


# TODO - Need to figure out a way to allow multiple triggers to be epoched in epoch_data
# TODO - Check on filter type
# TODO - somewhere there is an abs value being applied - find it and turn it off!
# TODO - figure out what's up with the scale on the plots

ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'
single_participant_mode = False


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
        self.scores = None
        self.nouns = {}
        self.verbs = {}
        self.noun_epochs = None
        self.verb_epochs = None
        self.N400_electrodes = None
        self.noun_condition = None
        self.verb_condition = None
        self.all_evoked = None
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
        if single_participant_mode:
            self.raw.plot(block=True, scalings=dict(eeg=50, eog=50),
                          n_channels=31, title="Raw Data")
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
        # raw_new_ref.plot(block=True, scalings=dict(eeg=50, grad=1e13, mag=1e15, eog=50), n_channels=31)

        # set reference to average of A1 and A2
        raw_new_ref.set_eeg_reference(ref_channels=['ch4_A2'])
        # raw_new_ref.plot(block=True, scalings=dict(eeg=50e-6, grad=1e13, mag=1e15, eog=50e-6), n_channels=31)

    def filter(self):
        self.raw.filter(1, 30., fir_design='firwin')
        if single_participant_mode:
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
        if single_participant_mode:
            fig = self.raw.plot(block=True, scalings=dict(eeg=50,eog=50), n_channels=31,
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

        if single_participant_mode:
            self.epochs.plot_drop_log()
            self.epochs.plot(block=True, scalings=dict(eeg=50, eog=50), n_channels=31, title="Epochs")

    def average(self):
        self.epochs.apply_baseline((-.2, 0))  # baseline correct
        self.N400_electrodes = ['ch12_LMFr', 'ch13_RMFr', 'ch16_LMCe', 'ch17_RMCe',
                                'ch20_MiCe', 'ch21_MiPa', 'ch24_LDPa', 'ch25_RDPa']
        self.noun_condition = ['Congruent Unambiguous Nouns', 'Congruent Ambiguous Nouns',
                               'Jabberwocky Unambiguous Nouns', 'Jabberwocky Ambiguous Nouns',
                               'Random Unambiguous Nouns', 'Random Ambiguous Nouns']
        self.verb_condition = ['Congruent Unambiguous Verbs', 'Congruent Ambiguous Verbs',
                               'Jabberwocky Unambiguous Verbs', 'Jabberwocky Ambiguous Verbs',
                               'Random Unambiguous Verbs', 'Random Ambiguous Verbs']

        self.all_evoked = dict((cond, self.epochs[cond].average()) for cond in self.event_dict)

        for evoked in self.all_evoked:
            if evoked in self.noun_condition:
                self.nouns[evoked] = self.all_evoked[evoked]
            else:
                self.verbs[evoked] = self.all_evoked[evoked]

        self.noun_epochs = self.epochs[self.noun_condition].average(picks=self.N400_electrodes)
        self.verb_epochs = self.epochs[self.verb_condition].average(picks=self.N400_electrodes)

    def plot_erps(self):
        self.noun_epochs.plot_image(group_by=self.N400_electrodes)
        self.verb_epochs.plot_image(group_by=self.N400_electrodes)

        mne.viz.plot_compare_evokeds(self.all_evoked, invert_y=True,
                                     colors=['lightcoral', 'indianred', 'maroon',
                                             'honeydew', 'palegreen', 'darkseagreen',
                                             'lightcyan', 'paleturquoise', 'darkslategray',
                                             'lavenderblush', 'deeppink', 'mediumvioletred'],
                                     split_legend=True, picks=self.N400_electrodes, title="All Conditions",
                                     )

        mne.viz.plot_compare_evokeds(self.nouns, invert_y=True,
                                     colors=['indianred', 'maroon', 'palegreen', 'darkseagreen',
                                             'paleturquoise', 'darkslategray',
                                             'deeppink', 'mediumvioletred'],
                                     split_legend=True, picks=self.N400_electrodes, title="Noun Conditions")

        mne.viz.plot_compare_evokeds(self.verbs, invert_y=True,
                                     colors=['indianred', 'maroon', 'palegreen', 'darkseagreen',
                                             'paleturquoise', 'darkslategray',
                                             'deeppink', 'mediumvioletred'],
                                     split_legend=True, picks=self.N400_electrodes, title="Verb Conditions")

    def evoked_to_list(self, congruent_unambiguous_nouns, congruent_ambiguous_nouns, jabberwocky_unambiguous_nouns,
                       jabberwocky_ambiguous_nouns, random_unambiguous_nouns, random_ambiguous_nouns,
                       congruent_unambiguous_verbs, congruent_ambiguous_verbs, jabberwocky_unambiguous_verbs,
                       jabberwocky_ambiguous_verbs, random_unambiguous_verbs, random_ambiguous_verbs):

        for condition in self.all_evoked:
            if condition == "Congruent Unambiguous Nouns":
                congruent_unambiguous_nouns.append(self.all_evoked[condition])
            elif condition == "Congruent Ambiguous Nouns":
                congruent_ambiguous_nouns.append(self.all_evoked[condition])
            elif condition == "Jabberwocky Unambiguous Nouns":
                jabberwocky_unambiguous_nouns.append(self.all_evoked[condition])
            elif condition == "Jabberwocky Ambiguous Nouns":
                jabberwocky_ambiguous_nouns.append(self.all_evoked[condition])
            elif condition == "Random Unambiguous Nouns":
                random_unambiguous_nouns.append(self.all_evoked[condition])
            elif condition == "Random Ambiguous Nouns":
                random_ambiguous_nouns.append(self.all_evoked[condition])

            elif condition == "Congruent Unambiguous Verbs":
                congruent_unambiguous_verbs.append(self.all_evoked[condition])
            elif condition == "Congruent Ambiguous Verbs":
                congruent_ambiguous_verbs.append(self.all_evoked[condition])
            elif condition == "Jabberwocky Unambiguous Verbs":
                jabberwocky_unambiguous_verbs.append(self.all_evoked[condition])
            elif condition == "Jabberwocky Ambiguous Verbs":
                jabberwocky_ambiguous_verbs.append(self.all_evoked[condition])
            elif condition == "Random Unambiguous Verbs":
                random_unambiguous_verbs.append(self.all_evoked[condition])
            elif condition == "Random Ambiguous Verbs":
                random_ambiguous_verbs.append(self.all_evoked[condition])

            else:
                print("This condition is not going in a list for grand averaging!")


def grand_average(congruent_unambiguous_nouns, congruent_ambiguous_nouns, jabberwocky_unambiguous_nouns,
                  jabberwocky_ambiguous_nouns, random_unambiguous_nouns, random_ambiguous_nouns,
                  congruent_unambiguous_verbs, congruent_ambiguous_verbs, jabberwocky_unambiguous_verbs,
                  jabberwocky_ambiguous_verbs, random_unambiguous_verbs, random_ambiguous_verbs):

    congruent_unambiguous_nouns_ga = mne.grand_average(congruent_unambiguous_nouns, interpolate_bads=False,
                                                       drop_bads=True)
    congruent_ambiguous_nouns_ga = mne.grand_average(congruent_ambiguous_nouns, interpolate_bads=False,
                                                     drop_bads=True)
    jabberwocky_unambiguous_nouns_ga = mne.grand_average(jabberwocky_unambiguous_nouns, interpolate_bads=False,
                                                         drop_bads=True)
    jabberwocky_ambiguous_nouns_ga = mne.grand_average(jabberwocky_ambiguous_nouns, interpolate_bads=False,
                                                       drop_bads=True)
    random_unambiguous_nouns_ga = mne.grand_average(random_unambiguous_nouns, interpolate_bads=False,
                                                    drop_bads=True)
    random_ambiguous_nouns_ga = mne.grand_average(random_ambiguous_nouns, interpolate_bads=False,
                                                  drop_bads=True)

    noun_ga_dict = {'Congruent_Unambiguous_Nouns': congruent_unambiguous_nouns_ga,
                    'Congruent_Ambiguous_Nouns': congruent_ambiguous_nouns_ga,
                    'Jabberwocky_Unambiguous_Nouns': jabberwocky_unambiguous_nouns_ga,
                    'Jabberwocky_Ambiguous_Nouns': jabberwocky_ambiguous_nouns_ga,
                    'Random_Unambiguous_Nouns': random_unambiguous_nouns_ga,
                    'Random_Ambiguous_Nouns': random_ambiguous_nouns_ga}

    congruent_unambiguous_verbs_ga = mne.grand_average(congruent_unambiguous_verbs, interpolate_bads=False,
                                                       drop_bads=True)
    congruent_ambiguous_verbs_ga = mne.grand_average(congruent_ambiguous_verbs, interpolate_bads=False,
                                                     drop_bads=True)
    jabberwocky_unambiguous_verbs_ga = mne.grand_average(jabberwocky_unambiguous_verbs, interpolate_bads=False,
                                                         drop_bads=True)
    jabberwocky_ambiguous_verbs_ga = mne.grand_average(jabberwocky_ambiguous_verbs, interpolate_bads=False,
                                                       drop_bads=True)
    random_unambiguous_verbs_ga = mne.grand_average(random_unambiguous_verbs, interpolate_bads=False,
                                                    drop_bads=True)
    random_ambiguous_verbs_ga = mne.grand_average(random_ambiguous_verbs, interpolate_bads=False,
                                                  drop_bads=True)

    verb_ga_dict = {'Congruent_Unambiguous_Verbs': congruent_unambiguous_verbs_ga,
                    'Congruent_Ambiguous_Verbs': congruent_ambiguous_verbs_ga,
                    'Jabberwocky_Unambiguous_Verbs': jabberwocky_unambiguous_verbs_ga,
                    'Jabberwocky_Ambiguous_Verbs': jabberwocky_ambiguous_verbs_ga,
                    'Random_Unambiguous_Verbs': random_unambiguous_verbs_ga,
                    'Random_Ambiguous_Verbs': random_ambiguous_verbs_ga}

    mne.viz.plot_compare_evokeds(noun_ga_dict, picks=['ch12_LMFr', 'ch13_RMFr', 'ch16_LMCe', 'ch17_RMCe',
                                                      'ch20_MiCe', 'ch21_MiPa', 'ch24_LDPa', 'ch25_RDPa'],
                                 colors=['indianred', 'maroon', 'palegreen', 'darkseagreen','paleturquoise',
                                         'darkslategray', 'deeppink', 'mediumvioletred'], split_legend=True,
                                 title="Nouns")

    mne.viz.plot_compare_evokeds(verb_ga_dict, picks=['ch12_LMFr', 'ch13_RMFr', 'ch16_LMCe', 'ch17_RMCe',
                                                      'ch20_MiCe', 'ch21_MiPa', 'ch24_LDPa', 'ch25_RDPa'],
                                 colors=['indianred', 'maroon', 'palegreen', 'darkseagreen', 'paleturquoise',
                                         'darkslategray', 'deeppink', 'mediumvioletred'], split_legend=True,
                                 title="Verbs")


def main():
    if single_participant_mode:
        data = PpPreprocess()
        data.get_chan_pp_lists()
        data.get_raw_data('06')
        data.plot_raw()
        data.re_reference()
        data.filter()
        data.artifact_reject()
        data.get_events()
        data.epoch_data()
        data.average()
        data.plot_erps()

    else:

        connection = sqlite3.connect(DB)  # connect to your DB
        cursor = connection.cursor()  # get a cursor
        participant_list = [participant[0] for participant in cursor.execute("SELECT pp_list FROM data_table")]
        participant_set = set(participant_list)


        congruent_unambiguous_nouns = []
        congruent_ambiguous_nouns = []
        jabberwocky_unambiguous_nouns = []
        jabberwocky_ambiguous_nouns = []
        random_unambiguous_nouns = []
        random_ambiguous_nouns = []

        congruent_unambiguous_verbs = []
        congruent_ambiguous_verbs = []
        jabberwocky_unambiguous_verbs = []
        jabberwocky_ambiguous_verbs = []
        random_unambiguous_verbs = []
        random_ambiguous_verbs = []

        print("The set of participants to preprocess:", participant_set)

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
            data.plot_erps()
            data.evoked_to_list(congruent_unambiguous_nouns, congruent_ambiguous_nouns, jabberwocky_unambiguous_nouns,
                                jabberwocky_ambiguous_nouns, random_unambiguous_nouns, random_ambiguous_nouns,
                                congruent_unambiguous_verbs, congruent_ambiguous_verbs, jabberwocky_unambiguous_verbs,
                                jabberwocky_ambiguous_verbs, random_unambiguous_verbs, random_ambiguous_verbs)

        grand_average(congruent_unambiguous_nouns, congruent_ambiguous_nouns, jabberwocky_unambiguous_nouns,
                      jabberwocky_ambiguous_nouns, random_unambiguous_nouns, random_ambiguous_nouns,
                      congruent_unambiguous_verbs, congruent_ambiguous_verbs, jabberwocky_unambiguous_verbs,
                      jabberwocky_ambiguous_verbs, random_unambiguous_verbs, random_ambiguous_verbs)


main()
