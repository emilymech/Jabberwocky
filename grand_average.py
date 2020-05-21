from pathlib import Path
import sqlite3
import mne
import os
import re

ABS_PATH = Path(__file__).parent.absolute()
DB = f'{ABS_PATH}{"/reformatted_data.sqlite"}'
SAVE_PATH = f'{ABS_PATH}/datasets/output'


class Average:

    def __init__(self):
        self.epoch_list = []
        self.participant_list = []
        self.N400_electrodes = ['ch12_LMFr', 'ch13_RMFr', 'ch16_LMCe', 'ch17_RMCe',
                                'ch20_MiCe', 'ch21_MiPa', 'ch24_LDPa', 'ch25_RDPa']

        self.noun_condition = ['Congruent Unambiguous Nouns', 'Congruent Ambiguous Nouns',
                               'Jabberwocky Unambiguous Nouns', 'Jabberwocky Ambiguous Nouns',
                               'Random Unambiguous Nouns', 'Random Ambiguous Nouns']

        self.verb_condition = ['Congruent Unambiguous Verbs', 'Congruent Ambiguous Verbs',
                               'Jabberwocky Unambiguous Verbs', 'Jabberwocky Ambiguous Verbs',
                               'Random Unambiguous Verbs', 'Random Ambiguous Verbs']

        self.event_dict = {'Congruent Unambiguous Nouns': 71, 'Congruent Unambiguous Verbs': 72,
                           'Congruent Ambiguous Nouns': 73, 'Congruent Ambiguous Verbs': 74,
                           'Jabberwocky Unambiguous Nouns': 81, 'Jabberwocky Unambiguous Verbs': 82,
                           'Jabberwocky Ambiguous Nouns': 83, 'Jabberwocky Ambiguous Verbs': 84,
                           'Random Unambiguous Nouns': 91, 'Random Unambiguous Verbs': 92,
                           'Random Ambiguous Nouns': 93, 'Random Ambiguous Verbs': 94}
        self.all_evoked = None
        self.pp_evoked = None
        self.nouns = {}
        self.verbs = {}
        self.noun_evoked = []
        self.verb_evoked = []
        self.congruent_unambiguous_nouns = None
        self.congruent_ambiguous_nouns = None
        self.jabberwocky_unambiguous_nouns = None
        self.jabberwocky_ambiguous_nouns = None
        self.random_unambiguous_nouns = None
        self.random_ambiguous_nouns = None
        self.congruent_unambiguous_verbs = None
        self.congruent_ambiguous_verbs = None
        self.jabberwocky_unambiguous_verbs = None
        self.jabberwocky_ambiguous_verbs = None
        self.random_unambiguous_verbs = None
        self.random_ambiguous_verbs = None
        self.noun_ga_dict = None
        self.verb_ga_dict = None

    def load_preprocessed_data(self):
        print("Loading Preprocessed datasets...")
        processed_data_path = f'{ABS_PATH}/datasets/processed/'
        for participant in os.listdir(processed_data_path):
            pp_num = re.search(r'\d+', participant)
            participant_path = f"{processed_data_path}{participant}"
            pp_epochs = mne.read_epochs(participant_path, preload=False)
            self.epoch_list.append(pp_epochs)
            self.participant_list.append(pp_num.group())
        print("Finished loading preprocessed datasets...")

    def average_by_pp(self):
        i = 0
        for pp_epochs in self.epoch_list:
            print(f"Averaging for Participant {self.participant_list[i]}...")

            if not self.all_evoked:
                self.all_evoked = dict((cond, [pp_epochs[cond].average(picks=self.N400_electrodes, method="mean")])
                                      for cond in self.event_dict)
            else:
                for cond in self.event_dict:
                    pp_average = pp_epochs[cond].average(picks=self.N400_electrodes, method="mean")
                    self.all_evoked[cond].append(pp_average)
            i += 1

        for evoked in self.all_evoked:
            if evoked in self.noun_condition:
                self.nouns[evoked] = self.all_evoked[evoked]
            else:
                self.verbs[evoked] = self.all_evoked[evoked]

    def plot_erps(self):
        style_plot = dict(
            colors=['indianred', 'maroon', 'palegreen', 'darkseagreen', 'paleturquoise', 'darkslategray',
                    'deeppink', 'mediumvioletred'],
            split_legend=True,
            ci=.68,
            picks='ch20_MiCe',
        )

        mne.viz.plot_compare_evokeds(self.nouns, invert_y=True, title="Noun Conditions", **style_plot)

        mne.viz.plot_compare_evokeds(self.verbs, invert_y=True, title="Verb Conditions", **style_plot)

    def evoked_to_list(self):
        for condition in self.all_evoked:
            if condition == "Congruent Unambiguous Nouns":
                self.congruent_unambiguous_nouns = (self.all_evoked[condition])
            elif condition == "Congruent Ambiguous Nouns":
                self.congruent_ambiguous_nouns = (self.all_evoked[condition])
            elif condition == "Jabberwocky Unambiguous Nouns":
                self.jabberwocky_unambiguous_nouns = (self.all_evoked[condition])
            elif condition == "Jabberwocky Ambiguous Nouns":
                self.jabberwocky_ambiguous_nouns = (self.all_evoked[condition])
            elif condition == "Random Unambiguous Nouns":
                self.random_unambiguous_nouns = (self.all_evoked[condition])
            elif condition == "Random Ambiguous Nouns":
                self.random_ambiguous_nouns = (self.all_evoked[condition])

            elif condition == "Congruent Unambiguous Verbs":
                self.congruent_unambiguous_verbs = (self.all_evoked[condition])
            elif condition == "Congruent Ambiguous Verbs":
                self.congruent_ambiguous_verbs = (self.all_evoked[condition])
            elif condition == "Jabberwocky Unambiguous Verbs":
                self.jabberwocky_unambiguous_verbs = (self.all_evoked[condition])
            elif condition == "Jabberwocky Ambiguous Verbs":
                self.jabberwocky_ambiguous_verbs = (self.all_evoked[condition])
            elif condition == "Random Unambiguous Verbs":
                self.random_unambiguous_verbs = (self.all_evoked[condition])
            elif condition == "Random Ambiguous Verbs":
                self.random_ambiguous_verbs = (self.all_evoked[condition])

            else:
                print("This condition is not going in a list for grand averaging!")

    def grand_average(self):

        congruent_unambiguous_nouns_ga = mne.grand_average(self.congruent_unambiguous_nouns, interpolate_bads=False,
                                                           drop_bads=True)
        congruent_ambiguous_nouns_ga = mne.grand_average(self.congruent_ambiguous_nouns, interpolate_bads=False,
                                                         drop_bads=True)
        jabberwocky_unambiguous_nouns_ga = mne.grand_average(self.jabberwocky_unambiguous_nouns, interpolate_bads=False,
                                                             drop_bads=True)
        jabberwocky_ambiguous_nouns_ga = mne.grand_average(self.jabberwocky_ambiguous_nouns, interpolate_bads=False,
                                                           drop_bads=True)
        random_unambiguous_nouns_ga = mne.grand_average(self.random_unambiguous_nouns, interpolate_bads=False,
                                                        drop_bads=True)
        random_ambiguous_nouns_ga = mne.grand_average(self.random_ambiguous_nouns, interpolate_bads=False,
                                                      drop_bads=True)

        self.noun_ga_dict = {'Congruent_Unambiguous_Nouns': congruent_unambiguous_nouns_ga,
                             'Congruent_Ambiguous_Nouns': congruent_ambiguous_nouns_ga,
                             'Jabberwocky_Unambiguous_Nouns': jabberwocky_unambiguous_nouns_ga,
                             'Jabberwocky_Ambiguous_Nouns': jabberwocky_ambiguous_nouns_ga,
                             'Random_Unambiguous_Nouns': random_unambiguous_nouns_ga,
                             'Random_Ambiguous_Nouns': random_ambiguous_nouns_ga}

        congruent_unambiguous_verbs_ga = mne.grand_average(self.congruent_unambiguous_verbs, interpolate_bads=False,
                                                           drop_bads=True)
        congruent_ambiguous_verbs_ga = mne.grand_average(self.congruent_ambiguous_verbs, interpolate_bads=False,
                                                         drop_bads=True)
        jabberwocky_unambiguous_verbs_ga = mne.grand_average(self.jabberwocky_unambiguous_verbs, interpolate_bads=False,
                                                             drop_bads=True)
        jabberwocky_ambiguous_verbs_ga = mne.grand_average(self.jabberwocky_ambiguous_verbs, interpolate_bads=False,
                                                           drop_bads=True)
        random_unambiguous_verbs_ga = mne.grand_average(self.random_unambiguous_verbs, interpolate_bads=False,
                                                        drop_bads=True)
        random_ambiguous_verbs_ga = mne.grand_average(self.random_ambiguous_verbs, interpolate_bads=False,
                                                      drop_bads=True)

        self.verb_ga_dict = {'Congruent_Unambiguous_Verbs': congruent_unambiguous_verbs_ga,
                             'Congruent_Ambiguous_Verbs': congruent_ambiguous_verbs_ga,
                             'Jabberwocky_Unambiguous_Verbs': jabberwocky_unambiguous_verbs_ga,
                             'Jabberwocky_Ambiguous_Verbs': jabberwocky_ambiguous_verbs_ga,
                             'Random_Unambiguous_Verbs': random_unambiguous_verbs_ga,
                             'Random_Ambiguous_Verbs': random_ambiguous_verbs_ga}

    def plot_grand_averages(self):
        style_plot = dict(
            colors=['indianred', 'maroon', 'palegreen', 'darkseagreen', 'paleturquoise', 'darkslategray',
                    'deeppink', 'mediumvioletred'],
            split_legend=True,
            ci=.68,
            picks='ch20_MiCe',
        )

        mne.viz.plot_compare_evokeds(self.noun_ga_dict, invert_y=True, title="Noun Grand Average", **style_plot)

        mne.viz.plot_compare_evokeds(self.verb_ga_dict, invert_y=True, title="Verb Grand Average", **style_plot)


def main():
    print("Getting the list of all participants in the datasets...")
    connection = sqlite3.connect(DB)  # connect to your DB
    cursor = connection.cursor()  # get a cursor
    participant_list = [participant[0] for participant in cursor.execute("SELECT pp_list FROM data_table")]
    participant_set = set(participant_list)
    print("The total set of participants in the datasets:", participant_set)

    average = Average()
    average.load_preprocessed_data()
    print("The list of participants included in these averages:", average.participant_list)
    average.average_by_pp()
    average.plot_erps()
    average.evoked_to_list()
    average.grand_average()
    average.plot_grand_averages()


main()
