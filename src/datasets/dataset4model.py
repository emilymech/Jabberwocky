import numpy as np
import sys
import pickle
import random
import datetime

from src import config, my_utils


class InstanceFeatureDataset:
    def __init__(self):
        self.dataset_type = 'EEG'
        self.dataset_name = None
        self.dataset_path = None
        self.start_datetime = None

        self.num_categories = None
        self.category_size_list = None
        self.category_list = None
        self.category_index_dict = None

        self.num_instances = None
        self.instance_list = None
        self.instance_index_dict = None

        self.num_features = None
        self.feature_list = None
        self.feature_index_dict = None

        self.instance_category_dict = None
        self.instance_category_matrix = None
        self.instance_feature_matrix = None

        self.num_folds = None
        self.training_fold_list = None

        self.row_dimension_loadings = None
        self.singular_values = None
        self.column_dimension_loadings = None

        self.normalization_method = None

    def __str__(self):
        output_string = "Jabberwocky Dataset for Model"
        return output_string

    def create_dataset(self):

        self.num_categories = 2
        self.category_size_list = [2, 2]
        self.num_instances = 4
        self.num_features = 2
        self.category_list = ['a', 'b']
        self.category_index_dict = {'a': 0, 'b': 1}
        self.instance_list = ['a1', 'a2', 'b1', 'b2']
        self.instance_index_dict = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3}
        self.feature_list = ['F1', 'F2']
        self.feature_index_dict = {'F1': 0, 'F2': 1}
        self.instance_category_dict = {'a1': 'a', 'a2': 'a', 'b1': 'b', 'b2': 'b'}

        self.name_dataset()

    def name_dataset(self, name='jabberwocky'):
        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.dataset_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(name,
                                                                   self.num_categories, self.num_instances,
                                                                   self.num_features,
                                                                   self.start_datetime[1],
                                                                   self.start_datetime[2],
                                                                   self.start_datetime[3],
                                                                   self.start_datetime[4],
                                                                   self.start_datetime[5],
                                                                   self.start_datetime[6])
        self.dataset_path = '../datasets/' + self.dataset_name

    def load_dataset(self, dataset_name):
        pass

    def create_training_folds(self, num_folds=config.JabberwockyDataset.num_folds, remove_unknowns=True):
        shuffled_instances = self.instance_list.copy()
        random.shuffle(shuffled_instances)

        self.training_fold_list = []
        self.num_folds = num_folds

        k, m = divmod(self.num_instances, self.num_folds)
        self.training_fold_list = list(
            (shuffled_instances[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.num_folds)))

    def create_instance_category_matrix(self):
        self.instance_category_matrix = np.zeros([self.num_instances, self.num_categories], float)

        for i in range(self.num_instances):
            category = self.instance_category_dict[self.instance_list[i]]
            category_index = self.category_index_dict[category]
            self.instance_category_matrix[i, category_index] = 1

    def normalize_instance_feature_matrix(self, normalization_method):
        if normalization_method is None:
            pass
        elif normalization_method == 'rowsums':
            self.instance_feature_matrix = my_utils.row_sum_normalize(self.instance_feature_matrix)
        elif normalization_method == 'columnsums':
            self.instance_feature_matrix = my_utils.column_sum_normalize(self.instance_feature_matrix)
        elif normalization_method == 'rowzscore':
            self.instance_feature_matrix = my_utils.row_zscore_normalize(self.instance_feature_matrix)
        elif normalization_method == 'columnzscore':
            self.instance_feature_matrix = my_utils.column_zscore_normalize(self.instance_feature_matrix)
        elif normalization_method == 'rowlogentropy':
            self.instance_feature_matrix = my_utils.row_log_entropy_normalize(self.instance_feature_matrix)
        elif normalization_method == 'ppmi':
            self.instance_feature_matrix = my_utils.ppmi_normalize(self.instance_feature_matrix)
        else:
            print("ERROR: Unrecognized normalization_method", self.normalization_method)
            sys.exit()

    def svd_instance_feature_matrix(self):
        self.row_dimension_loadings, self.singular_values, self.column_dimension_loadings = \
            np.linalg.svd(self.instance_feature_matrix)
        with open(self.dataset_path + "/singular_values.txt", 'w') as f:
            print(self.singular_values)
            for i in range(len(self.singular_values)):
                f.write(str(self.singular_values[i]) + '\n')

        np.savetxt(self.dataset_path + '/row_dimension_loadings.csv', self.row_dimension_loadings, delimiter=",",
                   fmt="%0.4f")
        np.savetxt(self.dataset_path + '/column_dimension_loadings.csv', self.column_dimension_loadings, delimiter=",",
                   fmt="%0.4f")
