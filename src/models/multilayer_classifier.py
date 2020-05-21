import numpy as np
from src.models import classifier
from src import config
import sys
import random


class NumpyMultilayerClassifier(classifier.Classifier):
    def __init__(self, dataset, all_folds=True, verbose=False,
                 hidden_size=config.MultilayerClassifier.hidden_size,
                 learning_rate=config.MultilayerClassifier.learning_rate,
                 num_epochs=config.MultilayerClassifier.num_epochs,
                 weight_init_stdev=config.MultilayerClassifier.weight_init_stdev,
                 hidden_activation_function=config.MultilayerClassifier.hidden_activation_function,
                 output_freq=config.MultilayerClassifier.output_freq,
                 save_f1_history=config.MultilayerClassifier.save_f1_history,
                 save_ba_history=config.MultilayerClassifier.save_ba_history):
        super().__init__(dataset, all_folds, verbose)

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_stdev = weight_init_stdev
        self.hidden_activation_function = hidden_activation_function
        self.output_freq = output_freq
        self.save_f1_history = save_f1_history
        self.save_ba_history = save_ba_history

        self.input_size = self.dataset.num_features
        self.output_size = self.dataset.num_categories
        self.y_bias_list = []
        self.y_h_weight_list = []
        self.h_bias_list = []
        self.h_x_weight_list = []

        print("\nTraining Multilayer Classifier of Size {}-{}-{} on {} for {} epochs"
              .format(self.input_size, self.hidden_size, self.output_size, self.dataset.dataset_name, self.num_epochs))

        self.create_model_name('mlc_eeg')
        self.create_model_directory()
        self.create_model_config_files()
        self.add_model_config_details()

        self.init_weights()

        self.train()

        self.create_confusion_matrix()
        self.save_confusion_matrix()
        self.compute_performance_summary()
        self.save_performance_summary()
        self.save_model()
        self.save_full_results()

    def add_model_config_details(self):
        f = open(self.model_path + '/config.csv', 'a')
        f.write("learning_rate: {}".format(self.learning_rate))
        f.write("num_epochs: {}".format(self.num_epochs))
        f.write("weight_init_stdev: {}".format(self.weight_init_stdev))
        f.write("hidden_size: {}".format(self.hidden_size))
        f.write("hidden_activation_function: {}".format(self.hidden_activation_function))
        f.close()

    def init_weights(self):
        for i in range(self.dataset.num_folds):
            self.y_bias_list.append(np.random.normal(0, self.weight_init_stdev,
                                                     [self.output_size]))
            self.y_h_weight_list.append(np.random.normal(0, self.weight_init_stdev,
                                                         [self.output_size, self.hidden_size]))
            self.h_bias_list.append(np.random.normal(0, self.weight_init_stdev,
                                                     [self.hidden_size]))
            self.h_x_weight_list.append(np.random.normal(0, self.weight_init_stdev,
                                                         [self.hidden_size, self.input_size]))

    def train(self):
        print("    Training for {} Epochs".format(self.num_epochs))
        sse_matrix = np.zeros([self.num_epochs, self.dataset.num_folds])

        if self.save_ba_history or self.save_f1_history:
            self.calculate_and_save_current_epoch(-1)

        for i in range(self.num_epochs):

            for j in range(self.dataset.num_folds):
                if i != j:

                    current_fold = self.dataset.training_fold_list[j].copy()
                    random.shuffle(current_fold)
                    y_h_weights = np.copy(self.y_h_weight_list[j])
                    y_bias = np.copy(self.y_bias_list[j])
                    h_x_weights = np.copy(self.h_x_weight_list[j])
                    h_bias = np.copy(self.h_bias_list[j])
                    for k in range(len(current_fold)):

                        current_instance = current_fold[k]
                        instance_index = self.dataset.instance_index_dict[current_instance]
                        current_category = self.dataset.instance_category_dict[current_instance]
                        category_index = self.dataset.category_index_dict[current_category]
                        x = self.dataset.instance_feature_matrix[instance_index, :]
                        y = self.dataset.instance_category_matrix[instance_index, :]
                        o, h = self.forward(x, y_bias, y_h_weights, h_bias, h_x_weights)
                        cost = self.calculate_cost(y, o)
                        sse_matrix[i,j] += (cost**2).sum()
                        y_bias, y_h_weights, h_bias, h_x_weights = \
                            self.update_weights(x, o, h, cost, y_bias, y_h_weights, h_bias, h_x_weights)

                    self.y_h_weight_list[j] = y_h_weights
                    self.y_bias_list[j] = y_bias
                    self.h_x_weight_list[j] = h_x_weights
                    self.h_bias_list[j] = h_bias

            if (i+1) % self.output_freq == 0:
                print("        Finished Epoch", i+1, sse_matrix[i, :])
                if self.save_ba_history or self.save_f1_history:
                    self.calculate_and_save_current_epoch(i)

    def create_confusion_matrix(self):
        self.confusion_matrix = np.zeros([self.dataset.num_folds,
                                          self.dataset.num_categories,
                                          self.dataset.num_categories])
        self.full_result_list = []  # [[fold, item, actual_cat, guess_cat, correct, sims], ] 

        for i in range(self.dataset.num_folds):
            current_fold = self.dataset.training_fold_list[i]
            for j in range(len(current_fold)):
                current_instance = current_fold[j]
                instance_index = self.dataset.instance_index_dict[current_instance]
                correct_category = self.dataset.instance_category_dict[current_instance]
                correct_category_index = self.dataset.category_index_dict[correct_category]
                x = self.dataset.instance_feature_matrix[instance_index, :]
                y = self.dataset.instance_category_matrix[instance_index, :]
                o, h = self.forward(x, self.y_bias_list[i], self.y_h_weight_list[i], self.h_bias_list[i], self.h_x_weight_list[i])
                guess_category_index = np.argmax(o)
                guess_category = self.dataset.category_list[guess_category_index]
                if guess_category_index == correct_category_index:
                    correct = 1
                else:
                    correct = 1
                self.confusion_matrix[i, correct_category_index, guess_category_index] += 1
                self.full_result_list.append([[i, current_instance, correct_category, guess_category, correct], o])

    def forward(self, x, y_bias, y_h_weights, h_bias, h_x_weights):
        z_h = np.dot(h_x_weights, x) + h_bias

        if self.hidden_activation_function == 'sigmoid':
            h = 1/(1+np.exp(-z_h))
        elif self.hidden_activation_function == 'tanh':
            h = np.tanh(z_h)
        else:
            print("ERROR: invalid hidden activation function", self.hidden_activation_function)
            sys.exit()

        z_o1 = np.dot(y_h_weights, h) 
        z_o2 = z_o1 + y_bias
        o = 1/(1+np.exp(-z_o2))
        return o, h

    def calculate_cost(self, y, o):
        return y - o

    def update_weights(self, x, o, h, cost, y_bias, y_h_weights, h_bias, h_x_weights):
        y_delta = cost * 1/(1+np.exp(-o)) * (1 - 1/(1+np.exp(-o)))
        h_cost = np.dot(y_delta, y_h_weights)
        if self.hidden_activation_function == 'sigmoid':
            h_delta = h_cost * 1/(1+np.exp(-h)) * (1 - 1/(1+np.exp(-h)))
        elif self.hidden_activation_function == 'tanh':
            h_delta = h_cost * (1.0 - np.tanh(h)**2)
        else:
            print("ERROR: invalid hidden activation function", self.hidden_activation_function)
            sys.exit()

        y_bias += y_delta * self.learning_rate
        y_h_weights += (np.dot(y_delta.reshape(len(y_delta), 1), h.reshape(1, len(h))) * self.learning_rate)
        h_bias += h_delta * self.learning_rate
        h_x_weights += (np.dot(h_delta.reshape(len(h_delta), 1), x.reshape(1, len(x))) * self.learning_rate)
        return y_bias, y_h_weights, h_bias, h_x_weights
