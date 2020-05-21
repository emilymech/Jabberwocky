import sys
import numpy as np
import pickle
import datetime
import os


class Classifier:
    #############################################################################################################################
    def __init__(self, dataset, all_folds=True, verbose=False):
        self.name = "classifer"
        self.dataset = dataset
        self.all_folds = all_folds
        self.verbose = verbose

        self.confusion_matrix = None

        self.category_sensitivity_matrix = None
        self.category_specificity_matrix = None
        self.category_precision_matrix = None
        self.category_negative_predictive_value_matrix = None
        self.category_balanced_accuracy_matrix = None
        self.category_f1_matrix = None

        self.sensitivity = None
        self.specificity = None
        self.precision = None
        self.negative_predictive_value = None
        self.balanced_accuracy = None
        self.f1 = None
        self.performance_history_list = None

    #############################################################################################################################
    def create_model_name(self, model_type='classifier'):
        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(model_type, self.start_datetime[1],
                                               self.start_datetime[2],
                                               self.start_datetime[3],
                                               self.start_datetime[4],
                                               self.start_datetime[5],
                                               self.start_datetime[6])
        self.model_path = '../models/' + self.model_name

    #############################################################################################################################
    def create_model_directory(self):
        if not os.path.isdir("../models/"):
            os.mkdir("../models/")

        # create the directory if it doesnt already exist
        if os.path.isdir(self.model_path):
            print("Model {} already exists".format(self.model_path))
            sys.exit()
        else:
            os.mkdir(self.model_path)

    #############################################################################################################################
    def create_model_config_files(self):
        f = open(self.model_path+'/config.txt', 'w')
        f.write("model: {}\n".format(self.model_name))
        f.write("dataset: {}\n".format(self.dataset.dataset_name))
        f.write("num_categories: {}\n".format(self.dataset.num_categories))
        f.write("num_instances: {}\n".format(self.dataset.num_instances))
        f.write("num_features: {}\n".format(self.dataset.num_features))
        f.write("num_folds: {}\n".format(self.dataset.num_folds))
        f.close()

        f = open(self.model_path+'/categories.csv', 'w')
        for i in range(self.dataset.num_categories):
            current_category = self.dataset.category_list[i]
            output_string = "{},{}\n".format(i,current_category)
            f.write(output_string)
        f.close()

        f = open(self.model_path+'/instances.csv', 'w')
        for i in range(self.dataset.num_instances):
            current_instance = self.dataset.instance_list[i]
            category = self.dataset.instance_category_dict[current_instance]
            output_string = "{},{},{}\n".format(i,category,current_instance)
            f.write(output_string)
        f.close()

        f = open(self.model_path+'/features.csv', 'w')
        for i in range(self.dataset.num_features):
            output_string = "{},{}\n".format(i, self.dataset.feature_list[i])
            f.write(output_string)
        f.close()
    
    #############################################################################################################################
    def save_confusion_matrix(self):
        summed_confusion_matrix = self.confusion_matrix.sum(0)
        np.savetxt(self.model_path+'/confusion_matrix.csv', summed_confusion_matrix, delimiter=",", fmt="%0.0f")

    #############################################################################################################################
    def compute_performance_summary(self):
        summed_confusion_matrix = self.confusion_matrix.sum(0)

        total_n = summed_confusion_matrix.sum()
        guess_sums = summed_confusion_matrix.sum(0)
        actual_sums = summed_confusion_matrix.sum(1)
        
        self.true_positives = np.diagonal(summed_confusion_matrix)              # hits
        self.false_positives = guess_sums - self.true_positives                   # false alarms, type 1 error
        self.false_negatives = actual_sums - self.true_positives                  # misses, type 2 error
        self.true_negatives = total_n - actual_sums - self.false_negatives        # correct rejections

        self.category_sensitivity_matrix = self.true_positives / (self.true_positives + self.false_negatives)                  # recall, or true positive rate, or hit rate, % correct when correct=yes
        self.category_specificity_matrix = self.true_negatives / (self.true_negatives + self.false_positives)                  # true negative rate, % correct when correct=no
        self.category_precision_matrix = self.true_positives / (self.true_positives + self.false_positives)                    # % correct when you guessed yes
        self.category_negative_predictive_value_matrix = self.true_negatives / (self.true_negatives + self.false_negatives)    # % correct when you guessed no
        
        self.category_balanced_accuracy_matrix = (self.category_sensitivity_matrix + self.category_specificity_matrix)/2                        
        self.category_f1_matrix = 2*self.true_positives / (2*self.true_positives + self.false_positives + self.false_negatives)

        category_weights = np.array(self.dataset.category_size_list)
        category_weights = category_weights/category_weights.sum()

        self.sensitivity = np.average(self.category_sensitivity_matrix, weights=category_weights)
        self.specificity = np.average(self.category_specificity_matrix, weights=category_weights)
        self.precision = np.average(self.category_precision_matrix, weights=category_weights)
        self.negative_predictive_value = np.average(self.category_negative_predictive_value_matrix, weights=category_weights)
        self.balanced_accuracy = np.average(self.category_balanced_accuracy_matrix, weights=category_weights)
        self.f1 = np.average(self.category_f1_matrix, weights=category_weights)
        
        if self.verbose:
            print("Confusion Matrix")
            print(summed_confusion_matrix)
            print()
            print("     guess sums", guess_sums)
            print("     actual sums", actual_sums)
            print()
            print("     true positives", self.true_positives)
            print("     true negatives", self.true_negatives)
            print("     false_positives", self.false_positives)
            print("     false_negatives", self.false_negatives)
            print()
            print("                  sensitivity {:0.3f}".format(self.sensitivity), self.category_sensitivity_matrix)
            print("                  specificity {:0.3f}".format(self.specificity), self.category_specificity_matrix)
            print("                    precision {:0.3f}".format(self.precision), self.category_precision_matrix)
            print("    negative predictive value {:0.3f}".format(self.negative_predictive_value), self.category_negative_predictive_value_matrix)
            print("            balanced accuracy {:0.3f}".format(self.balanced_accuracy), self.category_balanced_accuracy_matrix)
            print("                           f1 {:0.3f}".format(self.f1), self.category_f1_matrix)

    #############################################################################################################################
    def save_performance_summary(self):
        f = open(self.model_path+'/performance_summary.csv', 'w')
        self.write_count_performance_result(f, "true_positives", self.true_positives)
        self.write_count_performance_result(f, "true_negatives", self.true_negatives)
        self.write_count_performance_result(f, "false_positives", self.false_positives)
        self.write_count_performance_result(f, "false_negatives", self.false_negatives)

        self.write_rate_performance_result(f, "sensitivity", self.sensitivity, self.category_sensitivity_matrix)
        self.write_rate_performance_result(f, "specificity", self.specificity, self.category_specificity_matrix)
        self.write_rate_performance_result(f, "precision", self.precision, self.category_precision_matrix)
        self.write_rate_performance_result(f, "negative_predictive_value", self.negative_predictive_value, self.category_negative_predictive_value_matrix)
        self.write_rate_performance_result(f, "balanced_accuracy", self.balanced_accuracy, self.category_balanced_accuracy_matrix)
        self.write_rate_performance_result(f, "f1", self.f1, self.category_f1_matrix)
        f.close()

    ############################################################################################################
    def calculate_and_save_current_epoch(self, i):
        self.create_confusion_matrix()
        self.compute_performance_summary()

        if self.save_ba_history:
            f = open(self.model_path+'/ba_history.csv', 'a')
            vector_string = ",".join(np.char.mod('%0.3f', self.category_balanced_accuracy_matrix))
            f.write("{},{:0.3f},{}\n".format(i+1, self.balanced_accuracy, vector_string))
            f.close()
        
        if self.save_f1_history:
            f = open(self.model_path+'/f1_history.csv', 'a')
            vector_string = ",".join(np.char.mod('%0.3f', self.category_f1_matrix))
            f.write("{},{:0.3f},{}\n".format(i+1, self.f1, vector_string))
            f.close()

    #############################################################################################################################
    def save_model(self):
        pickle_file = open(self.model_path + '/model_object.p', 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()
    
    #############################################################################################################################
    @staticmethod
    def write_count_performance_result(f, label, data):
        data_string = ",".join(np.char.mod('%i', data))
        f.write("{},{:0.0f},{}\n".format(label, data.sum(), data_string))

    #############################################################################################################################
    @staticmethod
    def write_rate_performance_result(f, label, value, vector):
        vector_string = ",".join(np.char.mod('%0.3f', vector))
        f.write("{},{:0.3f},{}\n".format(label, value, vector_string))
    
    ############################################################################################################
    def save_full_results(self):
        f = open(self.model_path+'/instance_results.csv', 'w')
        header_list = ["fold","item","actual_category","guess_category","correct"]
        for i in range(self.dataset.num_categories):
            header_list.append(self.dataset.category_list[i]+"_sim")
        header_string = ','.join(header_list)
        f.write(header_string+'\n')
        for i in range(len(self.full_result_list)):
            current_results = [str(i) for i in self.full_result_list[i][0]]
            current_results = ",".join(current_results)
            current_acts = ",".join(np.char.mod('%0.3f', self.full_result_list[i][1]))
            output_string = ",".join(current_results)
            f.write("{},{}\n".format(current_results, current_acts))
        f.close()