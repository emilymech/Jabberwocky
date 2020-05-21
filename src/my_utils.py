import numpy as np
import time


def calculate_distance(a, b, m=2):
    distance = np.linalg.norm(a-b, ord=m)
    return distance


def calculate_cosine(a, b):
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return similarity


def calculate_correlation(a, b):
    similarity = np.corrcoef(a,b)[0,1]
    return similarity


def row_sum_normalize(input_matrix):
    row_sums = input_matrix.sum(1)
    if not np.all(row_sums):
        for i in range(len(row_sums)):
            if row_sums[i] == 0:
                print("     Warning! Row {} has sum of zero. Setting all norm values for this row to 0.".format(i))
                row_sums[i] = 1
    output_matrix = (input_matrix.transpose() / row_sums).transpose()
    return output_matrix


def column_sum_normalize(input_matrix):
    col_sums = input_matrix.sum(0)
    if not np.all(col_sums):
        for i in range(len(col_sums)):
            if col_sums[i] == 0:
                print("     Warning! Column {} has sum of zero. Setting all norm values for this row to 0.".format(i))
                col_sums[i] = 1
    output_matrix = input_matrix / col_sums
    return output_matrix


def row_zscore_normalize(input_matrix):
    row_means = input_matrix.mean(1)
    row_stdevs = input_matrix.std(1)
    if not np.all(row_stdevs):
        for i in range(len(row_stdevs)):
            if row_stdevs[i] == 0:
                print("     Warning! Row {} has stdev of zero. Setting all norm values for this row to 0.".format(i))
                row_stdevs[i] = 1
    mean_diffs = input_matrix.transpose()-row_means
    output_matrix = (mean_diffs / row_stdevs).transpose()
    return output_matrix


def column_zscore_normalize(input_matrix):
    col_means = input_matrix.mean(0)
    col_stdevs = input_matrix.std(0)
    if not np.all(col_stdevs):
        for i in range(len(col_stdevs)):
            if col_stdevs[i] == 0:
                print("     Warning! Column {} has stdev of zero. Setting all norm values for this row to 0.".format(i))
                col_stdevs[i] = 1
    output_matrix = (input_matrix-col_means) / col_stdevs
    return output_matrix


def row_log_entropy_normalize(input_matrix):
    num_rows = len(input_matrix[:,0])
    num_columns = len(input_matrix[0,:])
    column_sums = input_matrix.sum(0)
    row_sums = input_matrix.sum(1)
    matrix_sum = input_matrix.sum()

    output_matrix = np.zeros([num_rows, num_columns], float)
    start_time = time.time()
    for i in range(num_rows):
        for j in range(num_columns):
            if row_sums[i] == 0:
                print("     Warning! Row {} has Sum of zero. Setting norm values for this item to 0.".format(i))
                output_matrix[i,j] = 0
            elif column_sums[j] == 0:
                print("     Warning! Column {} has Sum of zero. Setting norm values for this item to 0.".format(i))
                output_matrix[i,j] = 0
            elif input_matrix[i,j] == 0:
                output_matrix[i,j] = 0
            else:
                output_matrix[i,j] = np.log((input_matrix[i,j] / matrix_sum) /
                                            ((row_sums[i]/matrix_sum) * (column_sums[j]/matrix_sum)))
        if (i+1) % 100 == 0:
            took = time.time() - start_time
            print("     Finished {}/{} rows. Took {:0.2f} sec.".format(i+1, num_rows, took))
            start_time = time.time()
    return output_matrix


def ppmi_normalize(input_matrix):

    num_rows = len(input_matrix[:,0])
    num_columns = len(input_matrix[0,:])
    column_sums = input_matrix.sum(0)
    row_sums = input_matrix.sum(1)
    matrix_sum = input_matrix.sum()
    
    output_matrix = np.zeros([num_rows, num_columns], float)
    start_time = time.time()
    for i in range(num_rows):
        for j in range(num_columns):
            if row_sums[i] == 0:
                print("      Warning! Row {} has Sum of zero. Setting norm values for this item to 0.".format(str(i)))
                output_matrix[i,j] = 0
            elif column_sums[j] == 0:
                print("     Warning! Column {} has Sum of zero. Setting norm values for this item to 0.".format(str(i)))
                output_matrix[i,j] = 0
            elif input_matrix[i,j] == 0:
                output_matrix[i,j] = 0
            else:
                output_matrix[i,j] = np.log((input_matrix[i, j] / matrix_sum) /
                                            ((row_sums[i]/matrix_sum) * (column_sums[j]/matrix_sum)))
                if output_matrix[i,j] < 0:
                    output_matrix[i,j] = 0
        if (i+1) % 100 == 0:
            took = time.time() - start_time
            print("     Finished {}/{} rows. Took {:0.2f} sec.".format(i+1, num_rows, took))
            start_time = time.time()
    return output_matrix
