class JabberwockyDataset:
    num_folds = 4


class MultilayerClassifier:
    hidden_size = 64
    learning_rate = 0.10
    weight_init_stdev = 0.01
    num_epochs = 5000
    hidden_activation_function = 'tanh'   # can be 'sigmoid' or 'tanh'
    output_freq = 10
    save_f1_history = True
    save_ba_history = True
