from src.models import multilayer_classifier
import numpy as np


def main():
    np.set_printoptions(precision=3)

    jabberwocky_dataset = dm_feature_categories.DMFeatureCategories()
    jabberwocky_dataset.load_dataset('dmfc')
    jabberwocky_dataset.create_training_folds()
    jabberwocky_multilayer_model = multilayer_classifier.NumpyMultilayerClassifier(jabberwocky_dataset, verbose=False)
    jabberwocky_multilayer_model.compute_full_similarity_matrix('correlation')
    jabberwocky_multilayer_model.get_nearest_neighbors(20)


main()
