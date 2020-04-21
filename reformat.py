import pickle as pkl
from pathlib import Path
from src.dataset import Participant, get_data_list


def save_pickle(data_dict):
    absolute_path = Path(__file__).parent.absolute()
    filename = '{}{}{}'.format(absolute_path, '/data/reformatted/', 'reformatted_dataset.pkl')
    print(filename)
    with open(filename, 'wb') as outfile:
        pkl.dump(data_dict, outfile)


def main():
    data_dict = {}
    file_list = get_data_list()

    for participant in file_list:
        pp = Participant()
        pp.load_raw_data_file(participant)
        pp.preprocess_raw_data()
        pp.save_preprocessed_data()
        data_dict[pp.pp] = pp.data

    save_pickle(data_dict)
    print("Finished reformatting all Pp")


main()
