import os
import sqlite3
from pathlib import Path
from src.dataset import Participant


VERBOSE = False
absolute_path = Path(__file__).parent.absolute()
path = '{}{}'.format(absolute_path, '/data/raw')


def get_data_list():
    print("Getting data list...")
    file_list = []
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        if file[0] != '_':
            file_list.append(filepath)
    return file_list


def data_to_database(file_list):
    # create database
    db_name = f'{absolute_path}{"/reformatted_data.sqlite"}'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    try:
        c.execute("CREATE TABLE data_table (pp_list str, ch0_HE float, ch1_lhe float, ch2_rhe float, "
                  "ch3_LE float, ch4_A2 float, ch5_MiPf float, ch6_LLPf float, ch7_RLPf float, ch8_LMPf float, "
                  "ch9_RMPf float, ch10_LDF float, ch11_RDFr float, ch12_LMFr float, ch13_RMFr float, "
                  "ch14_LLFr float, ch15_RLFr float, ch16_LMCe float, ch17_RMCe float, ch18_LDCe float, "
                  "ch19_RDCe float, ch20_MiCe float, ch21_MiPa float, ch22_LLTe float, ch23_RLTe float, "
                  "ch24_LDPa float, ch25_RDPa float, ch26_LLOc float, ch27_RLOc float, ch28_LMOc float, "
                  "ch29_RMOc float, ch30_MiOc float, event_list float)")

        print('Table created!')

    except sqlite3.OperationalError:
        print("Table already exists!")

    for participant in file_list:
        pp = Participant()
        pp.load_raw_data_file(participant)
        pp.reformat_raw_data()

        print(f"Adding values to database for Pp {pp.pp}")

        for i in range(len(pp.ch0_HE)):
            pp.pp_list.append(pp.pp)
            values = (pp.pp_list[i], pp.ch0_HE[i], pp.ch1_lhe[i], pp.ch2_rhe[i], pp.ch3_LE[i], pp.ch4_A2[i],
                      pp.ch5_MiPf[i], pp.ch6_LLPf[i], pp.ch7_RLPf[i], pp.ch8_LMPf[i], pp.ch9_RMPf[i],
                      pp.ch10_LDFr[i], pp.ch11_RDFr[i], pp.ch12_LMFr[i], pp.ch13_RMFr[i], pp.ch14_LLFr[i],
                      pp.ch15_RLFr[i], pp.ch16_LMCe[i], pp.ch17_RMCe[i], pp.ch18_LDCe[i], pp.ch19_RDCe[i],
                      pp.ch20_MiCe[i], pp.ch21_MiPa[i], pp.ch22_LLTe[i], pp.ch23_RLTe[i], pp.ch24_LDPa[i],
                      pp.ch25_RDPa[i], pp.ch26_LLOc[i], pp.ch27_RLOc[i], pp.ch28_LMOc[i], pp.ch29_RMOc[i],
                      pp.ch30_MiOc[i], pp.event_list[i])

            command = "INSERT INTO data_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, " \
                      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

            if VERBOSE:
                print(values)
            c.execute(command, values)

    conn.commit()  # save changes
    conn.close()

    print('Saved changes and closed database.')


def main():
    file_list = get_data_list()
    data_to_database(file_list)


main()
