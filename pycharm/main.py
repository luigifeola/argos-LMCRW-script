# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import config
import utils
from MSD import *
import numpy as np
import matplotlib.pyplot as plt
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # folder_experiments = 'results/OCT_2021/pkl_pos'
    # folder_experiments = 'results/JAN_2022/pkl_pos'
    # folder_experiments = 'results/FEB_2022/pkl_pos'
    # folder_experiments = 'results/FEB_2022_10#robots_TEST/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_TEST/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_TEST1/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_rho#0.0_TEST/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_rho#0.9_oldTable/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_rho#0.9_oldTable_noRing/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_rho#0.9_newTable_exactLinVel/pkl_pos'
    # folder_experiments = 'results/FEB_2022_5#robots_rho#0.9_newTable_BUG/pkl_pos'
    folder_experiments = 'results/FEB_2022_5#robots_rho#0.9_oldTable_BUG/pkl_pos'

    print(colored("--------------MSD_per_run------------------------------------------", 'green'))
    MSD_per_run(main_folder, folder_experiments)
    print(colored("--------------MSD_mean---------------------------------------------", 'green'))
    MSD_mean(main_folder, folder_experiments)
    print(colored("--------------MSD_per_run_per_robot--------------------------------", 'green'))
    MSD_per_run_per_robot(main_folder, folder_experiments)

