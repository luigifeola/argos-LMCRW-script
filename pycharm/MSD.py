import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from termcolor import colored

window_size = 1
debug_time = False
time_window = 61


def MSD_mean(main_folder, folder_experiments):
    for dirName, subdirList, fileList in os.walk(os.path.join(main_folder, folder_experiments)):
        print(colored("DirName:", 'blue'), dirName)

        num_robots = -1
        for fileName in fileList:
            print(colored("\tfileName:", 'blue'), fileName)

            if 'real' in fileName:
                exp_type = 'real'
            else:
                exp_type = 'simulated'

            elements = fileName.split('_')
            for e in elements:
                if e.startswith('robots'):
                    num_robots = int(e.split('#')[1])
                    print('\tnum_robots:', num_robots)

            if num_robots == -1:
                print('Error!!! num_robots not a right value')
                exit(-1)

            np_position = np.load(os.path.join(main_folder, folder_experiments, fileName))

            if debug_time:
                print('\tWARNING: remove after test')
                np_position = np_position[:time_window]

            print('\tnp_position.shape:', np_position.shape)



            msd_matrix = np.array([])
            for f in range(window_size, np_position.shape[0], window_size):
                #     print('tf: {}, ti: {}'.format(f, f - window_size))
                tf = np_position[f]
                ti = np_position[f - window_size]
                #     print('tf.shape:', tf.shape)
                sq_distance = np.sum((tf - ti) ** 2, axis=1)
                msd = np.true_divide(sq_distance, window_size ** 2)

                msd_matrix = np.row_stack([msd_matrix, msd]) if msd_matrix.size else msd

            msd_matrix_mean = np.mean(msd_matrix, axis=1)
            msd_matrix_std = np.std(msd_matrix, axis=1)
            # print('\t msd_matrix.shape', msd_matrix.shape)


            fig, ax = plt.subplots()
            plt.grid()

            times = np.arange(window_size, msd_matrix_mean.size + window_size, window_size) * 10
            ax.plot(times, msd_matrix_mean, marker='o')
            ax.fill_between(times, msd_matrix_mean + msd_matrix_std, msd_matrix_mean - msd_matrix_std, alpha=0.5)

            plt.xticks(np.arange(0, msd_matrix_mean.size + window_size, 10) * 10,
                       labels=np.arange(0, msd_matrix_mean.size + window_size, 10) // 10)
            plt.ylim(0, 0.015)
            plt.title(fileName[:-4])
            plt.xlabel('time(s)' + r"$ 10^2$")

            fig_name = exp_type + '_msd_mean_per_experiment_' + fileName[:-4] + '.png'
            print(colored("\tSaving figure:", 'blue'), fig_name)
            save_dir = os.path.join('Plots', folder_experiments.split('/')[1], sys._getframe().f_code.co_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(os.path.join(save_dir, fig_name))
            # plt.show()
            plt.close()


def MSD_per_run(main_folder, folder_experiments):
    for dirName, subdirList, fileList in os.walk(os.path.join(main_folder, folder_experiments)):
        print(colored("DirName:", 'blue'), dirName)

        num_robots = -1
        for fileName in fileList:
            print(colored("\tfileName:", 'blue'), fileName)

            if 'real' in fileName:
                exp_type = 'real'
            else:
                exp_type = 'simulated'

            elements = fileName.split('_')
            for e in elements:
                if e.startswith('robots'):
                    num_robots = int(e.split('#')[1])
                    print('\tnum_robots:', num_robots)

            if num_robots == -1:
                print('Error!!! num_robots not a right value')
                exit(-1)

            np_position = np.load(os.path.join(main_folder, folder_experiments, fileName))

            if debug_time:
                print('\tWARNING: remove after test')
                np_position = np_position[:time_window]

            print('\tnp_position.shape:', np_position.shape)

            plt.grid()

            for run in range(0, np_position.shape[1], num_robots):
                # print('\t', run, run + num_robots)
                single_run = np_position[:, run:run + num_robots, :]

                msd_matrix = np.array([])
                for f in range(window_size, np_position.shape[0], window_size):
                    #     print('tf: {}, ti: {}'.format(f, f - window_size))
                    tf = single_run[f]
                    ti = single_run[f - window_size]
                    #     print('tf.shape:', tf.shape)
                    sq_distance = np.sum((tf - ti) ** 2, axis=1)
                    msd = np.true_divide(sq_distance, window_size ** 2)

                    msd_matrix = np.row_stack([msd_matrix, msd]) if msd_matrix.size else msd

                msd_matrix_mean = np.mean(msd_matrix, axis=1)
                msd_matrix_std = np.std(msd_matrix, axis=1)
                # print('\t msd_matrix.shape', msd_matrix.shape)

                times = np.arange(window_size, msd_matrix_mean.size + window_size, window_size) * 10
                plt.plot(times, msd_matrix_mean, label=(run // num_robots) + 1, marker='o')
                plt.legend()
                plt.xticks(np.arange(0, msd_matrix_mean.size + window_size, 10) * 10,
                           labels=np.arange(0, msd_matrix_mean.size + window_size, 10) // 10)
                plt.ylim(0, 0.015)
                plt.title(fileName[:-4])
                plt.xlabel('time(s)' + r"$ 10^2$")

            figName = exp_type + '_msd_mean_per_run_' + fileName[:-4] + '.png'
            print(colored("\tSaving figure:", 'blue'), figName)
            saveDir = os.path.join('Plots', folder_experiments.split('/')[1], sys._getframe().f_code.co_name)

            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            plt.savefig(os.path.join(saveDir, figName))
            plt.close()
            # plt.show()


def MSD_per_run_per_robot(main_folder, folder_experiments):
    for dirName, subdirList, fileList in os.walk(os.path.join(main_folder, folder_experiments)):
        print(colored("DirName:", 'blue'), dirName)

        num_robots = -1
        for fileName in fileList:
            print(colored("\tfileName:", 'blue'), fileName)


            # TODO: be careful to return at real exp
            if 'real' in fileName:
                exp_type = 'real'
            else:
                # exp_type = 'simulated'
                continue

            elements = fileName.split('_')
            for e in elements:
                if e.startswith('robots'):
                    num_robots = int(e.split('#')[1])
                    print('\t num_robots:', num_robots)

            if num_robots == -1:
                print('Error!!! num_robots not a right value')
                exit(-1)

            np_position = np.load(os.path.join(main_folder, folder_experiments, fileName))

            if debug_time:
                print('\tWARNING: remove after test')
                np_position = np_position[:time_window]

            print('\t np_position.shape:', np_position.shape)


            for run in range(0, np_position.shape[1], num_robots):
                print('\t run:', run//num_robots)
                single_run = np_position[:, run:run + num_robots, :]

                msd_matrix = np.array([])
                for f in range(window_size, np_position.shape[0], window_size):
                    # if f > 60:
                    #     break
                    # print('tf: {}, ti: {}'.format(f, f - window_size))
                    tf = single_run[f]
                    ti = single_run[f - window_size]
                    #     print('tf.shape:', tf.shape)
                    sq_distance = np.sum((tf - ti) ** 2, axis=1)
                    msd = np.true_divide(sq_distance, window_size ** 2)

                    msd_matrix = np.row_stack([msd_matrix, msd]) if msd_matrix.size else msd



                print('\t msd_matrix.shape:', msd_matrix.shape)
                msd_mean = np.mean(msd_matrix, axis=0)
                msd_std = np.std(msd_matrix, axis=0)
                print('\t msd_matrix_mean.shape', msd_mean.shape)

                times = np.arange(num_robots)
                plt.xticks(times)
                plt.errorbar(times, msd_mean, msd_std, fmt='o', solid_capstyle='projecting', capsize=5)
                plt.ylim(0, 0.015)
                plt.grid(alpha=0.5, linestyle=':')
                plt.title('run ' + str(run//num_robots) + ' ' + fileName[:-4])

                plt.axhline(y=0.004, color='r', linestyle='-')

                figName = 'msd_per_robot_' + fileName[:-13] + 'run#' + str(run//num_robots) + '_' + exp_type + '.png'
                saveDir = os.path.join('Plots', folder_experiments.split('/')[1], sys._getframe().f_code.co_name)
                print(colored("\tSaving figure at:", 'blue'), os.path.join(saveDir, figName))

                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)

                plt.savefig(os.path.join(saveDir, figName))
                plt.close()
                # plt.show()

        # break
