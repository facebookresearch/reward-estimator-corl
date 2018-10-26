# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import re
import os
import csv
import pickle
import argparse
import time

from visdom import Visdom
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))
    if len(infiles) < 1:
        return [None, None]

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def visdom_plot(viz, win, folder, game, name, bin_size=100, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    # Ugly hack to detect atari
    if game.find('NoFrameskip') > -1:
        plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
                   ["1M", "2M", "4M", "6M", "8M", "10M"])
        plt.xlim(0, 10e6)
    else:
        plt.xticks([1e5, 2e5, 4e5, 6e5, 8e5, 1e5],
                   ["0.1M", "0.2M", "0.4M", "0.6M", "0.8M", "1M"])
        plt.xlim(0, 1e6)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)


def bin_y(y, bin_size=1):
    return [sum(y[i - min(bin_size, i - i): i + 1]) / min(bin_size, i - 1) for i in range(len(y))]


def align_data(xs, ys, tick_interval=1000, max_x=1000000):
    min_x = min(min([data_x[-1] for data_x in xs]), max_x)
    new_xs = np.arange(0, int(min_x), tick_interval)
    new_data = np.empty((len(xs), new_xs.shape[0]))
    for i, (xx, yy) in enumerate(zip(xs, ys)):
        new_data[i] = np.interp(new_xs, xx, yy)
    return new_data, new_xs

def visdom_plot_batched(viz, win, x, y, title, name, bin_size=1, smooth=1, x_label='Number of Epochs',
                        y_label='Training Error', max_x=None, save_loc=None, ylim=None, log=False):
    if y is None:
        return win
    fig = plt.figure()

    if type(name) is list:
        assert len(x) == len(y) and len(y) == len(name)
        methods = {}
        min_x = max_x
        y_min = None
        y_max = None
        for xx, yy, n in zip(x, y, name):
            if len(xx) < 1:
                continue
            min_x = xx[-1] if xx[-1] < min_x else min_x
            if n in methods:
                methods[n]['x'].append(xx)
                methods[n]['y'].append(yy)
            else:
                methods[n] = {}
                methods[n]['x'] = [xx]
                methods[n]['y'] = [yy]
  
        for method in methods:
            m = methods[method]
            data, xs = align_data(m['x'], m['y'], max_x=max_x)
            error = np.std(data, axis=0)
            mean = np.mean(data, axis=0) 
            if bin_size > 1:
                mean = bin_y(mean, bin_size=bin_size)
            upper = mean + error
            lower = mean - error
            cur_min = np.amin(lower)
            cur_max = np.amax(upper)
       
            y_min = cur_min if y_min is None or cur_min < y_min else y_min
            y_max = cur_max if y_max is None or cur_max > y_max else y_max

            plt.plot(xs, mean, label="{}".format(method))
            if not log:
                plt.fill_between(xs, lower, upper, alpha=0.1)
            else:
                positive = lower > 0
                plt.fill_between(xs, lower, upper, where=positive, alpha=0.1)
    else:
        if bin_size > 1:
            y = bin_y(y, bin_size=bin_size)
        plt.plot(x, y, label="{}".format(name))


    if title.find('NoFrameskip') > -1:
        plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
                   ["1M", "2M", "4M", "6M", "8M", "10M"], fontsize=14)
        plt.xlim(0, 10e6)
    else:
        plt.xticks([1e5, 2e5, 4e5, 6e5, 8e5, 1e6],
                   ["0.1M", "0.2M", "0.4M", "0.6M", "0.8M", "1M"], fontsize=14)
        plt.xlim(0, 1e6)
    plt.yticks(fontsize=14)

    if ylim is not None:
        plt.ylim(ylim)
    elif y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    
    plt.xlabel(x_label, fontsize=18, fontname='arial')
    plt.ylabel(y_label, fontsize=18, fontname='arial')
    if log:
        plt.yscale('log')
        plt.gca().spines['left']._adjust_location()
            

    plt.legend(loc='lower right', prop={'size' : 8})

    plt.title(title, fontsize=18)
    
    plt.show()
    plt.draw()
    if save_loc is not None and len(x) > 1:
        if '.' in title:
            title = title.replace(".", "-", 1)
        plt.savefig(save_loc + title + '_' + y_label + '.png', bbox_inches='tight')

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    if viz is not None:
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)


def load_batched(infile):
    y = []
    x = []
    
    if '\0' in open(infile).read():
        reader = csv.reader(x.replace('\0', '') for x in infile)
    with open(infile, "r") as f:
        reader = csv.reader(f)
        _, _, max_x = next(reader)
        for row in reader:
            y.append(float(row[-1]))
            x.append(int(round(float(row[0]))))

    return x, y, int(round(float(max_x)))


def add_to_plot_dict(monitor_type, monitor_dict, ylim, game_replay_key, x, y, max_x, algo_name):
    if monitor_type in monitor_dict[game_replay_key]:
        monitor_dict[game_replay_key][monitor_type]['x'].append(x)
        monitor_dict[game_replay_key][monitor_type]['y'].append(y)
        monitor_dict[game_replay_key][monitor_type]['max_x'].append(max_x)
        monitor_dict[game_replay_key][monitor_type]['algo'].append(algo_name)
    else:
        monitor_dict[game_replay_key][monitor_type] = {'x': [x], 'y': [y], 'max_x': [max_x],
                                                       'algo': [algo_name],
                                                       'ylim': ylim}


def plot_dict(monitor_dict, viz, save_loc=None):
    vizes = []
    for game_replay_key in monitor_dict:
        for monitor_type in monitor_dict[game_replay_key]:
            xs = monitor_dict[game_replay_key][monitor_type]['x']
            ys = monitor_dict[game_replay_key][monitor_type]['y']
            max_xs = monitor_dict[game_replay_key][monitor_type]['max_x']
            algos = monitor_dict[game_replay_key][monitor_type]['algo']
            ylim = monitor_dict[game_replay_key][monitor_type]['ylim']
            max_x = max(max_xs)
            x_label = "Number of Steps" if "error" in monitor_type else "Number of Steps"
            log = False
            if "MSE" in monitor_type:
                log = True
                if "0.0" in monitor_type:
                    y_label = "Reward_Estimation_Error"
                elif "0.99" in monitor_type:
                    y_label = "Advantage"
                else:
                    continue
            else:
                y_label = "Episode_Reward"
            bin_size = 1 #100 if "error" in monitor_type else 1
            vizes.append(visdom_plot_batched(viz, None, xs, ys, title=game_replay_key, name=algos, max_x=max_x,
                                x_label=x_label, y_label=y_label, bin_size=bin_size, save_loc=save_loc, ylim=ylim,
                                             log=log))
    return vizes


def get_dirs(log_dirs, dir_key):
    dirs = [d for log_dir in log_dirs for d in glob.glob(log_dir + '*/')]
    dirs.sort()
    dirs = reversed(dirs)
    r_dir = re.compile(dir_key)
    dirs = filter(r_dir.match, dirs)
    dirs = [dir for dir in dirs]
    return dirs


def load_other_data(dir, base, monitor_key):
    infiles = glob.glob(os.path.join(dir, base))
    r = re.compile(monitor_key)
    return filter(r.match, infiles)

def reshaped_data(data, name_list, name_order):
    new_data = np.empty((len(name_order), data.shape[-1], int(data.shape[0] / len(name_order))), dtype=data.dtype)
    indices = [0 for _ in range(len(name_order))]
    for xx, name in zip(data, name_list):
        cur_index = name_list.index(name)
        # print(new_data.shape, cur_index, len(indices), indices[cur_index])
        new_data[cur_index, :, indices[cur_index]] = xx
        indices[cur_index] += 1
    return new_data


def log_variance_to_csv(monitor_dict, csv_loc):
    keys = list(monitor_dict.keys())
    all_algos = [monitor_dict[keys[0]][monitor_type]['algo'] for monitor_type in monitor_dict[keys[0]]][0]
    algos = []
    for algo in all_algos:
        if algo not in algos:
            algos.append(algo)
    with open(csv_loc + 'var_results.csv', "wt") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([''] + algos + algos)
    for game_replay_key in sorted(monitor_dict):
        for monitor_type in sorted(monitor_dict[game_replay_key]):
            xs = monitor_dict[game_replay_key][monitor_type]['x']
            ys = monitor_dict[game_replay_key][monitor_type]['y']
            # print(len(xs), len(ys), game_replay_key, monitor_type)
            # data = np.empty((len(ys), 100))
            # for i in range(len(ys)):
            #     data[i] = ys[i][-100:]
            # data, _ = align_data(xs, ys)

            # print(data.shape)
            #
            data = np.array(ys)
            data = reshaped_data(data, monitor_dict[game_replay_key][monitor_type]['algo'], algos)
            data_xs = np.array(xs)
            data_xs = reshaped_data(data_xs, monitor_dict[game_replay_key][monitor_type]['algo'], algos)

            mean = np.mean(data, axis=(-2, -1))
            mean = [str(avg) for avg in mean]

            mean_xs = np.mean(data_xs, axis=(-2, -1))
            mean_xs = [str(avg) for avg in mean_xs]

            with open(csv_loc + 'var_results.csv', "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([game_replay_key] + mean + mean_xs)


def log_avg_to_csv(monitor_dict, csv_loc, last_hundred=False):
    keys = list(monitor_dict.keys())
    all_algos = [monitor_dict[keys[0]][monitor_type]['algo'] for monitor_type in monitor_dict[keys[0]]][0]
    algos = []
    for algo in all_algos:
        if algo not in algos:
            algos.append(algo)
    with open(csv_loc + 'avg_results.csv', "wt") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([''] + algos)
    for game_replay_key in sorted(monitor_dict):
        for monitor_type in sorted(monitor_dict[game_replay_key]):
            xs = monitor_dict[game_replay_key][monitor_type]['x']
            ys = monitor_dict[game_replay_key][monitor_type]['y']
            # print(len(xs), len(ys), game_replay_key, monitor_type)
            if last_hundred:
                data = np.empty((len(ys), 100))
                for i in range(len(ys)):
                    data[i] = ys[i][-100:]
            else:
                min_eps = min([len(y) for y in ys])
                data = np.empty((len(ys), min_eps))
                for i in range(len(ys)):
                    data[i] = ys[i][-min_eps:]
            #
            data = reshaped_data(data, monitor_dict[game_replay_key][monitor_type]['algo'], algos)
            mean = np.mean(data, axis=(-2, -1))
            mean = [str(avg) for avg in mean]
            with open(csv_loc + 'avg_results.csv', "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([game_replay_key] + mean)

# Format for log directory should be should be log_dir + game_name + "_" + algo_name + run + "/"
def visdom_plot_all(viz, log_dir, dir_key='.*', monitor_key='.*', plot_other=False, mode_name=None, mode_list=None,
                    save_loc=None, log_avg=False, log_variance=False, plot=True, last_hundred=False):
    monitor_dict = {}
    num_modes = 1 if mode_list is None else len(mode_list)
    game_ylims = {}
    for i in range(num_modes):
        dir_ext = '' if mode_list is None else mode_list[i]
        for d, dir in enumerate(get_dirs(log_dir, dir_key + dir_ext + '/')):
            local_dir = dir[:-1]
            local_dir = local_dir[local_dir.rfind('/') + 1:]
            game_name = local_dir[:local_dir.find('_')]
            algo_name = local_dir[local_dir.rfind('_') + 1:-3]
            game_replay_key = game_name + '_' + mode_name if mode_list is None else game_name + '_' + mode_name + '_' + mode_list[i]

            if game_replay_key not in monitor_dict:
                monitor_dict[game_replay_key] = {}

            if not plot_other and not log_variance:
                tx, ty = load_data(dir, smooth=1, bin_size=100)

                if tx is not None and ty is not None:
                    if game_name not in game_ylims:
                        game_ylims[game_name] = [int(min(ty)), int(max(ty))]
                    else:
                        game_ylims[game_name][0] = np.min([int(np.min(ty)), game_ylims[game_name][0]])
                        game_ylims[game_name][1] = np.max([int(np.max(ty)), game_ylims[game_name][1]])

                    add_to_plot_dict('Reward', monitor_dict, game_ylims[game_name], game_replay_key, tx, ty, int(1e7), algo_name)

            elif not log_variance and plot_other:
                monitor_types = []
                algo_name = algo_name + local_dir[-3:] if '0.0' in monitor_key else algo_name
                for f in load_other_data(dir, '*_monitor.csv', monitor_key):
                    x, y, max_x = load_batched(f)
                    if x is None or len(x) == 0:
                        continue


                    if game_name not in game_ylims:
                        game_ylims[game_name] = [min(y), max(y)]
                    else:
                        game_ylims[game_name][0] = np.min([np.min(y), game_ylims[game_name][0]])
                        game_ylims[game_name][1] = np.max([np.max(y), game_ylims[game_name][1]])
                    local_file = f[len(dir):]
                    monitor_type = local_file[: local_file.rfind(('_'))]
                    add_to_plot_dict(monitor_type, monitor_dict, game_ylims[game_name], game_replay_key, x, y, int(1e7), algo_name)
                    monitor_types.append(monitor_type)
            else:
                with open(dir + 'true_rewards.pkl', "rb") as output_file:
                    true_rewards = pickle.load(output_file)
                with open(dir + 'noisy_rewards.pkl', "rb") as output_file:
                    noisy_rewards = pickle.load(output_file)
                with open(dir + 'estimate_rewards.pkl', "rb") as output_file:
                    estimate_rewards = pickle.load(output_file)
                true_rewards_var = np.var([sum(tr) for tr in true_rewards])
                noisy_rewards_var = np.var([sum(tr) for tr in noisy_rewards])
                estimate_rewards_var = np.var([sum(tr) for tr in estimate_rewards])

                true_rewards_mse = 0
                noisy_rewards_mse = sum([np.mean((np.array(tr) - np.array(nr)) ** 2) for tr, nr in zip(true_rewards, noisy_rewards)]) / len(true_rewards)
                estimate_rewards_mse = sum([np.mean((np.array(tr) - np.array(er)) ** 2) for tr, er in zip(true_rewards, estimate_rewards)]) / len(true_rewards)

                add_to_plot_dict('Variance', monitor_dict, None, game_replay_key, true_rewards_mse, true_rewards_var, int(1e7), algo_name +'true')
                add_to_plot_dict('Variance', monitor_dict, None, game_replay_key, noisy_rewards_mse, noisy_rewards_var, int(1e7),
                                 algo_name + 'noisy')
                add_to_plot_dict('Variance', monitor_dict, None, game_replay_key, estimate_rewards_mse, estimate_rewards_var, int(1e7),
                                 algo_name + 'estimate')


    if log_variance:
        log_variance_to_csv(monitor_dict, save_loc)

    if log_avg:
        log_avg_to_csv(monitor_dict, csv_loc=save_loc, last_hundred=last_hundred)

    if plot:
        return plot_dict(monitor_dict, viz, save_loc=save_loc)


def parse_args():
    parser = argparse.ArgumentParser("visdom parser")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--log-dir", type=str, nargs='*', default=['./logs/'],
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--vis-interval", type=int, default=None,
                        help="num seconds between vis plotting, default just one plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    viz = Visdom()
    if not viz.check_connection():
        subprocess.Popen(["python", "-m", "visdom.server", "-p", str(args.port)])

    if args.vis_interval is not None:
        vizes = []
        while True:
            for v in vizes:
                viz.close(v)
            try:
                vizes = visdom_plot_all(viz, args.log_dir, mode_name='Noise', mode_list=['0.0', '0.1', '0.2', '0.3', '0.4'],
                                        save_loc=None, plot_other=False, log_avg=False, plot=True, last_hundred=False)
            except IOError:
                pass
            time.sleep(args.vis_interval)
    else:
        visdom_plot_all(viz, args.log_dir, mode_name='Noise', mode_list=['0.0', '0.1', '0.2', '0.3', '0.4'],
                        save_loc=None, plot_other=False, log_avg=False, plot=True, last_hundred=False)

