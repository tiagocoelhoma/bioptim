import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import load_pickle, save_pickle

intervals = {
    'rect_fem_r':          # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # RF RIGHT LEG
            # f'MEC_90_050': [[350, 550], [1290, 1490]], 	        #	50
            f'MEC_90_100': [[1820, 2080], [2340, 2560]], 	    #	100
            f'MEC_90_150': [[2800, 3064], [3310, 3520]], 	    #	150
            f'MEC_90_200': [[3810, 4050], [4290, 4510]], 	    #	200
            f'MEC_90_250': [[4720, 4940], [5170, 5390]], 	    #	250
            f'MEC_90_300': [[5630, 5800], [6030, 6260]], 	    #	300
            f'MEC_90_350': [[6510, 6750], [6950, 7190]], 	    #	350
            f'MEC_90_400': [[7413, 7635], [7800, 8060]], 	    #	400
            f'MEC_90_450': [[8280, 8520], [8735, 8935]], 	    #	450
            f'MEC_90_500': [[10620, 10780], [11000, 11190]],	#	500
            f'MEC_90_550': [[11500, 11660], [11950, 12150]],	#	550
            f'MEC_90_600': [[12470, 12610], [12900, 13100]],	#	600
            f'MEC_90_650': [[13430, 13620], [13900, 14080]],	#	650
            f'MEC_90_700': [[14390, 14550], [14860, 15020]],	#	700
            f'MEC_90_750': [[15320, 15465], [15745, 15925]],	#	750
            f'MEC_90_800': [[16224, 16375], [16680, 16820]],	#	800
        },
    'vasmed_r':  # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # HAM RIGHT LEG
            # f'MEC_90_050': [[350, 550], [350, 550]], 	        #	50  -> nao pegou
            f'MEC_90_100': [[1230, 1430], [2815, 3015]], 	        #	100
            f'MEC_90_150': [[3480, 3660], [3920, 4090]], 	        #	150
            f'MEC_90_200': [[4388, 4548], [4827, 4996]], 	    #	200
            f'MEC_90_250': [[5300, 5450], [5745, 5895]], 	    #	250
            f'MEC_90_300': [[6215, 6375], [6675, 6845]], 	    #	300
            f'MEC_90_350': [[7145, 7285], [7590, 7740]], 	    #	350
            f'MEC_90_400': [[8045, 8225], [8490, 8650]], 	    #	400
            f'MEC_90_450': [[8940, 9110], [9350, 9525]], 	    #	450
            f'MEC_90_500': [[9820, 9975], [10235, 10400]],	    #	500
            f'MEC_90_550': [[10650, 10820], [11080, 11250]],	    #	550
            f'MEC_90_600': [[11525, 11695], [11940, 12120]],	#	600
            f'MEC_90_650': [[12430, 12590], [12870, 13000]],	#	650
            f'MEC_90_700': [[13290, 13490], [13700, 13890]],	#	700
            f'MEC_90_750': [[14160, 14340], [14580, 14745]],	#	750
            f'MEC_90_800': [[15060, 15240], [15485, 15666]],	#	800
        },
    'bflh140_r':  # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # HAM RIGHT LEG
            # f'MEC_90_050': [[350, 550], [1290, 1490]],  # 50  -> nao pegou
            f'MEC_90_100': [[350, 550], [1290, 1490]],  # 100  -> nao pegou
            f'MEC_90_150': [[350, 550], [1290, 1490]],  # 150  -> nao pegou
            f'MEC_90_200': [[3390, 3530], [3780, 3950]],  # 200
            f'MEC_90_250': [[4210, 4390], [4650, 4790]],  # 250
            f'MEC_90_300': [[5030, 5215], [5475, 5660]],  # 300
            f'MEC_90_350': [[5845, 6075], [6280, 6485]],  # 350
            f'MEC_90_400': [[6740, 6896], [7150, 7360]],  # 400
            f'MEC_90_450': [[7500, 7800], [7980, 8190]],  # 450
            f'MEC_90_500': [[8400, 8630], [8830, 9070]],  # 500
            f'MEC_90_550': [[9250, 9480], [9700, 9930]],  # 550
            f'MEC_90_600': [[10110, 10300], [10520, 10750]],  # 600
            f'MEC_90_650': [[10960, 11150], [11360, 11560]],  # 650
            f'MEC_90_700': [[11750, 11970], [12195, 12370]],  # 700
            f'MEC_90_750': [[12540, 12760], [12950, 13200]],  # 750
            f'MEC_90_800': [[13400, 13700], [13790, 14040]],  # 800
        },
    'rect_fem_l':  # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # RF LEFT LEG
            # 'MEC_90_050': [[150, 450], [150, 450]],
            f'MEC_90_100': [[982, 1082], [1316, 1480]],
            f'MEC_90_120': [[1739, 1879], [2097, 2212]],
            f'MEC_90_200': [[2556, 2730], [2942, 3100]],
            f'MEC_90_250': [[3296, 3435], [3659, 3809]],
            f'MEC_90_300': [[3960, 4155], [4301, 4485]],
            f'MEC_90_350': [[4715, 4861], [5063, 5223]],
            f'MEC_90_400': [[5488, 5672], [5972, 6184]],
            f'MEC_90_450': [[6427, 6613], [6823, 6989]],
            f'MEC_90_500': [[7210, 7378], [7584, 7783]],
            f'MEC_90_550': [[7957, 8174], [8484, 8672]],
            f'MEC_90_600': [[8995, 9170], [9493, 9674]],
            f'MEC_90_650': [[9980, 10177], [10511, 10703]],
            f'MEC_90_700': [[10993, 11200], [11525, 11695]],
            f'MEC_90_750': [[12025, 12205], [12470, 12658]],
            f'MEC_90_800': [[12974, 13202], [13500, 13687]],
        },
    'vasmed_l': {
        # 'MEC_90_050': [[150, 450], [150, 450]],
        'MEC_90_100': [[475, 685], [815, 1030]],
        'MEC_90_150': [[1325, 1535], [1825, 2036]],
        'MEC_90_200': [[2375, 2625], [2870, 3050]],
        'MEC_90_250': [[3335, 3545], [3785, 4020]],
        'MEC_90_300': [[4300, 4560], [4790, 5015]],
        'MEC_90_350': [[5295, 5525], [5755, 5985]],
        'MEC_90_400': [[6335, 6546], [6820, 7015]],
        'MEC_90_450': [[7330, 7515], [7790, 7985]],
        'MEC_90_500': [[8255, 8460], [8720, 8945]],
        'MEC_90_550': [[9335, 9575], [9735, 9945]],
        'MEC_90_600': [[10260, 10475], [10789, 11011]],
        'MEC_90_650': [[11345, 11595], [11791, 12022]],
        'MEC_90_700': [[12330, 12530], [12805, 13045]],
        'MEC_90_750': [[13305, 13545], [13865, 14155]],
        'MEC_90_800': [[14397, 14582], [14779, 15400]],
    },
    'bflh140_l':  # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # HAM LEFT LEG
            # 'MEC_90_050': [[956, 1150], [1250, 1430]],     # 50  -> nao pegou
            'MEC_90_100': [[956, 1150], [1250, 1430]],     # 100  -> nao pegou
            'MEC_90_150': [[956, 1150], [1250, 1430]],     # 150  -> nao pegou
            f'MEC_90_200': [[956, 1150], [1250, 1430]],  #
            f'MEC_90_250': [[1545, 1743], [2047, 2270]],
            f'MEC_90_300': [[2555, 2800], [3045, 3265]],
            f'MEC_90_350': [[3580, 3760], [4020, 4220]],
            f'MEC_90_400': [[4550, 4750], [5050, 5250]],
            f'MEC_90_450': [[5540, 5800], [6037, 6300]],
            f'MEC_90_500': [[6540, 6840], [7040, 7250]],
            f'MEC_90_550': [[7530, 7750], [7975, 8205]],
            f'MEC_90_600': [[8540, 8790], [9000, 9250]],
            f'MEC_90_650': [[9520, 9750], [10138, 10300]],
            f'MEC_90_700': [[10620, 10890], [11050, 11280]],
            f'MEC_90_750': [[11440, 11700], [11800, 12000]],
            f'MEC_90_800': [[12230, 12480], [12773, 12948]],
        },
}

def scaled_sigmoid_1(pw, a, pd0, pdt):
    return a * (1 - np.exp(-(pw - pd0) / pdt))

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    mode = 'same'
    return np.convolve(data, window, mode=mode)


def fitted_parameters(torque_data, mus, window_size, offset_interval):
    pulse_width, max_torque, muscle_labels = [], [], []

    filtered_torque = moving_average(torque_data, window_size)
    ft = np.zeros_like(filtered_torque)
    for i, interval_name in enumerate(intervals[mus]):
        for j in range(len(intervals[mus][interval_name])):
            idx1 = intervals[mus][interval_name][j][0]
            idx2 = intervals[mus][interval_name][j][1]
            ft[idx1:idx2] = filtered_torque[idx1:idx2] - np.mean(filtered_torque[idx1:idx1 + offset_interval])
            try:
                # print(f'Pulse width {interval_name[7:]} us; Max torque {round(max(ft[idx1:idx2]), 4)} Nm')
                pulse_width.append(interval_name[7:])
                max_torque.append(round(max(ft[idx1:idx2]), 4))

                mus_name = 'RF_R'
                mus_name = 'RF_L' if mus == 'rect_fem_l' else mus_name
                mus_name = 'VM_R' if mus == 'vasmed_r' else mus_name
                mus_name = 'VM_L' if mus == 'vasmed_l' else mus_name
                mus_name = 'HAM_R' if mus == 'bflh140_r' else mus_name
                mus_name = 'HAM_L' if mus == 'bflh140_l' else mus_name

                muscle_labels.append(mus_name)
            except:
                break

    # retrieving data from 50us to 800us pulse/phase width (no difference was observed for higher values)
    pw = np.array([100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400,
                   450, 450, 500, 500, 550, 550, 600, 600, 650, 650, 700, 700, 750, 750, 800, 800]) * 2e-6
    pw_values_filtered = np.linspace(min(pw), max(pw), 1000)
    mean_max_torque = []

    for i in range(len(pw)):
        mean_max_torque.append(round(max_torque[i], 4))

    return mean_max_torque, pw, muscle_labels


markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']  # as many as needed


def main(save_data=False, print_flag=True):
    data_file = {
        'rect_fem_r':   "Param_iden_cycling_Brazil/data_exp/files/20250627_112633_1st_iso_meas_RF.txt",
        'vasmed_r':     "Param_iden_cycling_Brazil/data_exp/files/20250627_115217_1st_meas_VM.txt",
        'bflh140_r':    "Param_iden_cycling_Brazil/data_exp/files/20250627_114039_1st_iso_meas_HAM.txt",
        'rect_fem_l':   "Param_iden_cycling_Brazil/data_exp/files/20250627_144915_1st_iso_meas_RF_L.txt",
        'vasmed_l':     "Param_iden_cycling_Brazil/data_exp/files/20250627_150638_1st_iso_meas_VM_L.txt",
        'bflh140_l':    "Param_iden_cycling_Brazil/data_exp/files/20250627_145728_1sr_iso_meas_HAM_L.txt",
    }

    plt.figure(figsize=(12, 7))
    muscles = sorted(set(data_file.keys()))
    colors = plt.cm.tab10(range(len(muscles)))
    style_map = {mus: (colors[i % len(colors)], markers[i % len(markers)]) for i, mus in enumerate(muscles)}

    for idx, mus in enumerate(data_file):
        torques, pw, muscle_labels = [], [], []
        # a_, pd0_, pdt_ = 0, 0, 0
        # print(mus)
        df = pd.read_csv(data_file[mus], delimiter='\t')
        torque_data = df['Torque (Nm)'].values.flatten()
        # obtaining the stimulation intervals automatically and then the fitted parameters
        t, p, ml = fitted_parameters(torque_data=torque_data, mus=mus, window_size=10, offset_interval=50)
        torques.extend(t)
        pw.extend(p)
        muscle_labels.extend(ml)

        params, covariance = curve_fit(f=scaled_sigmoid_1,
                                       xdata=pw, ydata=torques,
                                       p0=[1, 86.906e-6, 138.441e-6],
                                       bounds=([0, 0, 0], [100, 1e-3, 5e-3])  # Prevents unrealistic negative values
                                       )

        a, pd0, pdt = float(params[0]), float(params[1]), float(params[2])
        print(f'Parameters a:{a}; pd0:{pd0}; pdt:{pdt}')

        if save_data:
            filename = 'Param_iden_cycling_Brazil/opt_sol_results/pickles/pd0pdt_v2.pkl'
            if os.path.exists(filename):
                data = load_pickle(filename)
                # print(f"Previous data for muscle {mus} {data[mus]}")
            else:
                data = {}
                print(f"Creating file...")

            data[mus] = [a, pd0, pdt]
            save_pickle(filename, data)
            data = load_pickle(filename)
            print(f"New data for muscle {mus} {data[mus]}")

        if print_flag:
            pw_values_filtered = np.linspace(min(pw), max(pw), 1000)


            fitted_torque_filtered = scaled_sigmoid_1(pw_values_filtered, *params)
            print("Fitted Parameters 0:", params)
            plt.plot(pw_values_filtered, fitted_torque_filtered,
                     # label=f"{muscle_labels[0]}: $a$={a:.2f}, $pd_0$={pd0 * 1e6:.1f} µs, $pd_t$={pdt * 1e6:.1f} µs",
                     color=colors[idx], linewidth=1, linestyle='dashed')

            plt.scatter(pw, torques, color=colors[idx], marker=style_map[mus][-1],
                        label=f"{muscle_labels[0]}: $a$={a:.2f}, $pd_0$={pd0 * 1e6:.1f} µs, $pd_t$={pdt * 1e6:.1f} µs",
                        alpha=1.0, edgecolors='black')

            plt.title("Max Torque vs Pulse Duration")
            plt.xlabel("Pulse duration (s)", fontsize="12")
            plt.ylabel("Max Torque (Nm)", fontsize="12")
            plt.legend(loc="upper left", fontsize="12", frameon=False)
            plt.savefig('Param_iden_cycling_Brazil/opt_sol_results/images/pd0pdt_v2.pdf', format='pdf', dpi=2200, bbox_inches='tight')
            plt.grid(False)

            plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main(save_data=True, print_flag=True)
