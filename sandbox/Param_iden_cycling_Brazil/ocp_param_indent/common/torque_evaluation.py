import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


# msr = {     # 27 06 2025
#     'rect_fem_r':   "files/20250627_121053_2nd_iso_meas_RF.txt",    # Rectus femoris    (measurements 20250627)
#     'vasmed_r':     "files/20250627_122827_2nd_iso_meas_VM.txt",    # vastus medialis   (measurements 20250627)
#     'bflh140_r':    "files/20250627_122301_2nd_iso_meas_HAM.txt",    # hamstrings   (measurements 20250627)
#     'rect_fem_l': "files/20250627_151618_2nd_iso_meas_RF_L.txt",  # Rectus femoris    (measurements 20250627)
#     'vasmed_l': "files/20250627_153151_2nd_iso_meas_VM_L.txt",  # vastus medialis   (measurements 20250627)
#     'bflh140_l': "files/20250627_154107_2nd_iso_meas_HAM_L.txt",  # hamstrings   (measurements 20250627)
# }
# with these data (27 05 2025) it was not possible to converge the pedalling simulation. So I had
# to update the biomechanical model and first validate the covergence with A constant. Also, I validated
# the generated stimulaiton pattern with the participant. He was able to pedal. With these configurations
# i measured again the isometric with right leg (why only the right leg? because it was late already, the
# participant was fatigued (tired) and also it would be better to first try the simulation with simetric approach.
# so, the measurements done 02 07 2025 was taken to run simulation again. Hope it works!! Lets go!!

msr = {     # 02 07 2025    (only right side)
    # ISOMETRIC
    'rect_fem_r':   "files/20250702_175521_2nd_iso_meas_RF.txt",    # Rectus femoris    (measurements 20250702)
    'vasmed_r':     "files/20250702_181337_2nd_iso_meas_VM.txt",    # vastus medialis   (measurements 20250702)
    'bflh140_r':    "files/20250702_180748_2nd_iso_meas_HAM.txt",    # hamstrings       (measurements 20250702)
    'rect_fem_l':   "files/20250709_142106_2nd_iso_meas_RF_L.txt",  # Rectus femoris    (measurements 20250709)
    'vasmed_l':     "files/20250709_144116_2nd_iso_meas_VM_L.txt",  # vastus medialis   (measurements 20250709)
    'bflh140_l':    "files/20250709_143217_2nd_iso_meas_HAM_L.txt",  # hamstrings       (measurements 20250709)
    # ISOVELOCITY
    'rect_fem_r_vel': "files/20250704_100027_isovel_right_RF.txt",  # Rectus femoris    (measurements 202507x)
}

intervals = {
    'rect_fem_r':
        {
            # f'LEN_01':  [[1310, 1420], [1560, 1700]],
            f'LEN_02':  [[2190, 2320], [2600, 2720]],   # ws
            # f'LEN_03':  [[3060, 3200], [3360, 3510]],
            f'LEN_04':  [[3780, 3925], [3980, 4115]],   # ws
            # f'LEN_05':  [[4400, 4545], [4710, 4890]],
            f'LEN_06':  [[5280, 5390], [5480, 5615]],   # ws    (highest)
            # f'LEN_07':  [[5880, 5960], [6125, 6200]],
            f'LEN_08':  [[6600, 6740], [6900, 7010]],   # ws
            # f'LEN_09':  [[7250, 7385], [7475, 7595]],
            f'LEN_10':  [[7790, 7940], [8060, 8210]],   # ws
            # f'LEN_11':  [[8400, 8530], [8600, 8780]],
        },
    'vasmed_r':  # static knee ROM vastus medialis (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # f'LEN_01': [[1030 , 1200 ], [1315 , 1490]],
            # f'LEN_02': [[1710 , 1870 ], [2070 , 2230]],   # ws
            f'LEN_03': [[2468 , 2640 ], [2750 , 2900]],
            f'LEN_04': [[3135 , 3275 ], [3370 , 3510]],   # ws
            # f'LEN_05': [[3720 , 3875 ], [3920 , 4075]],
            f'LEN_06': [[4300 , 4475 ], [4520 , 4665]],   # ws
            # f'LEN_07': [[7150 , 7315 ], [7440 , 7600]],
            f'LEN_08': [[7840 , 7980 ], [8030 , 8160]],   # ws
            # f'LEN_09': [[8400 , 8570 ], [8630 , 8770]],
            f'LEN_10': [[9030 , 9170 ], [9180 , 9315]],   # ws
            # f'LEN_11': [[9490 , 9690 ], [9740 , 9890]],
            # f'LEN_12': [[10100, 10250], [10320, 10480]],
        },
    'bflh140_r':  # static knee ROM Hamstrings (file = 0157_Datafile.txt)
        {
            # f'LEN_01': [[2350, 2470], [2540, 2680]],
            # f'LEN_02': [[2865, 3030], [3135, 3270]], # REMOVER DO NOT USE (results in a = 0)
            # f'LEN_03': [[3495, 3640], [3715, 3850]],   # wsREMOVER DO NOT USE
            # f'LEN_04': [[4000, 4170], [4220, 4330]], # REMOVER DO NOT USE
            # f'LEN_05': [[4985, 5120], [5340, 5480]],   # ws -> not converging
            f'LEN_06': [[5930, 6060], [6360, 6490]],
            f'LEN_07': [[6804, 6960], [7130, 7300]],   # ws
            # f'LEN_08': [[7545, 7710], [7835, 8010]],
            f'LEN_09': [[8280, 8445], [8470, 8610]],   # ws
            f'LEN_10': [[8850, 9050], [9180, 9340]],
            f'LEN_11': [[9500, 9750], [9920, 10100]],   # ws
        },
    'rect_fem_l':  # static knee ROM rectus femoris  (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            # f'LEN_01': [[1020, 1125], [1215, 1335]],
            # f'LEN_02': [[1585, 1733], [1790, 1915]],
            f'LEN_03': [[2200, 2325], [2345, 2475]],   # ws
            # f'LEN_04': [[2885, 2990], [3080, 3185]],
            f'LEN_05': [[3540, 3650], [3760, 3880]],   # ws
            # f'LEN_06': [[4255, 4380], [4490, 4595]], # , [5210, 5330]
            f'LEN_07': [[4960, 5105], [5210, 5330]],    # ws# , [5210, 5330]
            # f'LEN_08': [[9145, 9260], [9340, 9475]],
            f'LEN_09': [[9830, 10060], [10615, 10875]],   # ws
            # f'LEN_10': [[11065, 11195], [11225, 11341]],
            f'LEN_11': [[12131, 12270], [12400, 12546]],   # ws
            # f'LEN_12': [[12960, 13120], [13215, 13345]],
            # f'LEN_13': [[13825, 13970], [14090, 14215]],
        },
    'vasmed_l':  # static knee ROM vastus medialis (file = 20250627_121053_2nd_iso_meas_RF.txt)
        {
            f'LEN_03': [[2920, 3060], [3150, 3320]],# ws
            # f'LEN_04': [[3730, 3880], [3930, 4080]],
            f'LEN_05': [[4540, 4695], [4935, 5094]],# ws
            # f'LEN_06': [[5410, 5570], [5670, 5800]],
            f'LEN_07': [[6190, 6335], [8500, 8675]],# ws
            # f'LEN_08': [[9005, 9140], [9230, 9380]],
            f'LEN_09': [[9765, 9920], [9945, 10080]],# ws
            # f'LEN_10': [[10410, 10540], [10630, 10780]],
            f'LEN_11': [[11880, 12100], [12165, 12335]],# ws
            # f'LEN_12': [[12685, 12830], [12885, 13050]],
            # f'LEN_13': [[13400, 13565], [13650, 13830]],
        },
    'bflh140_l':  # static knee ROM Hamstrings (file = 0157_Datafile.txt)
        {
            # f'LEN_01': [[7050, 7160], [7215 , 7315]],
            # f'LEN_02': [[8160, 8310], [8400 , 8500]],
            f'LEN_03': [[8715, 8870], [8920 , 9040]],# ws
            # f'LEN_04': [[9490, 9610], [9700 , 9805]],
            f'LEN_05': [[10130, 10250], [10370, 10520]],# ws
            # f'LEN_06': [[11340, 11490], [11540, 11670]],
            f'LEN_07': [[11920, 12080], [12120, 12250]],# ws
            # f'LEN_08': [[12490, 12625], [12650, 12780]],
            f'LEN_09': [[13025, 13175], [13240, 13387]],# ws
            # f'LEN_10': [[13600, 13760], [13780, 13915]],
            f'LEN_11': [[14185, 14330], [14350, 14470]],# ws
            # f'LEN_12': [[14876, 15042], [15060, 15202]],
            # f'LEN_13': [[15430, 15590], [15600, 15730]],
        },
    'rect_fem_r_vel':  # static knee ROM Hamstrings (file = 0157_Datafile.txt)
        {
            f'passive_1': [0, 850],
            f'active_1': [950, 1150],
            f'passive_2': [2100, 2300],
            f'active': [896, 1246],
            f'identif': [608, 678]
        },
}


def print_torque_data(data_1, label_1, data_2, label_2):

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('samples')
    ax1.set_ylabel(label_1, color=color)
    ax1.plot(data_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel(label_2, color=color)  # we already handled the x-label with ax1
    ax2.plot(data_2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# Moving average filter function
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size     # Create a normalized window of the specified size
    mode = 'same'                                   # for the same length as the original data
    return np.convolve(data, window, mode=mode)     # Apply the convolution of the data with the window


def remove_torque_offset(torque_data, measurement, window_size, offset_interval, print_flag=True):

    pulse_width, max_torque = [], []
    filtered_torque = moving_average(torque_data, window_size)
    ft = np.zeros_like(filtered_torque)

    for i, interval_name in enumerate(intervals[measurement]):
        for j in range(len(intervals[measurement][interval_name])):
            idx1 = intervals[measurement][interval_name][j][0]
            idx2 = intervals[measurement][interval_name][j][1]
            ft[idx1:idx2] = filtered_torque[idx1:idx2] - np.mean(filtered_torque[idx1:idx1+offset_interval])
            max_torque.append(round(max(ft[idx1:idx2]),4))

    if print_flag:
        print_torque_data(data_1=torque_data, label_1='original torque', data_2=ft, label_2='filtered torque')
    return ft


def data_for_ocp(torque_data, crank_angles, msr, offset_interval, stim_freq, window_size, print_flag, contraction=1, fxd_range=False):

    freq_sampling = 50     # according to data collected from rotor 2INPOWER (FAST DATA 20 ms)
    tau_offset = 0.40
    if print_flag: print(f'Old intervals: {intervals[msr]}')
    interval_names = list(intervals[msr])
    flag_activated = False
    interval_cnt, aux = 0, 0
    ft_mp, angle, stim_dt, time = [], [], [], []

    # adjusting sampled torque data to the stimulation frequency
    T = len(torque_data) / freq_sampling
    t_data = np.linspace(0, T, len(torque_data))
    t_ocp = np.linspace(0, T, int(T * stim_freq))
    interp_func = interp1d(t_data, torque_data, kind='linear')
    torque_data = interp_func(t_ocp)

    t_data = np.linspace(0, T, len(crank_angles))
    t_ocp = np.linspace(0, T, int(T * stim_freq))
    interp_func = interp1d(t_data, crank_angles, kind='linear')
    angle_data = interp_func(t_ocp)

    torque_data = remove_torque_offset(torque_data, msr, window_size, offset_interval, print_flag)
    fxd_range = False

    stim_flag = np.zeros_like(torque_data)
    torque_data[torque_data < 0] = 0    # removing negative values
    # identify the contractions and create stimulation data (stimulation = {0; 1})
    if not fxd_range:
        for idx, value in enumerate(torque_data):
            if not flag_activated and abs(value) > tau_offset:
                intervals[msr][interval_names[interval_cnt]][aux % 2][0] = idx - offset_interval
                flag_activated = True
            elif flag_activated and abs(value) < tau_offset:
                intervals[msr][interval_names[interval_cnt]][aux % 2][1] = idx + offset_interval
                flag_activated = False
                aux += 1
            if aux == 2:
                if contraction == 1:  # in the protocol, there are two contractions for each angle
                    idx1 = intervals[msr][interval_names[interval_cnt]][0][0]
                    idx2 = intervals[msr][interval_names[interval_cnt]][0][1]
                else:
                    idx1 = intervals[msr][interval_names[interval_cnt]][1][0]
                    idx2 = intervals[msr][interval_names[interval_cnt]][1][1]

                ft_mp.append(torque_data[idx1:idx2])
                idx1_stim = idx1 + offset_interval
                idx2_stim = idx2 - offset_interval - 12
                stim_flag[idx1_stim:idx2_stim] = 1    # todo: verify this value offset because pulse happens first
                stim_dt.append(stim_flag[idx1:idx2])
                angle.append(int(np.mean(angle_data[idx1_stim:idx2_stim])))
                time.append(round(len(stim_flag[idx1:idx2]) * (1/stim_freq), 4))
                interval_cnt +=1
                aux = 0
    else:
        for i, name in enumerate(interval_names):
            if contraction == 1:
                idx1 = intervals[msr][name][0][0]
                idx2 = intervals[msr][name][0][1]
            else:
                idx1 = intervals[msr][name][1][0]
                idx2 = intervals[msr][name][1][1]
            ft_mp.append(torque_data[idx1:idx2])
            idx1_stim = idx1 + offset_interval
            idx2_stim = idx2 - offset_interval - 12
            stim_flag[idx1_stim:idx2_stim] = 1    # todo: verify this value offset because pulse happens first
            stim_dt.append(stim_flag[idx1:idx2])
            angle.append(int(np.mean((angle_data[idx1:idx2]))))
            time.append(round(len(stim_flag[idx1:idx2]) * (1 / stim_freq), 4))
    if print_flag:
        print(f'New intervals: {intervals[msr]}')
        print_torque_data(data_1=np.concatenate(ft_mp), label_1='torque_interpolated', data_2=np.concatenate(stim_dt), label_2='stim')

    # creating a vector with torque data to be evaluated in the mult-phase problem for identification of params.
    tau, stim, nodes = [], [], []
    for i in range(len(ft_mp)):
        tau.append(ft_mp[i])
        stim.append(stim_dt[i])
        nodes.append(len(ft_mp[i]))
    print(time)
    time = [i for i in time]
    print(f'durations...? {time}')

    if print_flag:
        print_torque_data(data_1=np.concatenate(tau), label_1='torque',
                          data_2=np.concatenate(stim), label_2='stim')

    return [tau, stim, angle, nodes, time]


def main():
    # parameters to be defined by the user
    # define the muscle below {STATIC_2_VM, STATIC_2_RF, STATIC_2_HM}
    muscle = 'STATIC_2_RF'
    window_size = 10            # filtering torque data
    offset_interval = 50        # data to be taken into account prior to the contraction

    # reading data file
    data_file = msr[muscle]
    df = pd.read_csv(data_file, delimiter='\t')
    torque = df['Powermeter_Torque_right[Nm] '].values.flatten()

    # obtaining the stimulation intervals automatically
    torque_dt = remove_torque_offset(torque, muscle, window_size, offset_interval)
    return data_for_ocp(torque_dt, muscle, 50, True, True)


if __name__ == "__main__":
    main()