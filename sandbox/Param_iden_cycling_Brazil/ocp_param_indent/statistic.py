"""

"""
import numpy as np
import matplotlib.pyplot as plt
from Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import update_parameters_value, load_pickle


def statistic_plots(muscle):
    data = load_pickle('opt_sol_results/pickles/prediction.pkl')
    print(data['rect_fem_r'])
    tau_target = data['rect_fem_r']['target']
    knee_ang = data['rect_fem_r']['joint_angle']
    predicted_torques = data['rect_fem_r']['predicted_torques']

    # plt.plot(np.concatenate(tau_target))
    # plt.plot(np.concatenate(predicted_torques))
    # plt.show()

    from matplotlib import cm
    from mpl_toolkits.mplot3d.axes3d import get_test_data
    plt.figure(figsize=(10, 5))

    for i, torque in enumerate(tau_target):
        center = knee_ang[i]
        n = len(torque)
        x = np.arange(n) - n // 2 + center  # center the curve at the knee angle
        plt.plot(x, torque, label=f'knee = {center}°')
        plt.plot(x, predicted_torques[i], label=f'knee = {center}°', linestyle='dotted')

    plt.xlabel("Knee angle (°)")
    plt.ylabel("Torque (Nm)")
    plt.title("Torque curves centered at their respective knee angles")
    plt.legend()
    plt.grid(True)
    plt.xticks(knee_ang)  # show only the knee angles if you want it cleaner
    plt.tight_layout()
    plt.show()

    # 3d plot
    tau_exp, tau_pre = [], []
    for i, torque in enumerate(tau_target):
        tau_exp.append(torque)
        tau_pre.append(np.concatenate(predicted_torques[i]))
    tau_target_all_muscles = [
        tau_exp,
        tau_exp,
    ]
    predicted_torques_all_muscles = [
        tau_pre,
        tau_pre,
    ]

    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    for muscle_idx, (tau_target_muscle, predicted_muscle) in enumerate(
            zip(tau_target_all_muscles, predicted_torques_all_muscles)):
        for i, (target, predicted) in enumerate(zip(tau_target_muscle, predicted_muscle)):
            center = knee_ang[i]
            n = len(target)
            x = np.arange(n) - n // 2 + center  # center the curve at the knee angle
            y = np.full_like(x, muscle_idx)  # constant Y value for muscle layer
            ax.plot(x, y, target, label=None if muscle_idx > 0 else "Target", color="blue")
            # ax.plot(x, y, predicted, label=None if muscle_idx > 0 else "Target", color="red")
            ax.plot(x, y, predicted, linestyle='dotted', label=None if muscle_idx > 0 else "Predicted", color="red")

    ax.set_xlabel("Knee angle (°)")
    ax.set_ylabel("Muscle index")  # optionally replace with muscle name ticks
    ax.set_zlabel("Torque (Nm)")
    ax.set_title("3D Torque curves across muscles and knee angles")

    # Optional: show legend once
    ax.legend(["Target", "Predicted"])
    plt.tight_layout()
    plt.show()