import bioviz
import numpy as np
import biorbd


crank_opt_flag = False
if crank_opt_flag:
    # for optimized crank
    model_path = "leg_model_cyclic_muscle_inversed_root_opt_only_seg.bioMod"
    q_init = np.array([[-1.1198], [-2.1185], [1.9047], [0.9239], [0.845], [1.4614], [1.6171]])
    # q_init = np.array([[-1.1197], [-2.0684], [1.7286], [1.4364], [0.9601], [1.2029], [2.0671]])
    q_init = np.array([[-1.5495], [4.4963], [1.7551], [0.1985], [1.0628], [1.6257], [0.5956]])
    q_init = np.array([[-6.3003], [4.9974], [1.9865], [0.1753], [5.5781], [1.0031], [0.5679]])
    q_init = np.array([[-6.3003], [4.956], [2.0821], [0.1555], [5.4951], [1.1773], [0.6117]])
    q_init = np.array([[-6.3003], [4.9974], [2.0722], [0.1753], [5.5366], [1.1429], [0.5679]])

    # -1.5495, 4.5664, 1.6352, 0.1985, 1.1329, 1.5297, 0.5722


else:
    # for original crank
    # model_path = "legcyc_rectfem_r.bioMod"
    # model_path = "leg_model_cyclic_muscle_inversed_root_3.bioMod"
    model_path = "legcyc_hams_r.bioMod"
    q_init = np.array([[-2.09], [1.3758], [1.5722], [0.1631], [0.8935], [1.9052], [0.2907]])
    q_init = np.array([[-6.349995], [5.297289], [2.034617], [-0.058011], [5.637115], [1.234664], [0.468416]])
    q_init = np.array([[-6.2831],[5.3187],[1.8995],[-0.0717],[5.7158],[1.0079],[0.5319]])
    q_init = np.array([[-6.2831],[5.1769],[1.9108],[0.5177],[5.6024],[1.0474],[1.085]])
    q_init = np.array([[-6.2831],[5.0917],[1.9165],[0.2624],[5.659],[0.895],[0.7304]])
    q_init = np.array([[-7.009174],[6.016367],[1.828031],[0.152276],[6.211081],[1.017038],[0.782356]])
    # q_init = np.array([[-6.2831],[5.1485],[1.8657],[0.1206],[5.7441],[.7709],[0.6453]])
    # q_init = np.array([[-6.2831],[5.0917],[1.9391],[0.2198],[5.659],[0.9176],[0.6453]])
    # q_init = np.array([[-6.5451867], [5.4176772], [1.9334942], [0.1615153], [5.8753062], [0.9257969], [0.6828837]])
    # q_init = np.array([[-7.20947899], [6.27991849], [1.78557518], [0.11148532], [6.34229359], [1.12125819], [0.68472721]])
    # q_init = np.array([[-6.2831], [5.0917], [1.9391], [0.2198], [5.659], [0.9176], [0.6453]]) # before changes
    q_init = np.array([[-6.2831], [5.0389], [1.9686], [0.0104], [5.5781], [1.0363], [0.7776]])  # adjusted with regard to Jonatas ROM
    q_init = np.array([[-6.2831], [5.1944], [2.0845], [-0.1054], [5.6609], [1.0456], [0.5702]])
    q_init = np.array([[-6.2831], [5.1633], [2.0376], [-0.1054], [5.6609], [0.9818], [0.591]])
    q_init = np.array([[-6.2831], [5.2255], [1.961], [-0.1054], [5.7025], [0.935], [0.591]])
    q_init = np.array([[-6.2831], [5.267], [1.9738], [0.0], [5.7232], [0.952], [0.6739]])
    q_init = np.array([[-6.29694269e+00], [5.28450446e+00], [1.97443606e+00], [-4.29782484e-03], [5.73482051e+00], [9.51224607e-01], [6.76897580e-01]])
    q_init = np.array([[-6.2969], [5.2845], [1.9788], [0.071], [5.7348], [0.9512], [0.6768]])
    q_init = np.array([[-6.2969], [5.2463], [1.9788], [0.0709], [5.7025], [0.9605], [0.7154]])
    q_init = np.array([[-6.2969], [4.956], [2.11], [0.1763], [5.4536], [1.267], [0.7154]])
    q_init = np.array([[-6.3003], [4.9602], [2.1099], [0.1753], [5.4565], [1.2668], [0.568]])
    q_init = np.array([[-6.3003], [4.9352], [2.0916], [0.1753], [5.4536], [1.216], [0.5679]])
    q_init = np.array([[-6.3003], [5.0182], [1.9466], [0.1753], [5.5781], [0.9762], [0.5679]])
    q_init = np.array([[-6.3003], [5.0182], [1.941], [0.1753], [5.5781], [0.9817], [0.5679]])
    q_init = np.array([[-6.3003], [4.9974], [1.9865], [0.1753], [5.5781], [1.0031], [0.5679]]) # 20250728
    q_init = np.array([[-6.3003], [4.9974], [2.0722], [0.1753], [5.5366], [1.1429], [0.5679]]) # 20250728
    q_init = np.array([[-6.3003], [4.956], [2.1067], [0.197], [5.4951], [1.1872], [0.5679]])    # 20250728
    # q_init = np.array([[-6.3003], [4.9974], [1.9865], [0.1753], [5.5781], [1.0031], [0.5679]]) # que foi feito simulacoes
    # testes 30/07/2025
    # q_init = np.array([[-6.3003], [4.956], [2.0969], [0.197], [5.4951], [1.1872], [0.5679]])
    # q_init = np.array([[-6.3003], [4.8938], [2.0969], [0.197], [5.4536], [1.1872], [0.5679]])
    # q_init = np.array([[-6.3003], [4.873], [2.0969], [0.197], [5.4329], [1.197], [0.591]])
    # q_init = np.array([[-6.3003], [5.0182], [2.1117], [0.197], [5.5366], [1.197], [0.591]])
    
    # voltando com o original para testar
    # q_init = np.array([[-6.3003], [4.9974], [1.9865], [0.1753], [5.5781], [1.0031], [0.5679]])
    q_init = np.array([[-6.3003], [4.956], [2.1067], [0.4251], [5.4951], [1.1872], [0.6947]])
    q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])

    # q_init = np.array([[-6.73902495], [ 5.55291305],
    #  [ 1.96513904],
    #  [ 0.07987286],
    #  [ 5.93013991],
    #  [ 1.03142175],
    #  [ 0.62626329]])
    #
    # q_init = np.array([[-8.67574766],
    #  [ 7.9618385 ],
    #  [ 1.20136548],
    #  [ 0.37144367],
    #  [ 7.32128528],
    #  [ 1.82451201],
    #  [ 0.37875036]])
    #
    #
# model_path = "leg_model_cyclic_muscle_inversed_root_3.bioMod"
# model_path = "leg_model_cyclic_muscle_inversed_root_opt_only_seg.bioMod"
# model_path = "leg_model_cyclic_muscle_inversed_root_opt.bioMod"
model = biorbd.Model(model_path)

def print_initial_position():
    """
    This file is to display the human model into bioviz
    """

    export_model = True
    background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
    show_gravity_vector = True #False if export_model else True
    show_floor = False #if export_model else True
    show_local_ref_frame = False if export_model else True
    show_global_ref_frame = True  #False if export_model else True
    show_markers = False if export_model else True
    show_mass_center = False if export_model else True
    show_global_center_of_mass = False if export_model else True
    show_segments_center_of_mass = False if export_model else True

    b = bioviz.Viz(
        model_path,
        show_gravity_vector=show_gravity_vector,
        show_floor=show_floor,
        show_local_ref_frame=show_local_ref_frame,
        show_global_ref_frame=show_global_ref_frame,
        show_markers=show_markers,
        show_mass_center=show_mass_center,
        show_global_center_of_mass=show_global_center_of_mass,
        show_segments_center_of_mass=show_segments_center_of_mass,
        mesh_opacity=1,
        background_color=background_color,
    )
    b.load_movement(q_init)
    b.set_camera_position(0,0,-np.pi/2)
    b.resize(1500, 1000)
    if export_model:
        b.snapshot(f"../opt_sol_results/images/model_plot.png")
    # biorbd_viz.exec()


    # b = bioviz.Viz(model_path)
    b.load_movement(q_init)
    b.exec()


def main():
    print_initial_position()


if __name__ == "__main__":
    main()