import numpy as np
from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    BiorbdModel,
    ParameterList,
)
from bioptim.misc.fcn_enum import FcnEnum
from custom_package.fes_dynamics import CustomDynamicFunctions


class ConfigureDingModelPulseWidth:

    @staticmethod
    def configure_fes_dynamics(ocp: OptimalControlProgram,
                               nlp: NonLinearProgram,
                               with_contact: bool = False,
                               with_torque: bool = False,
                               with_resistive_torque: bool = False,
                               with_friction: bool = False,
                               fes_driven_task_flag: bool = False,
                               prediction_flag: bool = False,
                               only_parameter_a: bool = False,
                               identification_flag: bool = False,
                               **extra_parameters):
        # control variables
        if identification_flag or prediction_flag:
            ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                    ["fes_input"],
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)
        else:
            ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                    nlp.model.muscle_names,
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)
        if with_resistive_torque:
            ConfigureProblem.configure_new_variable("resistive_torque",
                                                        ["gear_torque"],
                                                        ocp,
                                                        nlp,
                                                        as_states=False,
                                                        as_controls=True)

        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        # state variables
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        name_elements = []
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureProblem.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                    name_elements,
                                                    ocp,
                                                    nlp,
                                                    as_states=True,
                                                    as_controls=False)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            CustomDynamicFunctions.fes_dynamics,
            with_contact=with_contact,
            with_torque=with_torque,
            with_friction=with_friction,
            fes_driven_task_flag=fes_driven_task_flag,
            prediction_flag=prediction_flag,
            only_parameter_a=only_parameter_a,
            identification_flag=identification_flag,
            **extra_parameters)
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp,
                                                        nlp,
                                                        CustomDynamicFunctions.contact_forces_from_fes_driven,
                                                        with_torque=with_torque,
                                                        with_friction=with_friction,
                                                        **extra_parameters)

    @staticmethod
    def configure_fes_cycling_dynamics(ocp: OptimalControlProgram,
                                       nlp: NonLinearProgram,
                                       with_contact: bool = False,
                                       with_torque: bool = False,
                                       with_friction: bool = False,
                                       with_resistive_torque: bool = False,
                                       **extra_parameters):
        # control variables
        ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                    nlp.model.muscle_names,
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)
        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        if with_resistive_torque:
            ConfigureProblem.configure_new_variable("resistive_torque",
                                                        ["gear_torque"],
                                                        ocp,
                                                        nlp,
                                                        as_states=False,
                                                        as_controls=True)
        # state variables
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        name_elements = []
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureProblem.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                    name_elements,
                                                    ocp,
                                                    nlp,
                                                    as_states=True,
                                                    as_controls=False)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            CustomDynamicFunctions.fes_cycling_dynamics,
            with_contact=with_contact,
            with_torque=with_torque,
            with_friction=with_friction,
            with_resistive_torque=with_resistive_torque,
            **extra_parameters)
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp,
                                                        nlp,
                                                        CustomDynamicFunctions.contact_forces_from_fes_cycling,
                                                        with_torque=with_torque,
                                                        with_friction=with_friction,
                                                        with_resistive_torque=with_resistive_torque,
                                                        **extra_parameters)

    @staticmethod
    def configure_fes_param_ident_dynamics(ocp: OptimalControlProgram,
                                           nlp: NonLinearProgram,
                                           with_contact: bool = False,
                                           with_torque: bool = False,
                                           with_resistive_torque: bool = False,
                                           with_friction: bool = False,
                                           fes_driven_task_flag: bool = False,
                                           prediction_flag: bool = False,
                                           only_parameter_a: bool = False,
                                           identification_flag: bool = False,
                                           **extra_parameters):
        # control variables
        ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                ["fes_input"],  # 1 channel only
                                                ocp,
                                                nlp,
                                                as_states=False,
                                                as_controls=True)
        if with_resistive_torque:
            ConfigureProblem.configure_new_variable("resistive_torque",
                                                    ["gear_torque"],
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)

        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        # state variables
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        name_elements = []
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureProblem.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                    name_elements,
                                                    ocp,
                                                    nlp,
                                                    as_states=True,
                                                    as_controls=False)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            CustomDynamicFunctions.fes_param_indent_dynamics,
            with_contact=with_contact,
            with_torque=with_torque,
            with_resistive_torque=with_resistive_torque,
            with_friction=with_friction,
            fes_driven_task_flag=fes_driven_task_flag,
            prediction_flag=prediction_flag,
            only_parameter_a=only_parameter_a,
            identification_flag=identification_flag,
            **extra_parameters)
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp,
                                                        nlp,
                                                        CustomDynamicFunctions.contact_forces_from_fes_driven,
                                                        with_resistive_torque=with_resistive_torque,
                                                        with_torque=with_torque,
                                                        **extra_parameters)

    @staticmethod
    def configure_fes_prediction_dynamics(ocp: OptimalControlProgram,
                                           nlp: NonLinearProgram,
                                           with_contact: bool = False,
                                           with_torque: bool = False,
                                           with_resistive_torque: bool = False,
                                           with_friction: bool = False,
                                           fes_driven_task_flag: bool = False,
                                           prediction_flag: bool = False,
                                           only_parameter_a: bool = False,
                                           identification_flag: bool = False,
                                           **extra_parameters):
        # control variables
        ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                ["fes_input"],  # 1 channel only
                                                ocp,
                                                nlp,
                                                as_states=False,
                                                as_controls=True)
        if with_resistive_torque:
            ConfigureProblem.configure_new_variable("resistive_torque",
                                                    ["gear_torque"],
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)

        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        # state variables
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        name_elements = []
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureProblem.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                    name_elements,
                                                    ocp,
                                                    nlp,
                                                    as_states=True,
                                                    as_controls=False)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            CustomDynamicFunctions.fes_prediction_dynamics,
            with_contact=with_contact,
            with_torque=with_torque,
            with_resistive_torque=with_resistive_torque,
            with_friction=with_friction,
            fes_driven_task_flag=fes_driven_task_flag,
            prediction_flag=prediction_flag,
            only_parameter_a=only_parameter_a,
            identification_flag=identification_flag,
            **extra_parameters)
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp,
                                                        nlp,
                                                        CustomDynamicFunctions.contact_forces_from_fes_driven,
                                                        with_resistive_torque=with_resistive_torque,
                                                        with_torque=with_torque,
                                                        **extra_parameters)


    @staticmethod
    def configure_fes_param_ident_dynamics_vel(ocp: OptimalControlProgram,
                                           nlp: NonLinearProgram,
                                           with_contact: bool = False,
                                           with_torque: bool = False,
                                           with_resistive_torque: bool = False,
                                           with_friction: bool = False,
                                           fes_driven_task_flag: bool = False,
                                           prediction_flag: bool = False,
                                           only_parameter_a: bool = False,
                                           identification_flag: bool = False,
                                           **extra_parameters):
        # control variables
        ConfigureProblem.configure_new_variable("pw",  # pulse width variable
                                                ["fes_input"],  # 1 channel only
                                                ocp,
                                                nlp,
                                                as_states=False,
                                                as_controls=True)
        if with_resistive_torque:
            ConfigureProblem.configure_new_variable("resistive_torque",
                                                    ["gear_torque"],
                                                    ocp,
                                                    nlp,
                                                    as_states=False,
                                                    as_controls=True)

        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        # state variables
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        name_elements = []
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureProblem.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                    name_elements,
                                                    ocp,
                                                    nlp,
                                                    as_states=True,
                                                    as_controls=False)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            CustomDynamicFunctions.fes_param_isovelocity_indent_dynamics,
            with_contact=with_contact,
            with_torque=with_torque,
            with_resistive_torque=with_resistive_torque,
            with_friction=with_friction,
            fes_driven_task_flag=fes_driven_task_flag,
            prediction_flag=prediction_flag,
            only_parameter_a=only_parameter_a,
            identification_flag=identification_flag,
            **extra_parameters)
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp,
                                                        nlp,
                                                        CustomDynamicFunctions.contact_forces_from_fes_driven,
                                                        with_resistive_torque=with_resistive_torque,
                                                        with_torque=with_torque,
                                                        **extra_parameters)


class FesDynamicsFcn(FcnEnum):
    """
    Selection of valid dynamics functions
    """
    FES_DRIVEN = ConfigureDingModelPulseWidth.configure_fes_dynamics
    FES_CYCLING = ConfigureDingModelPulseWidth.configure_fes_cycling_dynamics
    FES_PARAM_IDENT = ConfigureDingModelPulseWidth.configure_fes_param_ident_dynamics
    FES_PARAM_PREDICTION = ConfigureDingModelPulseWidth.configure_fes_prediction_dynamics
    FES_PARAM_IDENT_VEL = ConfigureDingModelPulseWidth.configure_fes_param_ident_dynamics_vel
