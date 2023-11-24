from typing import Callable
import sys
import numpy as np
from casadi import DM_eye, vertcat, Function

import pickle

from .optimization_vector import OptimizationVectorHelper
from .non_linear_program import NonLinearProgram as NLP
from ..dynamics.configure_problem import DynamicsList, Dynamics
from ..dynamics.ode_solver import OdeSolver
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..limits.constraints import (
    ConstraintFcn,
    ConstraintList,
    Constraint,
    ParameterConstraintList,
)
from ..limits.phase_transition import PhaseTransitionList, PhaseTransitionFcn
from ..limits.multinode_constraint import MultinodeConstraintList, MultinodeConstraintFcn
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveList, Objective, ParameterObjectiveList
from ..limits.constraints import ConstraintFunction
from ..limits.path_conditions import BoundsList
from ..limits.path_conditions import InitialGuessList, InitialGuess
from ..limits.penalty_controller import PenaltyController
from ..misc.enums import PhaseDynamics, InterpolationType
from ..misc.__version__ import __version__
from ..misc.enums import Node, ControlType
from ..misc.mapping import BiMappingList, Mapping, NodeMappingList, BiMapping
from ..misc.utils import check_version
from ..optimization.optimal_control_program import OptimalControlProgram
from ..optimization.parameters import ParameterList
from ..optimization.problem_type import SocpType
from ..optimization.solution.solution import Solution
from ..optimization.variable_scaling import VariableScalingList


class StochasticOptimalControlProgram(OptimalControlProgram):
    """
    The main class to define a stochastic ocp. This class prepares the full program and gives all
    the needed interface to modify and solve the program
    """

    def __init__(
        self,
        bio_model: list | tuple | StochasticBioModel,
        dynamics: Dynamics | DynamicsList,
        n_shooting: int | list | tuple,
        phase_time: int | float | list | tuple,
        x_bounds: BoundsList = None,
        u_bounds: BoundsList = None,
        s_bounds: BoundsList = None,
        x_init: InitialGuessList | None = None,
        u_init: InitialGuessList | None = None,
        s_init: InitialGuessList | None = None,
        objective_functions: Objective | ObjectiveList = None,
        constraints: Constraint | ConstraintList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        control_type: ControlType | list = ControlType.CONSTANT,
        variable_mappings: BiMappingList = None,
        time_phase_mapping: BiMapping = None,
        node_mappings: NodeMappingList = None,
        plot_mappings: Mapping = None,
        phase_transitions: PhaseTransitionList = None,
        multinode_constraints: MultinodeConstraintList = None,
        multinode_objectives: MultinodeObjectiveList = None,
        x_scaling: VariableScalingList = None,
        xdot_scaling: VariableScalingList = None,
        u_scaling: VariableScalingList = None,
        s_scaling: VariableScalingList = None,
        n_threads: int = 1,
        use_sx: bool = False,
        integrated_value_functions: dict[str, Callable] = None,
        problem_type=SocpType.TRAPEZOIDAL_IMPLICIT,
        **kwargs,
    ):
        _check_multi_threading_and_problem_type(problem_type, **kwargs)
        _check_has_no_ode_solver_defined(**kwargs)
        _check_has_no_phase_dynamics_shared_during_the_phase(problem_type, **kwargs)

        self.problem_type = problem_type
        self._s_init = s_init
        self._s_bounds = s_bounds
        self._s_scaling = s_scaling

        super(StochasticOptimalControlProgram, self).__init__(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=phase_time,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameter_bounds,
            parameter_init=parameter_init,
            parameter_objectives=parameter_objectives,
            parameter_constraints=parameter_constraints,
            ode_solver=None,
            control_type=control_type,
            variable_mappings=variable_mappings,
            time_phase_mapping=time_phase_mapping,
            node_mappings=node_mappings,
            plot_mappings=plot_mappings,
            phase_transitions=phase_transitions,
            multinode_constraints=multinode_constraints,
            multinode_objectives=multinode_objectives,
            x_scaling=x_scaling,
            xdot_scaling=xdot_scaling,
            u_scaling=u_scaling,
            n_threads=n_threads,
            use_sx=use_sx,
            integrated_value_functions=integrated_value_functions,
        )

    def _declare_multi_node_penalties(
        self,
        multinode_constraints: ConstraintList,
        multinode_objectives: ObjectiveList,
        constraints: ConstraintList,
        phase_transition: PhaseTransitionList,
    ):
        """
        This function declares the multi node penalties (constraints and objectives) to the penalty pool.

        Note
        ----
        This function overrides the method in OptimalControlProgram
        """
        multinode_constraints.add_or_replace_to_penalty_pool(self)
        multinode_objectives.add_or_replace_to_penalty_pool(self)

        # Add the internal multi-node constraints for the stochastic ocp
        if isinstance(self.problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            self._prepare_stochastic_dynamics_explicit(
                constraints=constraints,
            )
        elif isinstance(self.problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            self._prepare_stochastic_dynamics_implicit(
                constraints=constraints,
            )
        elif isinstance(self.problem_type, SocpType.COLLOCATION):
            self._prepare_stochastic_dynamics_collocation(
                constraints=constraints,
                phase_transition=phase_transition,
            )
        else:
            raise RuntimeError("Wrong choice of problem_type, you must choose one of the SocpType.")

    def _prepare_stochastic_dynamics_explicit(self, constraints):
        """
        Adds the internal constraint needed for the explicit formulation of the stochastic ocp.
        """

        constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        penalty_m_dg_dz_list = MultinodeConstraintList()
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0:
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                )
        penalty_m_dg_dz_list.add_or_replace_to_penalty_pool(self)

    def _prepare_stochastic_dynamics_implicit(self, constraints):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp.
        """

        constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        multi_node_penalties = MultinodeConstraintList()
        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_IMPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_IMPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                )

        # Constraints for P
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT,
                node=Node.ALL,
                phase=i_phase,
            )

        # Constraints for A
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_DF_DX_IMPLICIT,
                node=Node.ALL,
                phase=i_phase,
            )

        # Constraints for C
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DF_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DF_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase + 1),
                    nodes=(-1, 0),
                )

        multi_node_penalties.add_or_replace_to_penalty_pool(self)

    def _prepare_stochastic_dynamics_collocation(self, constraints, phase_transition):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp using collocation
        integration. This is the real implementation suggested in Gillis 2013.
        """

        if "ref" in self.nlp[0].stochastic_variables:
            constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_HELPER_MATRIX_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                expand=True,
            )

        # Constraints for P inner-phase
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                expand=True,
            )

        # Constraints for P inter-phase
        for i_phase, nlp in enumerate(self.nlp):
            if len(self.nlp) > 1 and i_phase < len(self.nlp) - 1:
                phase_transition.add(PhaseTransitionFcn.COVARIANCE_CONTINUOUS, phase_pre_idx=i_phase)

    def _set_stochastic_variables_to_original_values(
        self,
        s_init: InitialGuessList,
        s_bounds: BoundsList,
        s_scaling: VariableScalingList,
    ):
        self.original_values["s_init"] = s_init
        self.original_values["s_bounds"] = s_bounds
        self.original_values["s_scaling"] = s_scaling

    def _auto_initialize(self, x_init, u_init, parameter_init, s_init):
        def replace_initial_guess(key, n_var, var_init, s_init, i_phase):
            if n_var != 0:
                if key in s_init:
                    s_init[key] = InitialGuess(var_init, interpolation=InterpolationType.EACH_FRAME, phase=i_phase)
                else:
                    s_init.add(key, initial_guess=var_init, interpolation=InterpolationType.EACH_FRAME, phase=i_phase)

        def get_ref_init(time_vector, x_guess, u_guess, p_guess, nlp):
            casadi_func = Function(
                "sensory_reference",
                [nlp.time_cx, nlp.states.cx_start, nlp.controls.cx_start, nlp.parameters.cx_start],
                [
                    nlp.model.sensory_reference(
                        nlp.time_cx,
                        nlp.states.cx_start,
                        nlp.controls.cx_start,
                        nlp.parameters.cx_start,
                        None,  # Sensory reference should not depend on stochastic variables
                        nlp,
                    )
                ],
            )
            x_guess = x_guess[:, 0 :: (self.problem_type.polynomial_degree + 2)]
            for i in range(nlp.ns + 1):
                ref_init_this_time = casadi_func(
                    time_vector[i],
                    x_guess[:, i],
                    u_guess[:, i],
                    p_guess,
                )
                ref_init = ref_init_this_time if i == 0 else np.hstack((ref_init, ref_init_this_time))
            return ref_init

        def get_m_init(
            time_vector, x_guess, u_guess, p_guess, fake_stochastic_variables, nlp, variable_sizes, Fdz, Gdz
        ):
            m_init = np.zeros((variable_sizes["n_m"], nlp.ns + 1))
            for i in range(nlp.ns):
                index_this_time = [
                    i * (self.problem_type.polynomial_degree + 2) + j
                    for j in range(self.problem_type.polynomial_degree + 2)
                ]
                df_dz = Fdz(
                    time_vector[i],
                    x_guess[:, index_this_time[0]],
                    x_guess[:, index_this_time[1:]],
                    u_guess[:, i],
                    p_guess,
                    fake_stochastic_variables[:, i],
                    nlp.model.motor_noise_magnitude,
                    nlp.model.sensory_noise_magnitude,
                )
                dg_dz = Gdz(
                    time_vector[i],
                    x_guess[:, index_this_time[0]],
                    x_guess[:, index_this_time[1:]],
                    u_guess[:, i],
                    p_guess,
                    fake_stochastic_variables[:, i],
                    nlp.model.motor_noise_magnitude,
                    nlp.model.sensory_noise_magnitude,
                )

                m_this_time = df_dz @ np.linalg.inv(dg_dz)
                m_init[:, i] = np.reshape(StochasticBioModel.reshape_to_vector(m_this_time), (-1,))

            m_init[:, -1] = m_init[:, -2]
            return m_init

        def get_cov_init(
            time_vector,
            x_guess,
            u_guess,
            p_guess,
            fake_stochastic_variables,
            nlp,
            variable_sizes,
            m_init,
            Gdx,
            Gdw,
            initial_covariance,
        ):
            sigma_w_dm = vertcat(nlp.model.motor_noise_magnitude, nlp.model.sensory_noise_magnitude) * DM_eye(
                vertcat(nlp.model.motor_noise_magnitude, nlp.model.sensory_noise_magnitude).shape[0]
            )

            cov_init = np.zeros((variable_sizes["n_cov"], nlp.ns + 1))
            cov_init[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(initial_covariance), (-1,))
            for i in range(nlp.ns):
                index_this_time = [
                    i * (self.problem_type.polynomial_degree + 2) + j
                    for j in range(self.problem_type.polynomial_degree + 2)
                ]
                dg_dx = Gdx(
                    time_vector[i],
                    x_guess[:, index_this_time[0]],
                    x_guess[:, index_this_time[1:]],
                    u_guess[:, i],
                    p_guess,
                    fake_stochastic_variables[:, i],
                    nlp.model.motor_noise_magnitude,
                    nlp.model.sensory_noise_magnitude,
                )
                dg_dw = Gdw(
                    time_vector[i],
                    x_guess[:, index_this_time[0]],
                    x_guess[:, index_this_time[1:]],
                    u_guess[:, i],
                    p_guess,
                    fake_stochastic_variables[:, i],
                    nlp.model.motor_noise_magnitude,
                    nlp.model.sensory_noise_magnitude,
                )

                m_matrix = StochasticBioModel.reshape_to_matrix(m_init[:, i], nlp.model.matrix_shape_m)
                cov_matrix = StochasticBioModel.reshape_to_matrix(cov_init[:, i], nlp.model.matrix_shape_cov)
                cov_this_time = m_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w_dm @ dg_dw.T) @ m_matrix.T
                cov_init[:, i + 1] = np.reshape(StochasticBioModel.reshape_to_vector(cov_this_time), (-1))
            return cov_init

        if not isinstance(self.phase_time, list):
            phase_time = [self.phase_time]
        else:
            phase_time = self.phase_time

        if x_init.type != InterpolationType.ALL_POINTS:
            raise RuntimeError(
                "To initialize automatically the stochastic variables, you need to provide an x_init of type InterpolationType.ALL_POINTS"
            )
        if u_init.type != InterpolationType.EACH_FRAME:
            raise RuntimeError(
                "To initialize automatically the stochastic variables, you need to provide an u_init of type InterpolationType.EACH_FRAME"
            )

        # concatenate parameters into a single vector
        p_guess = np.zeros((0, 1))
        for key in self.parameters.keys():
            p_guess = np.concatenate((p_guess, parameter_init[key].init), axis=0)

        for i_phase, nlp in enumerate(self.nlp):
            time_vector = np.linspace(0, phase_time[i_phase], nlp.ns + 1)
            n_ref = nlp.model.n_references
            n_k = nlp.model.matrix_shape_k[0] * nlp.model.matrix_shape_k[1]
            n_m = nlp.model.matrix_shape_m[0] * nlp.model.matrix_shape_m[1]
            n_cov = nlp.model.matrix_shape_cov[0] * nlp.model.matrix_shape_cov[1]
            n_stochastic = n_ref + n_k + n_m + n_cov
            variable_sizes = {
                "n_ref": n_ref,
                "n_k": n_k,
                "n_m": n_m,
                "n_cov": n_cov,
                "n_stochastic": n_stochastic,
            }

            # concatenate x_init into a single matrix
            x_guess = np.zeros((0, (self.problem_type.polynomial_degree + 2) * nlp.ns + 1))
            for key in x_init[i_phase].keys():
                x_guess = np.concatenate((x_guess, x_init[i_phase][key].init), axis=0)

            # concatenate u_init into a single matrix
            if nlp.control_type == ControlType.CONSTANT:
                u_guess = np.zeros((0, nlp.ns))
            elif (
                nlp.control_type == ControlType.LINEAR_CONTINUOUS
                or nlp.control_type == ControlType.CONSTANT_WITH_LAST_NODE
            ):
                u_guess = np.zeros((0, nlp.ns + 1))
            elif nlp.control_type == ControlType.NONE:
                u_guess = np.zeros((0, 0))
            else:
                raise RuntimeError(
                    "The automatic initialization of stochastic variables is not implemented yet for nlp with control_type other than ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.LINEAR_CONTINUOUS or ControlType.NONE."
                )
            for key in u_init[i_phase].keys():
                u_guess = np.concatenate((u_guess, u_init[i_phase][key].init), axis=0)

            k_init = np.ones((n_k, nlp.ns + 1)) * 0.01
            replace_initial_guess("k", n_k, k_init, s_init, i_phase)

            ref_init = get_ref_init(time_vector, x_guess, u_guess, p_guess, nlp)
            replace_initial_guess("ref", n_ref, ref_init, s_init, i_phase)

            # Define the casadi functions needed to initialize m and cov
            penalty = Constraint(ConstraintFcn.STOCHASTIC_HELPER_MATRIX_COLLOCATION)
            fake_stochastic_variables = np.zeros((variable_sizes["n_stochastic"], nlp.ns + 1))
            fake_stochastic_variables[: variable_sizes["n_k"], :] = k_init
            fake_stochastic_variables[
                variable_sizes["n_k"] : variable_sizes["n_k"] + variable_sizes["n_ref"], :
            ] = ref_init
            penalty_controller = PenaltyController(
                ocp=self,
                nlp=nlp,
                t=time_vector,
                x=x_guess,
                u=u_guess,
                x_scaled=[],
                u_scaled=[],
                p=p_guess,
                s=fake_stochastic_variables,
                s_scaled=[],
                node_index=0,
            )
            _, _, Gdx, Gdz, Gdw, Fdz = ConstraintFunction.Functions.collocation_jacobians(penalty, penalty_controller)

            m_init = get_m_init(
                time_vector, x_guess, u_guess, p_guess, fake_stochastic_variables, nlp, variable_sizes, Fdz, Gdz
            )
            replace_initial_guess("m", n_m, m_init, s_init, i_phase)

            if i_phase == 0:
                initial_covariance = self.problem_type.initial_cov
            else:
                initial_covariance = cov_init[:, -1]
            cov_init = get_cov_init(
                time_vector,
                x_guess,
                u_guess,
                p_guess,
                fake_stochastic_variables,
                nlp,
                variable_sizes,
                m_init,
                Gdx,
                Gdw,
                initial_covariance,
            )
            replace_initial_guess("cov", n_cov, cov_init, s_init, i_phase)

    def _prepare_bounds_and_init(
        self, x_bounds, u_bounds, parameter_bounds, s_bounds, x_init, u_init, parameter_init, s_init
    ):
        self.parameter_bounds = BoundsList()
        self.parameter_init = InitialGuessList()

        if isinstance(self.problem_type, SocpType.COLLOCATION) and self.problem_type.auto_initialization == True:
            self._auto_initialize(x_init, u_init, parameter_init, s_init)

        self.update_bounds(x_bounds, u_bounds, parameter_bounds, s_bounds)
        self.update_initial_guess(x_init, u_init, parameter_init, s_init)
        # Define the actual NLP problem
        OptimizationVectorHelper.declare_ocp_shooting_points(self)

    @staticmethod
    def load(file_path: str) -> list:
        """
        Reload a previous optimization (*.bo) saved using save

        Parameters
        ----------
        file_path: str
            The path to the *.bo file

        Returns
        -------
        The ocp and sol structure. If it was saved, the iterations are also loaded
        """

        with open(file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except BaseException as error_message:
                raise ValueError(
                    f"The file '{file_path}' cannot be loaded, maybe the version of bioptim (version {__version__})\n"
                    f"is not the same as the one that created the file (version unknown). For more information\n"
                    "please refer to the original error message below\n\n"
                    f"{type(error_message).__name__}: {error_message}"
                )
            ocp = StochasticOptimalControlProgram.from_loaded_data(data["ocp_initializer"])
            for key in data["versions"].keys():
                key_module = "biorbd_casadi" if key == "biorbd" else key
                try:
                    check_version(sys.modules[key_module], data["versions"][key], ocp.version[key], exclude_max=False)
                except ImportError:
                    raise ImportError(
                        f"Version of {key} from file ({data['versions'][key]}) is not the same as the "
                        f"installed version ({ocp.version[key]})"
                    )
            sol = data["sol"]
            sol.ocp = Solution.SimplifiedOCP(ocp)
            out = [ocp, sol]
        return out

    def _set_default_ode_solver(self):
        """It overrides the method in OptimalControlProgram that set a RK4 by default"""
        if isinstance(self.problem_type, SocpType.TRAPEZOIDAL_IMPLICIT) or isinstance(
            self.problem_type, SocpType.TRAPEZOIDAL_EXPLICIT
        ):
            return OdeSolver.TRAPEZOIDAL()
        elif isinstance(self.problem_type, SocpType.COLLOCATION):
            return OdeSolver.COLLOCATION(
                method=self.problem_type.method,
                polynomial_degree=self.problem_type.polynomial_degree,
                duplicate_collocation_starting_point=True,
            )
        else:
            raise RuntimeError("Wrong choice of problem_type, you must choose one of the SocpType.")

    def _set_stochastic_internal_stochastic_variables(self):
        """
        Set the stochastic variables to their internal values

        Note
        ----
        This method overrides the method in OptimalControlProgram
        """
        return (
            self._s_init,
            self._s_bounds,
            self._s_scaling,
        )  # Nothing to do here as they are already set before calling super().__init__

    def _set_nlp_is_stochastic(self):
        """
        Set the is_stochastic variable to True for all the nlp

        Note
        ----
        This method overrides the method in OptimalControlProgram
        """
        NLP.add(self, "is_stochastic", True, True)


def _check_multi_threading_and_problem_type(problem_type, **kwargs):
    if not isinstance(problem_type, SocpType.COLLOCATION):
        if "n_thread" in kwargs:
            if kwargs["n_thread"] != 1:
                raise ValueError(
                    "Multi-threading is not possible yet while solving a trapezoidal stochastic ocp."
                    "n_thread is set to 1 by default."
                )


def _check_has_no_ode_solver_defined(**kwargs):
    if "ode_solver" in kwargs:
        raise ValueError(
            "The ode_solver cannot be defined for a stochastic ocp. "
            "The value is chosen based on the type of problem solved:"
            "\n- TRAPEZOIDAL_EXPLICIT: OdeSolver.TRAPEZOIDAL() "
            "\n- TRAPEZOIDAL_IMPLICIT: OdeSolver.TRAPEZOIDAL() "
            "\n- COLLOCATION: "
            "OdeSolver.COLLOCATION("
            "method=problem_type.method, "
            "polynomial_degree=problem_type.polynomial_degree, "
            "duplicate_collocation_starting_point=True"
            ")"
        )


def _check_has_no_phase_dynamics_shared_during_the_phase(problem_type, **kwargs):
    if not isinstance(problem_type, SocpType.COLLOCATION):
        if "phase_dynamics" in kwargs:
            if kwargs["phase_dynamics"] == PhaseDynamics.SHARED_DURING_THE_PHASE:
                raise ValueError(
                    "The dynamics cannot be SHARED_DURING_THE_PHASE with a trapezoidal stochastic ocp."
                    "phase_dynamics is set to PhaseDynamics.ONE_PER_NODE by default."
                )
