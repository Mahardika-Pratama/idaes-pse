import logging
from pyomo.environ import Constraint, Expression, log, NonNegativeReals,\
    Var, Set, Param, sqrt, log10, units as pyunits
from pyomo.opt import TerminationCondition
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core import (declare_process_block_class,
                        MaterialFlowBasis,
                        PhysicalParameterBlock,
                        StateBlockData,
                        StateBlock,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        Component,
                        LiquidPhase,
                        VaporPhase)

from idaes.core.util.constants import Constants as const
from idaes.core.util.initialization import (fix_state_vars,
                                            revert_state_vars,
                                            solve_indexed_blocks)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.model_statistics import degrees_of_freedom, \
                                             number_unfixed_variables
from idaes.core.util.misc import extract_data
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)

@declare_process_block_class("SMRParameterBlock")
class SMRParameterData(PhysicalParameterBlock):
    CONFIG = PhysicalParameterBlock.CONFIG()

def build(self):
    super(SMRParameterData, self).build()
    self._state_block_class = _SMRStateBlock

    self.ch4 = Component()
    self.h2o = Component()
    self.co2 = Component()
    self.h2 = Component()

    self.Liq = LiquidPhase()
    self.Vap = VaporPhase()

    # List of components in each phase (optional)
    self.phase_comp = { "Liq": self.component_list,
                        "Vap": self.component_list
                        }

    # List of phase equilibrium index
    self.phase_equilibrium_idx = Set(initialize=[1, 2, 3, 4])

    class _SMRStateBlock(StateBlock):
    def initialize(blk, state_args={}, state_vars_fixed=False,
                   hold_state=False, outlvl=idaeslog.NOTSET,
                   solver=None, optarg=None):
        """
        Initialization routine for property package.
        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. The keys for the state_args dictionary are:

                         flow_mol : value at which to initialize flow rate
                         mole_frac_comp : dict of values to use when
                                          initializing mole fractions
                         pressure : value at which to initialize pressure
                         temperature : value at which to initialize temperature
            outlvl : sets logger output level for initialization routine
            optarg : solver options dictionary object (default=None)
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states variables are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 release_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

        # Fix state variables if not already fixed
        if state_vars_fixed is False:
            flags = fix_state_vars(blk, state_args)

        else:
            pass

        # Deactivate sum of mole fractions constraint
        for k in blk.keys():
            if blk[k].config.defined_state is False:
                blk[k].mole_fraction_constraint.deactivate()

        # Check that degrees of freedom are zero after fixing state vars
        for k in blk.keys():
            if degrees_of_freedom(blk[k]) != 0:
                raise Exception("State vars fixed but degrees of freedom "
                                "for state block is not zero during "
                                "initialization.")

        # Set solver options
        if optarg is None:
            optarg = {"tol": 1e-8}

        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize property calculations

        # Check that there is something to solve for
        free_vars = 0
        for k in blk.keys():
            free_vars += number_unfixed_variables(blk[k])
        if free_vars > 0:
            # If there are free variables, call the solver to initialize
            try:
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = solve_indexed_blocks(opt, [blk], tee=slc.tee)
            except:
                res = None
        else:
            res = None

        init_log.info("Properties Initialized {}.".format(
            idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Return state to initial conditions
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)

        init_log.info("Initialization Complete")
def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        '''
        Method to release state variables fixed during initialization.
        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        '''
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")

        # Reactivate sum of mole fractions constraint
        for k in blk.keys():
            if blk[k].config.defined_state is False:
                blk[k].mole_fraction_constraint.activate()

        if flags is not None:
            # Unfix state variables
            revert_state_vars(blk, flags)

        init_log.info_high("State Released.")


@declare_process_block_class("SMRStateBlock")
class HDAStateBlockData(StateBlockData):
    """
    Example property package for an ideal gas containing benzene, toluene
    hydrogen, methane and diphenyl.
    """

    def build(self):
        """Callable method for Block construction."""
        super(HDAStateBlockData, self).build()

        # Add state variables
        self.flow_mol = Var(
                initialize=1,
                bounds=(1e-8, 1000),
                units=pyunits.mol/pyunits.s,
                doc='Molar flow rate')
        self.mole_frac_comp = Var(self.component_list,
                                  initialize=0.2,
                                  bounds=(0, None),
                                  units=pyunits.dimensionless,
                                  doc="Component mole fractions")
        self.pressure = Var(initialize=101325,
                            bounds=(101325, 400000),
                            units=pyunits.Pa,
                            doc='State pressure')
        self.temperature = Var(initialize=298.15,
                               bounds=(298.15, 1500),
                               units=pyunits.K,
                               doc='State temperature')

        self.mw_comp = Reference(self.params.mw_comp)

        if self.config.defined_state is False:
            self.mole_fraction_constraint = Constraint(
                expr=1e3 == sum(1e3*self.mole_frac_comp[j]
                                for j in self.component_list))

        self.dens_mol = Var(initialize=1,
                            units=pyunits.mol/pyunits.m**3,
                            doc="Mixture density")

        self.ideal_gas_eq = Constraint(
            expr=self.pressure ==
            const.gas_constant*self.temperature*self.dens_mol)

    def _enth_mol(self):
        # Specific enthalpy
        def enth_rule(b):
            params = self.params
            T = self.temperature
            Tr = params.temperature_ref
            return sum(self.mole_frac_comp[j] * (
                           (params.cp_mol_ig_comp_coeff_D[j]/4)*(T**4-Tr**4) +
                           (params.cp_mol_ig_comp_coeff_C[j]/3)*(T**3-Tr**3) +
                           (params.cp_mol_ig_comp_coeff_B[j]/2)*(T**2-Tr**2) +
                           params.cp_mol_ig_comp_coeff_A[j]*(T-Tr) +
                           params.enth_mol_form_vap_comp_ref[j])
                       for j in self.component_list)
        self.enth_mol = Expression(rule=enth_rule)

    def get_material_flow_terms(self, p, j):
        return self.flow_mol * self.mole_frac_comp[j]

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.flow_mol * self.enth_mol

    def default_material_balance_type(self):
        return MaterialBalanceType.componentPhase

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        return {"flow_mol": self.flow_mol,
                "mole_frac_comp": self.mole_frac_comp,
                "temperature": self.temperature,
                "pressure": self.pressure}
