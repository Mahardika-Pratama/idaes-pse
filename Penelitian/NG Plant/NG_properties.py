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
    super(HDAParameterData, self).build()
    self._state_block_class = IdealStateBlock

    self.CH4= Component()
    self.toluene = Component()
    self.methane = Component()
    self.hydrogen = Component()

    self.Liq = LiquidPhase()
    self.Vap = VaporPhase()

    # List of components in each phase (optional)
    self.phase_comp = {"Liq": self.component_list,
                    "Vap": self.component_list}

    # List of phase equilibrium index
    self.phase_equilibrium_idx = Set(initialize=[1, 2, 3, 4])