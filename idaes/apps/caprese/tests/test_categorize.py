##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Test categorize_dae_variables_and_constraints function.
"""

import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet
from pyomo.dae.flatten import flatten_dae_components

from idaes.apps.caprese.categorize import (
        categorize_dae_variables_and_constraints,
        )
from idaes.apps.caprese.common.config import VariableCategory as VC
from idaes.apps.caprese.common.config import ConstraintCategory as CC
from idaes.apps.caprese.tests.test_simple_model import make_model

__author__ = "Robert Parker"

@pytest.mark.unit
def test_categorize_deriv():
    """ The simplest test. Identify a differential and a derivative var.
    """
    m = pyo.ConcreteModel()
    m.time = dae.ContinuousSet(initialize=[0, 1])
    m.v = pyo.Var(m.time, initialize=0)
    m.dv = dae.DerivativeVar(m.v, wrt=m.time)
    m.diff_eqn = pyo.Constraint(
            m.time,
            rule={t: m.dv[t] == -m.v[t]**2 for t in m.time}
            )
    with pytest.raises(TypeError):
        # If we find a derivative var, we will try to access the disc eq.
        var_partition, con_partition = categorize_dae_variables_and_constraints(
                m,
                [m.v, m.dv],
                [m.diff_eqn],
                m.time,
                )
    disc = pyo.TransformationFactory('dae.finite_difference')
    disc.apply_to(m, wrt=m.time, nfe=1, scheme='BACKWARD')
    var_partition, con_partition = categorize_dae_variables_and_constraints(
            m,
            [m.v, m.dv],
            [m.diff_eqn, m.dv_disc_eq],
            m.time,
            )
    assert len(var_partition[VC.DIFFERENTIAL]) == 1
    assert var_partition[VC.DIFFERENTIAL][0] is m.v
    assert len(var_partition[VC.DERIVATIVE]) == 1
    assert var_partition[VC.DERIVATIVE][0] is m.dv
    assert len(con_partition[CC.DIFFERENTIAL]) == 1
    assert con_partition[CC.DIFFERENTIAL][0] is m.diff_eqn
    assert len(con_partition[CC.DISCRETIZATION]) == 1
    assert con_partition[CC.DISCRETIZATION][0] is m.dv_disc_eq

    for categ in VC:
        if (categ is not VC.DIFFERENTIAL and categ is not VC.DERIVATIVE
                and categ in var_partition):
            assert len(var_partition[categ]) == 0
    for categ in CC:
        if (categ is not CC.DIFFERENTIAL and categ is not CC.DISCRETIZATION
                and categ in con_partition):
            assert len(con_partition[categ]) == 0


@pytest.mark.unit
def test_categorize_deriv_fixed():
    """ If one of the derivative or diff var are fixed, the other
    should be categorized as algebraic.
    """
    m = pyo.ConcreteModel()
    m.time = dae.ContinuousSet(initialize=[0, 1])
    m.v = pyo.Var(m.time, initialize=0)
    m.dv = dae.DerivativeVar(m.v, wrt=m.time)
    m.diff_eqn = pyo.Constraint(
            m.time,
            rule={t: m.dv[t] == -m.v[t]**2 for t in m.time}
            )
    disc = pyo.TransformationFactory('dae.finite_difference')
    disc.apply_to(m, wrt=m.time, nfe=1, scheme='BACKWARD')
    
    #
    # Fix differential variable, e.g. it is an input
    #
    m.v.fix()
    m.diff_eqn.deactivate()

    var_partition, con_partition = categorize_dae_variables_and_constraints(
            m,
            [m.v, m.dv],
            [m.diff_eqn, m.dv_disc_eq],
            m.time,
            )
    # Expected categories have expected variables
    assert len(var_partition[VC.ALGEBRAIC]) == 1
    assert var_partition[VC.ALGEBRAIC][0] is m.dv
    assert len(con_partition[CC.ALGEBRAIC]) == 1
    assert con_partition[CC.ALGEBRAIC][0] is m.dv_disc_eq

    # Unexpected categories are empty
    for categ in VC:
        if (categ is not VC.ALGEBRAIC and categ is not VC.UNUSED
                and categ in var_partition):
            assert len(var_partition[categ]) == 0
    for categ in CC:
        if (categ is not CC.ALGEBRAIC and categ is not CC.UNUSED
                and categ in con_partition):
            assert len(con_partition[categ]) == 0

    #
    # We can accomplish something similar by making m.v an input
    #
    m.v.unfix()
    var_partition, con_partition = categorize_dae_variables_and_constraints(
            m,
            [m.v, m.dv],
            [m.diff_eqn, m.dv_disc_eq],
            m.time,
            input_vars=[m.v],
            )
    # Expected categories have expected variables
    assert len(var_partition[VC.ALGEBRAIC]) == 1
    assert var_partition[VC.ALGEBRAIC][0] is m.dv
    assert len(var_partition[VC.INPUT]) == 1
    assert var_partition[VC.INPUT][0] is m.v
    assert len(con_partition[CC.ALGEBRAIC]) == 1
    assert con_partition[CC.ALGEBRAIC][0] is m.dv_disc_eq

    #
    # Fix derivative var, e.g. pseudo-steady state
    #
    for var in m.dv.values():
        var.fix(0.0)
    m.diff_eqn.activate()
    m.dv_disc_eq.deactivate()
    var_partition, con_partition = categorize_dae_variables_and_constraints(
            m,
            [m.v, m.dv],
            [m.diff_eqn, m.dv_disc_eq],
            m.time,
            )
    # Expected categories have expected variables
    assert len(var_partition[VC.ALGEBRAIC]) == 1
    assert var_partition[VC.ALGEBRAIC][0] is m.v
    assert len(con_partition[CC.ALGEBRAIC]) == 1
    assert con_partition[CC.ALGEBRAIC][0] is m.diff_eqn


@pytest.mark.unit
def test_categorize_simple_model():
    """ Categorize variables and equations in the "simple model" used
    for the base class unit tests.
    """
    m = make_model()
    m.conc_in.unfix()
    m.flow_in.unfix()
    scalar_vars, dae_vars = flatten_dae_components(m, m.time, pyo.Var)
    scalar_cons, dae_cons = flatten_dae_components(m, m.time, pyo.Constraint)
    var_partition, con_partition = categorize_dae_variables_and_constraints(
            m,
            dae_vars,
            dae_cons,
            m.time,
            input_vars=[m.flow_in],
            disturbance_vars=[
                pyo.Reference(m.conc_in[:, 'A']),
                pyo.Reference(m.conc_in[:, 'B']),
                ],
            )
    t1 = m.time[2]
    # Expected variables:
    expected_vars = {
            VC.DIFFERENTIAL: ComponentSet([m.conc[t1, 'A'], m.conc[t1, 'B']]),
            VC.DERIVATIVE: ComponentSet([m.dcdt[t1, 'A'], m.dcdt[t1, 'B']]),
            VC.ALGEBRAIC: ComponentSet([
                    m.rate[t1, 'A'],
                    m.rate[t1, 'B'],
                    m.flow_out[t1],
                    ]),
            VC.INPUT: ComponentSet([m.flow_in[t1]]),
            VC.DISTURBANCE: ComponentSet([
                    m.conc_in[t1, 'A'],
                    m.conc_in[t1, 'B'],
                    ]),
            }

    # Expected constraints:
    expected_cons = {
            CC.DIFFERENTIAL: ComponentSet([
                    m.material_balance[t1, 'A'],
                    m.material_balance[t1, 'B'],
                    ]),
            CC.DISCRETIZATION: ComponentSet([
                    m.dcdt_disc_eq[t1, 'A'],
                    m.dcdt_disc_eq[t1, 'B'],
                    ]),
            CC.ALGEBRAIC: ComponentSet([
                    m.rate_eqn[t1, 'A'],
                    m.rate_eqn[t1, 'B'],
                    m.flow_eqn[t1],
                    ]),
            }

    # Expected categories have expected variables and constraints
    for categ in expected_vars:
        assert len(expected_vars[categ]) == len(var_partition[categ])
        for var in var_partition[categ]:
            assert var[t1] in expected_vars[categ]
    for categ in var_partition:
        if categ not in expected_vars:
            assert len(var_partition[categ]) == 0

    for categ in expected_cons:
        assert len(expected_cons[categ]) == len(con_partition[categ])
        for con in con_partition[categ]:
            assert con[t1] in expected_cons[categ]
    for categ in con_partition:
        if categ not in expected_cons:
            assert len(con_partition[categ]) == 0
