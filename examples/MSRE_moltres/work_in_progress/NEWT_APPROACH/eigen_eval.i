[GlobalParams]
  num_groups = 2
  num_precursor_groups = 6
  use_exp_form = false
  group_fluxes = 'group1 group2'
  temperature = temp
  sss2_input = false
  pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6'
  account_delayed = true
[]

[Problem]
  allow_initial_conditions_with_restart = true
[]

[Mesh]
  file = './precursor_dist_calc.e'
[../]

[AuxVariables]
  [./temp]
    scaling = 1e-4
    initial_from_file_var = temp
    initial_from_file_timestep = LATEST
  [../]
  [pre1]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre1
    initial_from_file_timestep = LATEST
    block = '0'
  []
  [pre2]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre2
    initial_from_file_timestep = LATEST
    block = '0'
  []
  [pre3]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre3
    initial_from_file_timestep = LATEST
    block = '0'
  []
  [pre4]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre4
    initial_from_file_timestep = LATEST
    block = '0'
  []
  [pre5]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre5
    initial_from_file_timestep = LATEST
    block = '0'
  []
  [pre6]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre6
    initial_from_file_timestep = LATEST
    block = '0'
  []
[]

[Variables]
  [./group1]
    order = FIRST
    family = LAGRANGE
    initial_from_file_var = group1
    initial_from_file_timestep = LATEST
    scaling = 1e4
  [../]
  [./group2]
    order = FIRST
    family = LAGRANGE
    scaling = 1e4
    initial_from_file_var = group2
    initial_from_file_timestep = LATEST
  [../]
[]


[Nt]
  var_name_base = group
  vacuum_boundaries = 'fuel_bottom fuel_top mod_bottom mod_top right'
  pre_blocks = '0'
  create_temperature_var = false
  eigen = true
[]

[Materials]
  [./fuel]
    type = GenericMoltresMaterial
    property_tables_root = './newt_msre_fuel_'
    interp_type = 'spline'
    block = '0'
    prop_names = 'k cp'
    prop_values = '.0553 1967' # Robertson MSRE technical report @ 922 K
  [../]
  [./rho_fuel]
    type = DerivativeParsedMaterial
    f_name = rho
    function = '2.146e-3 * exp(-1.8 * 1.18e-4 * (temp - 922))'
    coupled_variables = 'temp'
    derivative_order = 1
    block = '0'
  [../]
  [./moder]
    type = GenericMoltresMaterial
    property_tables_root = './newt_msre_mod_'
    interp_type = 'spline'
    prop_names = 'k cp'
    prop_values = '.312 1760' # Cammi 2011 at 908 K
    block = '1'
  [../]
  [./rho_moder]
    type = DerivativeParsedMaterial
    f_name = rho
    function = '1.86e-3 * exp(-1.8 * 1.0e-5 * (temp - 922))'
    coupled_variables = 'temp'
    derivative_order = 1
    block = '1'
  [../]
[]

[Executioner]
  type = Eigenvalue
  initial_eigenvalue = 1
  solve_type = 'PJFNK'
  petsc_options = '-snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'

  automatic_scaling = true
  compute_scaling_once = false
  resid_vs_jac_scaling_param = 0.1

  line_search = none
[]




[Postprocessors]
  [pre1_integral]
    type = ElementIntegralVariablePostprocessor
    variable = pre1
    execute_on = linear
    block = '0'
  []
  [k_eff]
    type = VectorPostprocessorComponent
    index = 0
    vectorpostprocessor = k_vpp
    vector_name = eigen_values_real
  []
  [bnorm]
    type = ElmIntegTotFissNtsPostprocessor
    block = '0'
    execute_on = linear
  []
  [tot_fissions]
    type = ElmIntegTotFissPostprocessor
    execute_on = linear
  []
  [powernorm]
    type = ElmIntegTotFissHeatPostprocessor
    execute_on = linear
  []
  [group1norm]
    type = ElementIntegralVariablePostprocessor
    variable = group1
    execute_on = linear
  []
  [group1max]
    type = NodalExtremeValue
    value_type = max
    variable = group1
    execute_on = timestep_end
  []
  [group1diff]
    type = ElementL2Diff
    variable = group1
    execute_on = 'linear timestep_end'
    use_displaced_mesh = false
  []
  [group2norm]
    type = ElementIntegralVariablePostprocessor
    variable = group2
    execute_on = linear
  []
  [group2max]
    type = NodalExtremeValue
    value_type = max
    variable = group2
    execute_on = timestep_end
  []
  [group2diff]
    type = ElementL2Diff
    variable = group2
    execute_on = 'linear timestep_end'
    use_displaced_mesh = false
  []
  # MULTIAPP
  [./inlet_mean_temp]
    type = Receiver
    initialize_old = true
    execute_on = 'timestep_begin'
  [../]
[]

[VectorPostprocessors]
  [k_vpp]
    type = Eigenvalues
    inverse_eigenvalue = false
  []
  [centerline_flux]
    type = LineValueSampler
    variable = 'group1 group2'
    start_point = '0 0 0'
    end_point = '0 150 0'
    num_points = 151
    sort_by = y
    execute_on = FINAL
  []
  [midplane_flux]
    type = LineValueSampler
    variable = 'group1 group2'
    start_point = '0 75 0'
    end_point = '69.375 75 0'
    num_points = 100
    sort_by = x
    execute_on = FINAL
  []
[]

[Outputs]
  [exodus]
    type = Exodus
  []
[]