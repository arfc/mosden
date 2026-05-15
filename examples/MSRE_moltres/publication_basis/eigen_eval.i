[GlobalParams]
  num_groups = 4
  num_precursor_groups = 6
  use_exp_form = false
  group_fluxes = 'group1 group2 group3 group4'
  temperature = temp
  sss2_input = true
  pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6'
  account_delayed = true
[]

[Problem]
  allow_initial_conditions_with_restart = true
[]

[Mesh]
  file = './auto_diff_rho_out.e'
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
    block = 'fuel'
  []
  [pre2]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre2
    initial_from_file_timestep = LATEST
    block = 'fuel'
  []
  [pre3]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre3
    initial_from_file_timestep = LATEST
    block = 'fuel'
  []
  [pre4]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre4
    initial_from_file_timestep = LATEST
    block = 'fuel'
  []
  [pre5]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre5
    initial_from_file_timestep = LATEST
    block = 'fuel'
  []
  [pre6]
    family = MONOMIAL
    order = CONSTANT
    initial_from_file_var = pre6
    initial_from_file_timestep = LATEST
    block = 'fuel'
  []
[]


[Nt]
  var_name_base = group
  vacuum_boundaries = 'fuel_bottoms fuel_tops moder_bottoms moder_tops outer_wall'
  pre_blocks = 'fuel'
  create_temperature_var = false
  eigen = true
[]

[Materials]
  [fuel]
    type = GenericMoltresMaterial
    property_tables_root = './data/msre_gentry_4g_fuel_rod0_'
    interp_type = 'spline'
    block = 'fuel'
    prop_names = 'k cp'
    prop_values = '.0553 1967' # Robertson MSRE technical report @ 922 K
    controller_gain = 0
  []
  [rho_fuel]
    type = DerivativeParsedMaterial
    property_name = rho
    expression = '2.146e-3 * exp(-1.8 * 1.18e-4 * (temp - 922))'
    coupled_variables = 'temp'
    derivative_order = 1
    block = 'fuel'
  []
  [moder]
    type = GenericMoltresMaterial
    property_tables_root = './data/msre_gentry_4g_moder_rod0_'
    interp_type = 'spline'
    prop_names = 'k cp'
    prop_values = '.312 1760' # Cammi 2011 at 908 K
    block = 'moder'
    controller_gain = 0
  []
  [rho_moder]
    type = DerivativeParsedMaterial
    property_name = rho
    expression = '1.86e-3 * exp(-1.8 * 1.0e-5 * (temp - 922))'
    coupled_variables = 'temp'
    derivative_order = 1
    block = 'moder'
  []
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
    block = 'fuel'
  []
  [k_eff]
    type = VectorPostprocessorComponent
    index = 0
    vectorpostprocessor = k_vpp
    vector_name = eigen_values_real
  []
  [bnorm]
    type = ElmIntegTotFissNtsPostprocessor
    block = 'fuel'
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
    variable = 'group1 group2 group3 group4'
    start_point = '0 0 0'
    end_point = '0 150 0'
    num_points = 151
    sort_by = y
    execute_on = FINAL
  []
  [midplane_flux]
    type = LineValueSampler
    variable = 'group1 group2 group3 group4'
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