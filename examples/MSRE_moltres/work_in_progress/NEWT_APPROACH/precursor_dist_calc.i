flow_velocity=21.7 # cm/s. See MSRE-properties.ods
nt_scale=1
ini_temp=922

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

[Mesh]
  coord_type = RZ
  [mesh]
    type = FileMeshGenerator
    file = 'mesh.e'
  []
[]

[Nt]
  var_name_base = group
  vacuum_boundaries = 'fuel_bottom fuel_top mod_bottom mod_top right'
  create_temperature_var = false
  pre_blocks = '0'
[]

[Variables]
  [./temp]
    initial_condition = ${ini_temp}
  [../]
[]

[Precursors]
 [./core]
  var_name_base = pre
  block = '0'
  outlet_boundaries = 'fuel_top'
  u_def = 0
  v_def = ${flow_velocity}
  w_def = 0
  nt_exp_form = false
  family = MONOMIAL
  order = CONSTANT
  loop_precursors = true
  multi_app = loopApp
  is_loopapp = false
  inlet_boundaries = 'fuel_bottom'
 [../]
[]

[Kernels]
  # Temperature
  [./temp_time_derivative]
    type = MatINSTemperatureTimeDerivative
    variable = temp
  [../]
  [./temp_source_fuel]
    type = TransientFissionHeatSource
    variable = temp
    nt_scale=${nt_scale}
    block = '0'
  [../]
  [./temp_diffusion]
    type = MatDiffusion
    diffusivity = 'k'
    variable = temp
  [../]
  [./temp_advection_fuel]
    type = ConservativeTemperatureAdvection
    velocity = '0 ${flow_velocity} 0'
    variable = temp
    block = '0'
  [../]
[]

[BCs]
  [./fuel_bottom_looped]
    boundary = 'fuel_bottom right'
    type = PostprocessorDirichletBC
    postprocessor = inlet_mean_temp
    variable = temp
  [../]
  [./temp_advection_outlet]
    boundary = 'fuel_top'
    type = TemperatureOutflowBC
    variable = temp
    velocity = '0 ${flow_velocity} 0'
  [../]
  #[temp_inlet]
  #  boundary = 'fuel_bottom'
  #  type = DirichletBC
  #  variable = temp
  #  value = ${ini_temp}
  #[]
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
  type = Transient
  end_time = 11000
  compute_scaling_once = false
  automatic_scaling = true
  scaling_group_variables = 'group1 group2; pre1 pre2 pre3 pre4 pre5 pre6; temp'

  nl_rel_tol = 1e-5
  nl_abs_tol = 1e-5

  solve_type = 'PJFNK'
  petsc_options = '-snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor'
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  line_search = 'none'

  nl_max_its = 30
  l_max_its = 100

  dtmin = 1e-5
  # dtmax = 1
  # dt = 1e-3
  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1e-3
    cutback_factor = 0.4
    growth_factor = 1.2
    optimal_iterations = 20
  [../]
[]

[Preconditioning]
  [./SMP]
    type = SMP
    full = true
  [../]
[]

[Postprocessors]
  [tot_fissions]
    type = ElmIntegTotFissPostprocessor
    execute_on = linear
  []
  [powernorm]
    type = ElmIntegTotFissHeatPostprocessor
    execute_on = linear
  []
  [pre1_integral]
    type = ElementIntegralVariablePostprocessor
    variable = pre1
    execute_on = linear
    block = '0'
  []
  [./group1_current]
    type = IntegralNewVariablePostprocessor
    variable = group1
    outputs = 'console exodus'
  [../]
  [./group1_old]
    type = IntegralOldVariablePostprocessor
    variable = group1
    outputs = 'console exodus'
  [../]
  [./multiplication]
    type = DivisionPostprocessor
    value1 = group1_current
    value2 = group1_old
    outputs = 'console exodus'
  [../]
  [./temp_fuel]
    type = ElementAverageValue
    variable = temp
    block = '0'
    outputs = 'exodus console'
  [../]
  [./coreEndTemp]
    type = SideAverageValue
    variable = temp
    boundary = 'fuel_top'
    outputs = 'exodus console'
  [../]
  # MULTIAPP
  [./inlet_mean_temp]
    type = Receiver
    initialize_old = true
    execute_on = 'timestep_begin'
  [../]
[]

[Outputs]
  #perf_graph = true
  #print_linear_residuals = true
  [./exodus]
    type = Exodus
    file_base = 'precursor_dist_calc'
    execute_on = 'timestep_end'
  [../]
[]

#[Debug]
#  show_var_residual_norms = true
#[]

[MultiApps]
  [./loopApp]
    type = TransientMultiApp
    app_type = MoltresApp
    execute_on = timestep_begin
    positions = '100.0 100.0 0.0'
   input_files = 'sub.i'
 [../]
[]

[Transfers]
  [./from_loop]
    type = MultiAppPostprocessorTransfer
    multi_app = loopApp
    from_postprocessor = loopEndTemp
    to_postprocessor = inlet_mean_temp
    direction = from_multiapp
    reduction_type = maximum
  [../]
  [./to_loop]
    type = MultiAppPostprocessorTransfer
    multi_app = loopApp
    from_postprocessor = coreEndTemp
    to_postprocessor = coreEndTemp
    direction = to_multiapp
  [../]
[]