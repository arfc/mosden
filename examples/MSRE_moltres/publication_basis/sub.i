flow_velocity=18.06 # cm/s. See MSRE-properties.ods
ini_temp = 922

[GlobalParams]
  num_groups = 4
  num_precursor_groups = 6
  group_fluxes = '0 0 0 0'
  temperature = temp
  sss2_input = true
  # pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6'
  # account_delayed = true
[]

[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 600
  xmax = 288.96
  elem_type = EDGE2
[../]

[Variables]
  [./temp]
    initial_condition = ${ini_temp}
    scaling = 1e-4
    family = MONOMIAL
    order = CONSTANT
  [../]
[]

[Precursors]
 [./core]
  var_name_base = pre
  outlet_boundaries = 'right'
  u_def = ${flow_velocity}
  v_def = 0
  w_def = 0
  nt_exp_form = false
  family = MONOMIAL
  order = CONSTANT
  loop_precursors = true
  multi_app = loopApp
  is_loopapp = true
  inlet_boundaries = left
 [../]
[]

[Kernels]
  # Temperature
  [./temp_time_derivative]
    type = MatINSTemperatureTimeDerivative
    variable = temp
  [../]
  # [./temp_source_fuel]
  #   type = TransientFissionHeatSource
  #   variable = temp
  #   nt_scale=${nt_scale}
  # [../]
  # [./temp_source_mod]
  #   type = GammaHeatSource
  #   variable = temp
  #   gamma = .0144 # Cammi .0144
  #   block = 'moder'
  #   average_fission_heat = 'average_fission_heat'
  # [../]
  # [./temp_diffusion]
  #   type = MatDiffusion
  #   diffusivity = 'k'
  #   variable = temp
  # [../]
  # [./temp_advection_fuel]
  #   type = ConservativeTemperatureAdvection
  #   velocity = '${flow_velocity} 0 0'
  #   variable = temp
  # [../]
[]

[DGKernels]
  [./temp_adv]
    type = DGTemperatureAdvection
    variable = temp
    velocity = '${flow_velocity} 0 0'
  [../]
[]



[BCs]
  [./fuel_bottoms_looped]
    boundary = 'left'
    type = PostprocessorTemperatureInflowBC
    postprocessor = coreEndTemp
    variable = temp
    uu = ${flow_velocity}
  [../]
  # [./diri]
  #   boundary = 'left'
  #   type = DirichletBC
  #   variable = temp
  #   value = 930
  # [../]
  [./temp_advection_outlet]
    boundary = 'right'
    type = TemperatureOutflowBC
    variable = temp
    velocity = '${flow_velocity} 0 0'
  [../]
[]

[Materials]
  [fuel]
    type = GenericMoltresMaterial
    property_tables_root = './data/msre_gentry_4g_fuel_rod0_'
    interp_type = 'spline'
    block = '0'
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
    block = '0'
  []
[]

[Executioner]
  type = Transient
  end_time = 10000

  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6

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
  [./temp_fuel]
    type = ElementAverageValue
    variable = temp
    outputs = 'exodus console'
  [../]
  [./loopEndTemp]
    type = SideAverageValue
    variable = temp
    boundary = 'right'
  [../]
  [./coreEndTemp]
    type = Receiver
  [../]
[]

[Outputs]
  perf_graph = true
  print_linear_residuals = true
  [./exodus]
    type = Exodus
    file_base = 'sub'
    execute_on = 'timestep_begin'
  [../]
[]

[Debug]
  show_var_residual_norms = true
[]

# connect inlet and outlet to multiapp
# [Transfers]
#   [./to_core]
#     type = MultiAppPostprocessorTransfer
#     multi_app = MoltresApp
#     from_postprocessor = loopEndTemp
#     to_postprocessor = inlet_mean_temp
#     direction = to_multiapp
#   [../]
#   [./from_core]
#     type = MultiAppPostprocessorTransfer
#     multi_app = MoltresApp
#     from_postprocessor = coreEndTemp
#     to_postprocessor = coreEndTemp
#     direction = to_multiapp
#   [../]
# []
