flow_velocity = 18.06 # cm/s
nt_scale = 1e13
ini_temp = 922
diri_temp = 922
gamma_frac = .075
R = 70.1675
H = 162.56

[GlobalParams]
  num_groups = 4
  num_precursor_groups = 12
  use_exp_form = false
  group_fluxes = 'group1 group2 group3 group4'
  temperature = temp
  sss2_input = true
  #pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6'
  pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6 pre7 pre8 pre9 pre10 pre11 pre12'
  account_delayed = true
  nt_scale = ${nt_scale}
[]

[Problem]
  restart_file_base = 'auto_diff_rho_out_cp/LATEST'
  allow_initial_conditions_with_restart = true
[]

[ICs]
  [pre1_zero]
    type = ConstantIC
    variable = pre1
    value = 0
  []
  [pre2_zero]
    type = ConstantIC
    variable = pre2
    value = 0
  []
  [pre3_zero]
    type = ConstantIC
    variable = pre3
    value = 0
  []
  [pre4_zero]
    type = ConstantIC
    variable = pre4
    value = 0
  []
  [pre5_zero]
    type = ConstantIC
    variable = pre5
    value = 0
  []
  [pre6_zero]
    type = ConstantIC
    variable = pre6
    value = 0
  []
  [pre7_zero]
    type = ConstantIC
    variable = pre7
    value = 0
  []
  [pre8_zero]
    type = ConstantIC
    variable = pre8
    value = 0
  []
  [pre9_zero]
    type = ConstantIC
    variable = pre9
    value = 0
  []
  [pre10_zero]
    type = ConstantIC
    variable = pre10
    value = 0
  []
  [pre11_zero]
    type = ConstantIC
    variable = pre11
    value = 0
  []
  [pre12_zero]
    type = ConstantIC
    variable = pre12
    value = 0
  []
[]

[Mesh]
  coord_type = RZ
  file = '2d_lattice_structured.msh'
[]

[Variables]
  [temp]
    initial_from_file_var = temp
    initial_from_file_timestep = LATEST
    scaling = 1e-4
  []
[]

[AuxVariables]
  [power_density]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[Precursors]
  [pres]
    var_name_base = pre
    block = 'fuel'
    outlet_boundaries = 'fuel_tops'
    u_def = 0
    v_def = ${flow_velocity}
    w_def = 0
    nt_exp_form = false
    loop_precursors = false
    #multi_app = loopApp
    is_loopapp = false
    inlet_boundaries = 'fuel_bottoms'
    family = MONOMIAL
    order = CONSTANT
    # jac_test = true
  []
[]

[Nt]
  var_name_base = group
  vacuum_boundaries = 'fuel_bottoms fuel_tops moder_bottoms moder_tops outer_wall'
  create_temperature_var = false
  scaling = 1e-4
  init_nts_from_file = true
  pre_blocks = 'fuel'
[]

[Kernels]
  # Temperature
  [temp_time_derivative]
    type = MatINSTemperatureTimeDerivative
    variable = temp
  []
  [temp_source_fuel]
    type = TransientFissionHeatSource
    variable = temp
    block = 'fuel'
  []
  [temp_source_mod]
    type = GammaHeatSource
    variable = temp
    block = 'moder'
    average_fission_heat = 'average_fission_heat'
    gamma = gamma_func
  []
  [temp_diffusion]
    type = MatDiffusion
    diffusivity = 'k'
    variable = temp
  []
  [temp_advection_fuel]
    type = ConservativeTemperatureAdvection
    velocity_variable = '0 ${flow_velocity} 0'
    variable = temp
    block = 'fuel'
  []
[]

[BCs]
  [temp_diri_cg]
    boundary = 'fuel_bottoms outer_wall'
    type = FlexiblePostprocessorDirichletBC
    postprocessor = coreEndTemp
    offset = -27.8
    variable = temp
  []
  # [./temp_diri_cg]
  #   boundary = 'moder_bottoms fuel_bottoms outer_wall'
  #   type = FunctionDirichletBC
  #   function = 'temp_bc_func'
  #   variable = temp
  # [../]
  [temp_advection_outlet]
    boundary = 'fuel_tops'
    type = TemperatureOutflowBC
    variable = temp
    velocity = '0 ${flow_velocity} 0'
  []
[]

[AuxKernels]
  [fuel]
    block = 'fuel'
    type = FissionHeatSourceTransientAux
    variable = power_density
  []
  [moderator]
    block = 'moder'
    type = ModeratorHeatSourceTransientAux
    average_fission_heat = 'average_fission_heat'
    variable = power_density
    gamma = gamma_func
  []
[]

[Functions]
  [temp_bc_func]
    type = ParsedFunction
    expression = '${ini_temp} - (${ini_temp} - ${diri_temp}) * tanh(t/1e-2)'
  []
  [gamma_func]
    type = ParsedFunction
    expression = '${gamma_frac} * pi^2 / 4 * cos(pi * x / (2. * ${R})) * sin(pi * y / ${H})'
  []
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
  type = Transient
  num_steps = 1
  dt = 1e-6

  nl_rel_tol = 1e-6
  nl_abs_tol = 6e-6

  solve_type = 'PJFNK'
  line_search = none
  petsc_options = '-snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor'
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  nl_max_its = 30
  l_max_its = 100

[]

[Preconditioning]
  [SMP]
    type = SMP
    full = true
  []
[]

[Postprocessors]
  [group1_current]
    type = IntegralNewVariablePostprocessor
    variable = group1
    outputs = 'console csv'
  []
  [group1_old]
    type = IntegralOldVariablePostprocessor
    variable = group1
    outputs = 'console csv'
  []
  [multiplication]
    type = DivisionPostprocessor
    value1 = group1_current
    value2 = group1_old
    outputs = 'console csv'
  []
  [temp_fuel]
    type = ElementAverageValue
    variable = temp
    block = 'fuel'
    outputs = 'csv console'
  []
  [temp_moder]
    type = ElementAverageValue
    variable = temp
    block = 'moder'
    outputs = 'csv console'
  []
  [average_fission_heat]
    type = AverageFissionHeat
    execute_on = 'linear nonlinear'
    outputs = 'csv console'
    block = 'fuel'
  []
  [coreEndTemp]
    type = SideAverageValue
    variable = temp
    boundary = 'fuel_tops'
    outputs = 'csv console'
    execute_on = 'linear nonlinear'
  []
  [limit_k]
    type = LimitK
    execute_on = 'timestep_end'
    k_postprocessor = multiplication
    growth_factor = 1.2
    cutback_factor = .4
    k_threshold = 1.5
  []
[]

[Outputs]
  perf_graph = true
  print_linear_residuals = true
  csv = true
  exodus = true
[]

[Debug]
  show_var_residual_norms = true
[]

#[MultiApps]
#  [./loopApp]
#    type = TransientMultiApp
#    app_type = MoltresApp
#    execute_on = timestep_begin
#    positions = '100.0 100.0 0.0'
#   input_files = 'sub.i'
# [../]
#[]