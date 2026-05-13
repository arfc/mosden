flow_velocity=21.7 # cm/s. See MSRE-properties.ods
ini_temp=922

[GlobalParams]
  num_groups = 2
  num_precursor_groups = 6
  group_fluxes = 'group1 group2'
  temperature = temp
  sss2_input = false
  # pre_concs = 'pre1 pre2 pre3 pre4 pre5 pre6'
  # account_delayed = true
  base_file = 'xsdata.json'
[]

[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 600
  xmax = 500
  elem_type = EDGE2
[../]

[Variables]
  [temp]
    initial_condition = ${ini_temp}
    scaling = 1e-4
    family = MONOMIAL
    order = CONSTANT
  []
[]


[AuxVariables]
  [./group1]
    order = FIRST
    family = LAGRANGE
    initial_condition = 0
  [../]
  [./group2]
    order = FIRST
    family = LAGRANGE
    initial_condition = 0
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


[DiracKernels]
  [./heat_exchanger]
    type = DiracHX
    variable = temp
    power = 5e2 # see controls 4e3
    point = '250 0 0'
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
  [./fuel]
    type = MoltresJsonMaterial
    interp_type = LINEAR
    block = '0'
    material_key = 'fuel'
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
  #perf_graph = true
  #print_linear_residuals = true
  [./exodus]
    type = Exodus
    file_base = 'sub'
    execute_on = 'timestep_begin'
  [../]
[]
