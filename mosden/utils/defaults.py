import os
import numpy as np
unprocessed_data_dir = os.path.join(os.path.dirname(__file__), '../data/unprocessed')
literature_data_dir = os.path.join(os.path.dirname(__file__), '../data/literature_groups')
reprocessing_data_dir = os.path.join(os.path.dirname(__file__), '../data/chemical_rates')
mosden_dir = os.path.dirname(__file__)
current_dir = os.getcwd()
seed = np.random.randint(0, 2**32 - 1)
DEFAULTS = {
    "name": "default",
    "multi_id": None,
    "file_options": {
        "overwrite": {
            "preprocessing": True,
            "concentrations": True,
            "count_rate": True,
            "group_fitting": True,
            "postprocessing": True,
            "logger": False
        },
        "unprocessed_data_dir": f"{unprocessed_data_dir}",
        "literature_data_dir": f"{literature_data_dir}",
        "reprocessing_data_dir": f"{reprocessing_data_dir}",
        "processed_data_dir": f"{current_dir}/",
        "output_dir": f"{current_dir}/",
        "log_level": 20,
        "log_file": f"{current_dir}/log.log"
    },
    "data_options": {
        "half_life": "iaea/eval.csv",
        "cross_section": "",
        "emission_probability": "iaea/eval.csv",
        "fission_yield": "endfb71/nfy/",
        "decay_time_spacing": "log",
        "temperature_K": 920,
        "density_g_cm3": 2.3275,
        "energy_MeV": 0.0253e-6,
        "fissile_fractions": {
            "U235": 1.0
        }
    },
    "modeling_options": {
        "concentration_handling": "CFY",
        "count_rate_handling": "data",
        "reprocessing_locations": [""],
        "spatial_scaling": "scaled",
        "reprocessing": {
            "Xe": 0.0
        },
        "irrad_type": "saturation",
        "incore_s": 10,
        "excore_s": 0,
        "net_irrad_s": 10000,
        "decay_time": 1200,
        "num_decay_times": 800
    },
    "group_options": {
        "num_groups": 6,
        "method": "nlls",
        "samples": 1,
        "sample_func": "normal",
        "seed": seed
    },
    "post_options": {
        "sensitivity_subplots": True,
        "top_num_nuclides": 3,
        "num_stacked_nuclides": 2,
        "lit_data": ['keepin', 'brady', 'synetos'],
        "nuclides": [
            'Br87',
            'I137',
            'Br88',
            'Br89',
            'I138',
            'Rb94',
            'Rb93',
            'Te136',
            'Ge86',
            'As86',
            'Br90',
            'As85'
        ]
    }
}