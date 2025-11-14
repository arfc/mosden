import os
import numpy as np
unprocessed_data_dir = os.path.join(os.path.dirname(__file__), '../data/unprocessed')
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
        "parent_feeding": False,
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
        "sensitivity_subplots": True
    }
}