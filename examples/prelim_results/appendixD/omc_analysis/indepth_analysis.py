from mosden.utils.input_handler import InputHandler
from mosden.base import BaseClass
import json


#def analysis_solver()

def modify_write_input(input_file, irrad, data_val):
    base_input = InputHandler(input_file).read_input()

    if irrad == 'pulse':
        base_input['modeling_options']['incore_s'] = 0.025
        base_input['modeling_options']['net_irrad_s'] = 0.025
    elif irrad == 'saturation':
        base_input['modeling_options']['incore_s'] = 1e5
        base_input['modeling_options']['incore_s'] = 1e6
    elif irrad == 'res-combined':
        # Instead of separating pulse and saturation, put the data together
        # and solve all at once.
        # For this, run BOTH pulse and saturation, record the data, then have 
        # a function here to manually solve for the best fit
        raise NotImplementedError
    elif irrad == 'long-short':
        # Run BOTH pulse and saturation, then take the longest-lived from
        # saturation and shortest-lived from pulse. (Keepin approach)
        raise NotImplementedError
     
    base_input['modeling_options']['residual_handling'] = data_val
    with open(f'./input_{irrad}_{data_val}.json', 'w') as f:
        json.dump(base_input, f, indent=4)
    return



if __name__ == '__main__':
    input_file = './input_analysis.json'
    irrad_selection = 0
    data_selection = 0
    irrad_types = ['pulse', 'saturation', 'res-combined', 'long-short']
    data_types = ['post-irrad', 'all', 'incore']

    original_input = InputHandler(input_file).read_input()

    base = BaseClass(input_file)
    irrad = irrad_types[irrad_selection]
    data_val = data_types[data_selection]

    modify_write_input(input_file, irrad, data_val)



