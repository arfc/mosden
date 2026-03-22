import subprocess
import json
import shutil
import os



if __name__ == "__main__":
    num_groups = [4, 6, 8, 10, 12]
    irrad_times = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    for group in num_groups:
        output_dir = f'./dataNet_{group}'
        input_file = './input.json'
        for T in irrad_times:
            with open(input_file, 'r') as f:
                data = json.load(f)
            eval_method = 'post-irrad'
            data['modeling_options']['residual_handling'] = [eval_method] 
            data['modeling_options']['incore_s'] = T
            data['modeling_options']['net_irrad_s'] = T
            data['group_options']['num_groups'] = group

            with open(input_file, 'w') as f:
                json.dump(data, f, indent=4)

            subprocess.run(['mosden', '-pre', input_file])
            subprocess.run(['mosden', '-m', input_file])
            os.makedirs(output_dir, exist_ok=True)
            shutil.move('./group_parameters.csv', f'{output_dir}/intermediate_{T}_{eval_method}.csv')
            subprocess.run(['mosden', '-g', input_file])

            eval_method = 'all'
            data['modeling_options']['residual_handling'] = [eval_method] 
            shutil.move('./group_parameters.csv', f'{output_dir}/intermediate_{T}_{eval_method}.csv')
