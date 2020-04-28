

import argparse 
import glob
import os 
import GraphicCodeColection.fileprocess as fproc

# #init with parser.
# parser = argparse.ArgumentParser()
# parser.add_argument("--target_input_parent", required=True, help="./some_dir/")
# parser.add_argument("--target_output_parent", required=True, help="./sub_dir/")
# arg = parser.parse_args()

# _target_input_dir = arg.target_input_parent
# _target_output_dir = arg.target_output_parent

_target_input_dir = "D:/lab/dataset/unprocessed_dataset"
_target_output_dir =  "./processed_dataset"
_target_template_path = "./data/template.obj"

dataset_dir_naming = ["plain", 'sparse', 'hole', 'downsampled']
flag = True

def is_satisfied(input_dir_list, required_data_names):
    if len(input_dir_list) != 3 : 
        return False

    for name in input_dir_list:
        name=os.path.split(name)[-1]
        if name in required_data_names:
            continue
        else :
            return False
    return True  

if  __name__ == "__main__":

    if not os.path.exists(_target_input_dir):
        pass
    else : 
        target_child_dir = glob.glob(os.path.join(_target_input_dir, "*"))
        target_child_dir = [d for d in target_child_dir if os.path.isdir(d)]
        child_name = [os.path.split(c)[-1] for c in target_child_dir]
        output_child_dir = [os.path.join(_target_output_dir, o) for o in child_name ]

        if not is_satisfied(target_child_dir, dataset_dir_naming):
            for name in target_child_dir:
                if os.path.split(name)[-1] == 'plain':
                    
                    output_prefix_name = os.path.split(name)[0]
                    if 'sparse' not in [os.path.split(t)[-1] for t in target_child_dir] : 
                        sparse_noise = fproc.NoiseProcessor([name],[os.path.join(output_prefix_name, 'sparse')], noise_type='sparse')
                        sparse_noise.save_noise_dataset() # save noise
                    
                    if 'downsampled' not in [os.path.split(t)[-1] for t in target_child_dir] : 
                        downsampled = fproc.DownSampleProcessor([name], [os.path.join(output_prefix_name, 'downsampled')], _target_template_path)
                        downsampled.downsample()
                        break
        print(output_child_dir, target_child_dir)
        prproc = fproc.DataPreprocessor(target_child_dir, output_child_dir)
        prproc.convert_mesh2numpy_expwise()

        



