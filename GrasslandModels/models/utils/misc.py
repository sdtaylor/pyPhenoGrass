import json
import os
from warnings import warn


def read_saved_model(model_file):
    with open(model_file, 'r') as f:
        m = json.load(f)
    return m

def write_saved_model(model_info, model_file, overwrite):
    if os.path.exists(model_file) and not overwrite:
        raise RuntimeWarning('File {f} exists. User overwrite=True to overwite'.format(f=model_file))
    else:
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=4)