import numpy as np
import os.path
import argparse
import time

### onnx
import onnx
import onnxruntime

### torch lib
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

### custom lib
from preprocess import local_reference, marker_config
from models import resnet_model
from data import lbs_import, configure_dataset 
import utils


def export(opts):
    '''
    export the network to onnx file
    
    Parameters:
        opts
    
    Returns:
        model_export_path   -- path to exported model
        torch_out           -- output of the model with random input (computed with pytorch, later compared with onnx result)
        x                   -- random input to the model
    '''

    ### start of exporting
    start = time.time()

    # path to LBS file
    lbs_path = os.path.join(opts.lbs_dir, opts.lbs_name)

    ### get input data to test
    # motion = lbs_import.readMotion("./data/samba_dance/Motion.txt")
    motion = lbs_import.importMotion('train')
    markers, _ = lbs_import.readLBS(lbs_path)
    
    # number of frames, markers
    noj = np.size(motion, axis=1)
    nom = len(markers)


    ### setup model
    model_save_dir = opts.load_model_dir             # directory to save trained weights
    model_save_file = opts.load_model_name + ".pth"           # file to save trained weights
    model_save_path = os.path.join(model_save_dir, model_save_file)

    export_model_name = opts.load_model_name + ".onnx"
    model_export_path = os.path.join(opts.export_model_dir, export_model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # resnet network
    model = resnet_model.ResNet(resnet_model.ResidualBlock, nom, noj)

    # load trained weights
    if os.path.exists(model_save_dir):
        if os.path.isfile(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print('Loaded trained weights')
    else:
        print('Cannot find trained model')
        exit(0)

    model = model.float()
    model.to(device)
    model.eval()

    # with torch.no_grad():
    x = torch.randn(opts.batch_size, 2*nom*3)
    x = x.float().to(device)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    model_export_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    export_time = time.time() - start
    print('Export done')
    print('Export time %.3f min\n' % (export_time/60))

    return model_export_path, torch_out, x


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_onnx(model_export_path, torch_out, x):
    '''
    In order to check if it is correctly exported,
    run the onnx file with onnx runtime and compare the output with the one computed through pytorch.
    
    Parameters:
        model_export_path   -- path to exported model
        torch_out           -- output of the model with random input (computed with pytorch, later compared with onnx result)
        x                   -- random input to the model
    
    Returns:
        None
    '''
    onnx_model = onnx.load(model_export_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = onnxruntime.InferenceSession(model_export_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    opts = utils.export_parser()
    print(f'\nReceived options:\n{opts}')

    ### export the network
    print('\nExporting the model...')
    model_export_path, torch_out, x = export(opts)

    ### check validation of onnx file
    print('\nChecking onnx file...')
    check_onnx(model_export_path, torch_out, x)
