import numpy as np
import argparse
import torch

#######################################
### Parser utils
#######################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def stats_parser():
    parser = argparse.ArgumentParser(description='Robust Solving Statistics')

    parser.add_argument('--stats_name',     type=str,       required=True,                 help='name of model to compute statistics')
    parser.add_argument('--stats_dir',      type=str,       default='./preprocess',      help='directory to save statistics')
    
    parser.add_argument('--lbs_name',       type=str,       default='LBS_test.txt',      help='name of LBS file')
    parser.add_argument('--lbs_dir',        type=str,       default='./data',      help='directory of LBS file')
 
    return parser.parse_args()



def train_parser():
    parser = argparse.ArgumentParser(description='Robust Solving ResNet Training')

    parser.add_argument('--load_model_name',     type=str,       required=True,          help='model name to load')
    parser.add_argument('--load_model_dir',      type=str,       default='./weights/walk',    help='model load directory')
    
    parser.add_argument('--save_model_name',     type=str,       default='none',          help='model name to save')
    parser.add_argument('--save_model_dir',      type=str,       default='./weights/walk',    help='model save directory')
    
    parser.add_argument('--stats_name',         type=str,       default='',      help='name of statistics file')
    parser.add_argument('--stats_dir',          type=str,       default='./preprocess',      help='directory to statistics files')
    
    parser.add_argument('--tensorboard_dir',    type=str,       default='./tensorboard',      help='directory to save tensorboard')
    
    parser.add_argument('--lbs_name',           type=str,       default='LBS_test.txt',      help='name of LBS file')
    parser.add_argument('--lbs_dir',            type=str,       default='./data',      help='directory of LBS file')
 
    parser.add_argument('--batch_size',         type=int,       default=128,              help='batch size')
    parser.add_argument('--resume',             type=int,       default=0,            help='number of trained epoch, resume from this number')
    parser.add_argument('--epoch',              type=int,       default=50,            help='number of training epoch')
    
    parser.add_argument('--lr',                 type=float,     default=1e-6,           help='learning rate')
    parser.add_argument('--optimizer',          type=str,       default='amsgrad'  ,          choices=['amsgrad', 'adam', 'sgd'],     help='optimizer')
    parser.add_argument('--weight_decay',       type=float,     default=0,              help='weight decay')
    parser.add_argument('--corrupt_markers',    type = str2bool, default = 'FALSE',    help='whether to corrupt markers, True/False')
    parser.add_argument('--F_noise',            type=str2bool,  default='FALSE',        help='whether to add noise to local reference frame, True/False')

    parser.add_argument('--num_workers',        type=int,       default=2,              help='number of workers for DataLoader')

    return parser.parse_args()



def infer_parser():
    parser = argparse.ArgumentParser(description='Robust Solving ResNet Testing')

    parser.add_argument('--load_model_name',     type=str,       required=True,                 help='model to test')
    parser.add_argument('--load_model_dir',      type=str,       default='./weights/walk',      help='model save directory')

    parser.add_argument('--stats_name',      type=str,       default='',      help='name of statistics file')
    parser.add_argument('--stats_dir',      type=str,       default='./preprocess',      help='directory to statistics files')

    parser.add_argument('--lbs_name',      type=str,       default='LBS_test.txt',      help='name of LBS file')
    parser.add_argument('--lbs_dir',      type=str,       default='./data',      help='directory of LBS file')

    parser.add_argument('--batch_size',         type=int,       default=128,              help='batch size')
    parser.add_argument('--corrupt_markers', type = str2bool, default = 'FALSE')
    parser.add_argument('--num_workers',    type=int,       default=2,              help='number of workers for DataLoader')
    
    ### visualize options
    parser.add_argument('--show_gt_joints',       type = str2bool,  default = 'TRUE')
    parser.add_argument('--show_input_markers',   type = str2bool,  default = 'FALSE')

    return parser.parse_args()


def export_parser():
    parser = argparse.ArgumentParser(description='Robust Solving Model & Test Input Exporting')

    parser.add_argument('--load_model_name',     type=str,       required=True,                 help='model to test')
    parser.add_argument('--load_model_dir',      type=str,       default='./weights/walk',      help='model save directory')
    
    parser.add_argument('--export_model_dir',      type=str,       default='./export',      help='model export directory')
    parser.add_argument('--export_input_dir',      type=str,       default='./export',      help='model export directory')

    parser.add_argument('--stats_name',      type=str,       default='test',      help='name of statistics file')
    parser.add_argument('--stats_dir',      type=str,       default='./preprocess',      help='directory to statistics files')

    parser.add_argument('--lbs_name',      type=str,       default='LBS_test.txt',      help='name of LBS file')
    parser.add_argument('--lbs_dir',      type=str,       default='./data',      help='directory of LBS file')

    parser.add_argument('--batch_size',         type=int,       default=128,              help='batch size')
    parser.add_argument('--corrupt_markers', type = str2bool, default = 'FALSE')
    parser.add_argument('--num_workers',    type=int,       default=2,              help='number of workers')
    
    return parser.parse_args()



#######################################
### Statistics utils
#######################################
def loadStatistics(path):
    '''
    Load statistics from txt file
    
    Parameters:
        path        -- path to statistics text file (string)
    
    Returns:
        nof     -- number of frames
        noj     -- number of joints
        nom     -- number of markers
        x_mean  -- mean of marker position X                    (nom, 3)
        x_std   -- standard deviation of marker position X      (nom, 3)
        y_mean  -- mean of joint transform Y                    (noj, 3, 4)
        y_std   -- standard deviation of joint transform Y      (noj, 3, 4)
        z_mean  -- mean of marker configuration Z               (nom, noj, 3)
        z_std   -- standard deviation of marker configuration Z (nom, noj, 3)
        zHat_mean   -- mean of pre-weighted local offset                (nom, 3)
        zHat_std    -- standard deviation of pre-weighted local offset  (nom, 3)
    '''

    f = open(path, 'r')

    line = f.readline() # nof
    nof = int(line)

    line = f.readline() # noj
    noj = int(line)

    line = f.readline() # nom
    nom = int(line)

    f.readline()
    f.readline()    # x_mean
    x_mean = np.loadtxt(f, max_rows=nom)  # (nom, 3)

    f.readline()
    f.readline()    # x_std
    x_std = np.loadtxt(f, max_rows=nom)  # (nom, 3)

    f.readline()
    f.readline()    # y_mean
    y_mean = np.loadtxt(f, max_rows=noj*3*4) 

    f.readline()
    f.readline()    # y_std
    y_std = np.loadtxt(f, max_rows=noj*3*4)
    
    f.readline()
    f.readline()    # z_mean
    z_mean = np.loadtxt(f, max_rows=nom)
    z_mean = np.reshape(z_mean, (nom, noj, 3))
    
    f.readline()
    f.readline()    # z_std
    z_std = np.loadtxt(f, max_rows=nom)
    z_std = np.reshape(z_std, (nom, noj, 3))

    f.readline()
    f.readline()    # zHat_mean
    zHat_mean = np.loadtxt(f, max_rows=nom)

    f.readline()
    f.readline()    # zHat_std
    zHat_std = np.loadtxt(f, max_rows=nom)

    f.close()

    # print(nof, noj, nom, x_mean.shape, x_std.shape, y_mean.shape, y_std.shape, z_mean.shape, z_std.shape, zHat_mean.shape, zHat_std.shape)

    return nof, noj, nom, x_mean, x_std, y_mean, y_std, z_mean, z_std, zHat_mean, zHat_std



def saveStatistics(nof, noj, nom, x_mean, x_std, y_mean, y_std, z_mean, z_std, zHat_mean, zHat_std, path):
    '''
    Save statistics to txt file
    
    Parameters:
        nof     -- number of frames
        noj     -- number of joints
        nom     -- number of markers
        x_mean  -- mean of marker position X                            (nom, 3)
        x_std   -- standard deviation of marker position X              (nom, 3)
        y_mean  -- mean of joint transform Y                            (noj, 3, 4)
        y_std   -- standard deviation of joint transform Y              (noj, 3, 4)
        z_mean  -- mean of marker configuration Z                       (nom, noj, 3)
        z_std   -- standard deviation of marker configuration Z         (nom, noj, 3)
        zHat_mean   -- mean of pre-weighted local offset                (nom, 3)
        zHat_std    -- standard deviation of pre-weighted local offset  (nom, 3)
        path        -- path to save statistics, including file name
    
    Returns:
        None
    '''

    f = open(path, 'w')

    # Write nof, noj, nom
    f.write(str(nof) + "\n" + str(noj) + "\n" + str(nom) + "\n")

    # Write statistics
    f.write('\nx_mean\n')
    np.savetxt(f, x_mean, fmt='%.8f')

    f.write('\nx_std\n')
    np.savetxt(f, x_std, fmt='%.8f')

    f.write('\ny_mean\n')                  
    np.savetxt(f, y_mean, fmt='%.8f')

    f.write('\ny_std\n')                   
    np.savetxt(f, y_std, fmt='%.8f')

    f.write('\nz_mean\n')                 
    np.savetxt(f, z_mean, fmt='%.8f')

    f.write('\nz_std\n')                 
    np.savetxt(f, z_std, fmt='%.8f')

    f.write('\nzHat_mean\n')
    np.savetxt(f, zHat_mean, fmt='%.8f')

    f.write('\nzHat_std\n')
    np.savetxt(f, zHat_std, fmt='%.8f')

    f.close()



def getLocalMarkerPos(f_marker_gpos, f_F):
    '''
    for a single frame,
    Transform markers into local reference frame 
    
    Parameters:
        f_marker_gpos     -- global marker position in every frame    (nom, 4)
        f_F               -- local reference frame F for this frame   (4, 4)
    
    Returns:
        f_marker_lpos     -- marker position relative to rigid body   (nom, 4)
    '''

    nom = np.size(f_marker_gpos, axis=0)

    f_marker_lpos = np.zeros((nom, 4))
    for m in range(nom):
        pos = f_marker_gpos[m]
        pos = np.matmul(np.linalg.inv(f_F), pos)
        f_marker_lpos[m] = pos

    return f_marker_lpos



def getLocalJntTrf(f_jnttrf, f_F):
    '''
    for a single frame,
    Transform markers into local reference frame 

    Parameters:
        f_jnttrf     -- global joint transformation     (noj, 4, 4)
        f_F          -- local reference frame F         (4, 4)

    Returns:
        f_jnttrf_local     -- marker position relative to rigid body   (noj, 4, 4)
    '''

    noj = np.size(f_jnttrf, axis=0)

    f_jnttrf_local = np.zeros((noj, 4, 4))
    for j in range(noj):
        trf = f_jnttrf[j]
        trf = np.matmul(np.linalg.inv(f_F), trf)
        f_jnttrf_local[j] = trf

    return f_jnttrf_local








#######################################
### Model utils
#######################################
def toTensor(arr, device):
    '''
    Convert numpy array to Tensor
    '''
    arr = torch.from_numpy(arr).float().to(device)
    return arr


def corruptMarkers(vpos, sigO, sigS, beta):
    '''
    Corrupt global marker position to emulate marker occlusion, swap, and noise
    
    Parameters:
        vpos    -- current single Marker's position (4, 1)
        sigO    -- parameter to sample occlusion probability
        sigS    -- parameter to sample shifting probability
        beta    -- magnitude for shifting
    
    Returns:
        corrupt_pos     -- corrupted marker position. same shape as vpos
    '''

    # Sample probability at which to occlude / shift markers
    alphaO = np.random.normal(0, sigO)
    alphaS = np.random.normal(0, sigS)

    # Sample using clipped probabilities if markers are occluded / shifted
    probO = min(abs(alphaO), 2*sigO)
    probS = min(abs(alphaS), 2*sigS)

    occluded = np.random.binomial(1, probO)
    shifted = np.random.binomial(1, probS)

    # m = copy.deepcopy(marker)
    corrupt_pos = np.copy(vpos)

    # Place occluded markers at zero  
    if occluded == 1:
        corrupt_pos = np.zeros((4,1))

    # Move shifted markers
    elif shifted == 1:
        # Sample the magnitude by which to shift each marker
        magnitude = np.random.uniform(-1*beta, beta)
        for i in range(3):
            corrupt_pos[i] = corrupt_pos[i] + magnitude

    return corrupt_pos


def getGlobalMarkerPosition(marker, marker_config, joint_trf):
    '''
    Get global marker position through Linear Blend Skinning
    
    Parameters:
        marker          -- single Marker
        marker_config   -- single marker configuration for this frame   (noj, 3)
        joint_trf       -- joint transform Y                            (noj, 4, 4)
    
    Returns:
        lbs_pos         -- position of the marker computed with LBS (4,)
    '''

    ### LBS position for a marker
    lbs_pos = np.zeros((4,))

    ### Sum weights of effective joints
    for i in range(len(marker.indices)):
        index = marker.indices[i]
        localOffset = np.append(marker_config[index], 1)     # make marker_config homogeneous
        jntTrf = joint_trf[index]                               # transformation matrix of a joint
        lbs_pos += np.matmul(jntTrf, localOffset.T) * marker.weights[i]

    return lbs_pos


def get_loss(outputs, target, noj, device):
    '''
    Compute weighted l1 norm for loss function

    Parameters:
        outputs         -- output tensor of the network         (batch_size, noj*3*4)
        target          -- target tensor                        (batch_size, noj*3*4)

    Returns:
        loss            -- weighted loss        (Tensor, gpu)
    '''

    # get batch_size
    size = list(target.size())
    batch_size = size[0]

    # (4, 780)
    diff = torch.abs(outputs - target)
    diff = torch.reshape(diff, (batch_size, noj, 3, 4))

    # multiply user weight
    user_weights = np.ones((noj, 3, 4)) * 0.01
    
    # apply larger weights for torso, arms, legs
    user_weights[0:11] = user_weights[0:11] * 10
    user_weights[31:35] = user_weights[31:35] * 10
    user_weights[55:58] = user_weights[55:58] * 10
    user_weights[60:63] = user_weights[60:63] * 10

    # apply larger weights for hands & feet (positional only)
    # hands
    user_weights[10,:,3] = user_weights[10,:,3] * 2
    user_weights[34,:,3] = user_weights[34,:,3] * 2
    # feet
    user_weights[56,:,3] = user_weights[56,:,3] * 2
    user_weights[62,:,3] = user_weights[62,:,3] * 2

    user_weights = toTensor(user_weights, device)
    weighted_diff = diff * user_weights
    
    loss = torch.sum(weighted_diff)
    return loss







