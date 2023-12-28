import numpy as np
import os.path
import argparse
import time

### torch lib
import torch
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


def export_input(opts):
    '''
    test the network
    
    Parameters:
        opts
    
    Returns:
        motion
        input_marker_position
        result_joint
    '''
    # paths to save rigid body, local reference frame, statistics
    rigidbody_name = "rigidbody_" + opts.stats_name + ".txt"
    localref_name = "localref_" + opts.stats_name + ".txt"
    statistics_name = "statistics_" + opts.stats_name + ".txt"

    rigidbody_path = os.path.join(opts.stats_dir, rigidbody_name)
    localref_path = os.path.join(opts.stats_dir, localref_name)
    statistics_path = os.path.join(opts.stats_dir, statistics_name)

    # path to LBS file
    lbs_path = os.path.join(opts.lbs_dir, opts.lbs_name)

    ### start of testing
    start = time.time()

    ### get input data to test
    # get global marker position
    motion = lbs_import.readMotion("./data/samba_dance/Motion.txt")
    # motion = lbs_import.importMotion('test')
    markers, offsets = lbs_import.readLBS(lbs_path)
    
    # (nof, nom, 4)
    marker_gpos = lbs_import.getMarkerGpos(motion, offsets, markers)

    # number of frames, joints, markers
    nof = np.size(motion, axis=0)
    noj = np.size(motion, axis=1)
    nom = len(markers)

    # path to save test input
    test_marker_name = opts.load_model_name + "_marker.txt"
    test_input_name = opts.load_model_name + "_input.txt"
    test_localref_name = opts.load_model_name + "_localref.txt"
    test_ystats_name = opts.load_model_name + "_ystats.txt"

    test_marker_path = os.path.join(opts.export_input_dir, test_marker_name)
    test_input_path = os.path.join(opts.export_input_dir, test_input_name)
    test_localref_path = os.path.join(opts.export_input_dir, test_localref_name)
    test_ystats_path = os.path.join(opts.export_input_dir, test_ystats_name)

    marker_file = open(test_marker_path, 'w')
    input_file = open(test_input_path, 'w')
    localref_file = open(test_localref_path, 'w')
    stats_file = open(test_ystats_path, 'w')

    marker_file.write(str(nof)+'\n'+str(nom)+'\n'+str(4)+'\n')
    input_file.write(str(nof)+'\n'+str(nom*2*3)+'\n')
    localref_file.write(str(nof)+'\n'+str(4)+'\n'+str(4)+'\n')
    stats_file.write(str(noj*3*4)+'\n')

    ### Get marker configuration zHat
    # compute local offset and get weighted sum
    marker_offset = np.empty((nom, 4)) 
    for m in range(nom):
        marker = markers[m]
        pos = np.zeros((4, 1))
        
        ### get number of effective joints
        jnum = np.size(marker.indices, 0)

        for i in range(jnum):
            ### joint which affects this marker's position
            index = marker.indices[i]
            weight = marker.weights[i]
            
            ### skinning offset matrix, and joint transform of this joint
            offsetMat = offsets[index]

            ### weighted sum
            pos += weight * np.matmul(offsetMat, marker.vpos)
        
        marker_offset[m] = pos.T

    # zHat is constant throughout frames for test
    zHat = np.array([marker_offset for i in range(nof)])
    zHat = np.delete(zHat, 3, axis=2)
    

    ### setup dataset
    joint_transform = np.array(motion) 
    local_reference_frame = local_reference.loadLocalReferenceFrame(localref_path)
    marker_dataset = configure_dataset.MarkerDataset(joint_transform, local_reference_frame)
    

    ### load statistics
    _, _, _, x_mean, x_std, y_mean, y_std, _, _, zHat_mean, zHat_std = utils.loadStatistics(statistics_path)
    stats_file.write("y_mean\n")
    np.savetxt(stats_file, y_mean, fmt='%.8f')

    stats_file.write('y_std\n')                   
    np.savetxt(stats_file, y_std, fmt='%.8f')

    stats_file.close()


    ### setup dataloader, network
    testloader = torch.utils.data.DataLoader(marker_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    

    ### test the model
    input_marker_position = np.zeros((nof, nom, 4))
    network_input = np.zeros((nof, 2*nom*3))
    local_reference_frame = np.zeros((nof, 4, 4))
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # running_loss = 0.0
            frame_idx = data['frame_idx'].numpy()
            b_joint = np.copy(data['joint_transform'].numpy())       # b_joint (batch_size, noj, 4, 4)

            batch_size = np.size(frame_idx, axis=0)

            # first input of the NN :
            # homogeneous marker position X
            X = np.zeros((batch_size, nom, 4))
            F = np.zeros((batch_size, 4, 4))

            # for each pose(frame)
            for j in range(batch_size):
                # frame index of j-th sample
                index = frame_idx[j]

                # global marker position for each frame
                frame_pos = np.zeros((nom, 4))
                for m in range(nom):
                    pos = marker_gpos[index][m]

                    if opts.corrupt_markers:
                        # Corrupt marker position
                        pos = utils.corruptMarkers(pos, 0.1, 0.1, 1)
                    
                    frame_pos[m] = pos.T

                # save marker positions for visualization
                input_marker_position[index] = frame_pos


                # compute F for this frame
                F[j] = local_reference.getLocalReferenceFrame(frame_pos, rigidbody_path)    
                local_reference_frame[index] = F[j]

                # Transform ground truth joint transforms into local space
                b_joint[j] = np.matmul(np.linalg.inv(F[j]), b_joint[j])

                # transform marker positions into the local reference
                for m in range(nom):
                    frame_pos[m] = np.matmul(np.linalg.inv(F[j]), frame_pos[m])
                
                X[j] = frame_pos


            # Normalize data
            X = np.delete(X, 3, axis=2)         # (batch_size, nom, 3)
            norm_X = (X - x_mean) / x_std       # x_mean, x_std (nom, 3)
            
            b_zHat = zHat[frame_idx, :, :]
            norm_Z = (b_zHat - zHat_mean) / zHat_std
                
            
            # Concatenate data
            inputs = np.stack((norm_X, norm_Z), axis=1)    # (batch_size, 2, m, 3)
            inputs = np.reshape(inputs, (batch_size,-1))
            network_input[i*opts.batch_size:i*opts.batch_size + batch_size, :] = inputs
            
        
        # write marker position in text file
        marker_file.write("marker positions\n")
        input_marker_position = np.reshape(input_marker_position, (nof, -1))
        np.savetxt(marker_file, input_marker_position, fmt='%.8f')
        marker_file.close()

        # write output joint transform in text file
        input_file.write("network_input\n")
        np.savetxt(input_file, network_input, fmt='%.8f')
        input_file.close() 

        # write local reference frame in text file
        localref_file.write("local reference frame\n")
        local_reference_frame = np.reshape(local_reference_frame, (nof, -1))
        np.savetxt(localref_file, local_reference_frame, fmt='%.8f')
        localref_file.close()           


    export_time = time.time() - start
    print('Export done')
    print('Export time %.3f min\n' % (export_time/60))



if __name__ == '__main__':
    opts = utils.export_parser()
    print(f'\nReceived options:\n{opts}')

    ### test the network
    print('\nExporting test input...')
    export_input(opts)
