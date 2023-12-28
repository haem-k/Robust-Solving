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
from data import configure_dataset, lbs_import
import utils


def prepare_input(b_F, b_joint, markers, z_mean, z_std):

    '''
    Get marker position X and marker configuration Z (input for neural network)

    Parameters:
        b_F         -- local reference frame for this batch             (batch_size, 4, 4)
        b_joint     -- joint transformation for this batch              (batch_size, noj, 4, 4)
        markers     -- motion info read from Motion.txt                 (nof, noj, 4, 4)
        z_mean      -- mean of marker configuration Z                   (nom, noj, 3)
        z_std       -- standard deviation of marker configuration Z     (nom, noj, 3)

    Returns:
        X               -- input marker position X                      (batch_size, nom, 3)
        sampled_zHat    -- pre-weighted local offset                    (batch_size, nom, 3)
                           computed from sampled marker configuration Z
    '''

    batch_size = np.size(b_joint, axis=0)
    noj = np.size(b_joint, axis=1)
    nom = len(markers)

    # Homogeneous marker position X 
    X = np.zeros((batch_size, nom, 4))
    sampled_zHat = np.zeros((batch_size, nom, 3))


    # Sample a set of marker configurations
    sampledZ = np.random.normal(z_mean, z_std, (batch_size, nom, noj, 3))

    # For each pose(frame),
    for j in range(batch_size):
        # Frame index of j-th sample
        # index = frame_idx[j]

        f_F = np.copy(b_F[j])

        # Add noise to local reference frame
        if opts.F_noise:
            f_F = local_reference.corruptLocalReferenceFrame(f_F, 0.1, 4, 2)

        # Transform joint transforms into local space
        b_joint[j] = np.matmul(np.linalg.inv(f_F), b_joint[j])

        # Global marker position for each pose
        global_pos = np.zeros((nom, 4)) 
        for m in range(nom):
            marker = markers[m]

            # Compute global marker positions via linear blend skinning
            pos = utils.getGlobalMarkerPosition(marker, sampledZ[j][m], b_joint[j])

            if opts.corrupt_markers:
                # Corrupt markers
                pos = utils.corruptMarkers(pos, 0.1, 0.1, 0.1)

            global_pos[m] = pos.T
        X[j] = global_pos

        # Compute pre-weighted marker offsets
        sampled_zHat[j] = marker_config.getWeightedOffset(sampledZ[j], markers)

    X = np.delete(X, 3, axis=2)         # (batch_size, nom, 3)

    return X, sampled_zHat






def main(opts):
    # paths to save rigid body, local reference frame, statistics
    localref_name = "localref_" + opts.stats_name + ".txt"
    statistics_name = "statistics_" + opts.stats_name + ".txt"

    localref_path = os.path.join(opts.stats_dir, localref_name)
    statistics_path = os.path.join(opts.stats_dir, statistics_name)

    # path for tensorboard
    tensorboard_path = opts.tensorboard_dir

    # path to LBS file
    lbs_path = os.path.join(opts.lbs_dir, opts.lbs_name)

    # import motion data
    motion = lbs_import.importMotion('train')
    markers, _ = lbs_import.readLBS(lbs_path)

    # setup model load path
    model_load_dir = opts.load_model_dir             # directory to save trained weights
    model_load_file = opts.load_model_name + ".pth"           # file to save trained weights
    model_load_path = os.path.join(model_load_dir, model_load_file)    
    
    # setup save load path
    model_save_dir = opts.save_model_dir             # directory to save trained weights
    if opts.save_model_name == 'none':
        model_save_file = opts.load_model_name + ".pth"           # file to save trained weights
    else:
        model_save_file = opts.save_model_name + ".pth"
    model_save_path = os.path.join(model_save_dir, model_save_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ### create dataset
    joint_transform = np.array(motion)
    local_reference_frame = local_reference.loadLocalReferenceFrame(localref_path)
    print(joint_transform.shape)
    print(local_reference_frame.shape)
    marker_dataset = configure_dataset.MarkerDataset(joint_transform, local_reference_frame)

    
    ### load statistics
    nof = np.size(joint_transform, axis=0)
    _, noj, nom, x_mean, x_std, y_mean, y_std, z_mean, z_std, zHat_mean, zHat_std = utils.loadStatistics(statistics_path)
    
    # put y statistics to gpu Tensor 
    y_mean = utils.toTensor(y_mean, device)
    y_std = utils.toTensor(y_std, device)

    print("Loaded statistics")


    

    ### setup dataloader, network
    # randomly split data into training and validation set
    validation_split = 0.2
    dataset_indices = list(range(nof))

    validation_len = int(np.floor(validation_split * nof))
    validation_idx = np.random.choice(dataset_indices, size=validation_len, replace=False)
    train_idx = list(set(dataset_indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    
    train_loader = torch.utils.data.DataLoader(marker_dataset, batch_size=opts.batch_size, sampler=train_sampler, num_workers=opts.num_workers)
    validation_loader = torch.utils.data.DataLoader(marker_dataset, batch_size=opts.batch_size, sampler=validation_sampler, num_workers=opts.num_workers)
    
    # resnet network
    model = resnet_model.ResNet(resnet_model.ResidualBlock, nom, noj)
    
    # load trained weights
    if os.path.exists(model_load_dir):
        if os.path.isfile(model_load_path):
            model.load_state_dict(torch.load(model_load_path))
            print('Loaded trained weights\n')
        else:
            print('Saving new weights\n')
    else:
        os.makedirs(model_load_dir)

    model = model.float()
    model.to(device)


    ### optimizer
    if opts.optimizer == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, amsgrad=True)
    elif opts.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    

    ### learning rate decay
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 500)


    ### setup tensorboard
    writer = SummaryWriter(tensorboard_path + model_save_file[:-4])


    ### start of training
    start = time.time()

    resume = opts.resume


    for epoch in range(opts.epoch):
        ### start of epoch
        epoch_start = time.time()

        ### training
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            model.train(True)

            frame_idx = data['frame_idx'].numpy()
            b_joint = np.copy(data['joint_transform'].numpy())       # b_joint (batch_size, noj, 4, 4)
            b_F = data['F'].numpy()                                  # b_F (batch_size, 4, 4)

            batch_size = np.size(frame_idx, axis=0)

            # preprocess data
            X, sampled_zHat = prepare_input(b_F, b_joint, markers, z_mean, z_std)

            # normalize input
            norm_X = (X - x_mean) / x_std                       # (nom, 3)
            norm_Z = (sampled_zHat - zHat_mean) / zHat_std      # (nom, 3)

            # Concatenate data
            inputs = np.concatenate((norm_X, norm_Z), axis=1)   # (batch_size, nom*2, 3)
            inputs = np.reshape(inputs, (batch_size, -1))
            inputs = utils.toTensor(inputs, device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)       # (batch_size, noj, 3, 4)

            # Denormalize Y
            outputs = outputs * y_std + y_mean

            # Compute loss 
            b_joint = np.delete(b_joint, 3, axis=2)     # Ground Truth : (batch_size, noj, 3, 4)
            b_joint = b_joint.reshape(batch_size, -1)   # (batch_size, noj*3*4)
            b_joint = utils.toTensor(b_joint, device)
            loss = utils.get_loss(outputs, b_joint, noj, device)

            # Backpropagate
            loss.backward()

            # Update parameters
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        # write training loss to tensorboard
        print('[%d epochs] training loss: %.3f ' % (epoch+resume+1, running_loss/len(train_loader)))
        writer.add_scalar('training_loss',
                        running_loss/len(train_loader),
                        epoch+resume)


        ### validation
        validation_loss = 0.0
        for i, data in enumerate(validation_loader, 0):
            model.eval()
            with torch.no_grad():
                frame_idx = data['frame_idx'].numpy()
                b_joint = np.copy(data['joint_transform'].numpy())       # b_joint (batch_size, noj, 4, 4)
                b_F = data['F'].numpy()                                  # b_F (batch_size, 4, 4)

                batch_size = np.size(frame_idx, axis=0)
                X, sampled_zHat = prepare_input(b_F, b_joint, markers, z_mean, z_std)

                # Normalize data
                norm_X = (X - x_mean) / x_std
                norm_Z = (sampled_zHat - zHat_mean) / zHat_std
                
                # Concatenate data
                inputs = np.stack((norm_X, norm_Z), axis=1)    # (batch_size, 2, m, 3)
                inputs = np.reshape(inputs, (batch_size, -1))
                inputs = utils.toTensor(inputs, device)

                # Forward
                outputs = model(inputs)       # (batch_size, noj, 3, 4)
                
                # Denormalize Y
                outputs = outputs * y_std + y_mean

                # Compute loss 
                b_joint = np.delete(b_joint, 3, axis=2)     # Ground Truth : (batch_size, noj, 3, 4)
                b_joint = b_joint.reshape(batch_size, -1)   # (batch_size, noj*3*4)
                b_joint = utils.toTensor(b_joint, device)

                loss = utils.get_loss(outputs, b_joint, noj, device)

                # Track loss
                validation_loss += loss.item()


        # write validation loss to tensorboard
        print('[%d epochs] validation loss: %.3f' % (epoch+resume+1, validation_loss/len(validation_loader)))
        writer.add_scalar('validation_loss',
                        validation_loss/len(validation_loader),
                        epoch+resume)

        ### end of epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time %.3f min\n' % (epoch_time/60))

        # scheduler.step()

        ### save trained model
        torch.save(model.state_dict(), model_save_path)

    writer.close()

    training_time = time.time() - start
    print('Training done')
    print('Training time %.3f min\n' % (training_time/60))
    

if __name__ == '__main__':
    opts = utils.train_parser()
    print(f'\nReceived options:\n{opts}')
    main(opts)

    