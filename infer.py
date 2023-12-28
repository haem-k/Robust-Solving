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

### visualize lib
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate

### custom lib
from preprocess import local_reference, marker_config
from models import resnet_model
from data import lbs_import, configure_dataset 
import utils


def infer(opts):
    '''
    Infer the network
    
    Parameters:
        opts
    
    Returns:
        motion                  -- motion info read from motion files                   (nof, noj, 4, 4)
        input_marker_position   -- input marker position for each inference             (nof, nom, 4)
        result_joint            -- output of the network transformed to global space    (nof, noj, 4, 4)
    
    '''
    # paths for rigidbody, local reference frame, statistics
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
    # motion = lbs_import.readMotion("./data/samba_dance/Motion.txt")
    motion = lbs_import.importMotion('test')
    markers, offsets = lbs_import.readLBS(lbs_path)
    
    # (nof, nom, 4)
    marker_gpos = lbs_import.getMarkerGpos(motion, offsets, markers)

    # number of frames, joints, markers
    nof = np.size(motion, axis=0)
    noj = np.size(motion, axis=1)
    nom = len(markers)


    ### Get marker configuration zHat
    # compute local offset and get weighted sum
    marker_offset = np.empty((nom, 4)) 
    for m in range(nom):
        marker = markers[m]
        pos = np.zeros((4, 1))
        
        # get number of effective joints
        jnum = np.size(marker.indices, 0)

        for i in range(jnum):
            # joint which affects this marker's position
            index = marker.indices[i]
            weight = marker.weights[i]
            
            # skinning offset matrix, and joint transform of this joint
            offsetMat = offsets[index]

            # weighted sum
            pos += weight * np.matmul(offsetMat, marker.vpos)
        
        marker_offset[m] = pos.T

    # zHat is constant throughout frames for test
    zHat = np.array([marker_offset for i in range(nof)])
    zHat = np.delete(zHat, 3, axis=2)
    
    

    ### setup model
    model_save_dir = opts.load_model_dir             # directory to save trained weights
    model_save_file = opts.load_model_name + ".pth"           # file to save trained weights
    model_save_path = os.path.join(model_save_dir, model_save_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ### setup dataset
    joint_transform = np.array(motion) 
    local_reference_frame = local_reference.loadLocalReferenceFrame(localref_path)
    marker_dataset = configure_dataset.MarkerDataset(joint_transform, local_reference_frame)
    

    ### load statistics
    _, _, _, x_mean, x_std, y_mean, y_std, _, _, zHat_mean, zHat_std = utils.loadStatistics(statistics_path)

    y_mean = torch.from_numpy(y_mean).float().to(device)
    y_std = torch.from_numpy(y_std).float().to(device)


    ### setup dataloader, network
    testloader = torch.utils.data.DataLoader(marker_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    
    # resnet network
    model = resnet_model.ResNet(resnet_model.ResidualBlock, nom, noj)

    # load trained weights
    if os.path.exists(model_save_dir):
        if os.path.isfile(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            print('Loaded trained weights')
    else:
        print('Cannot find trained model')
        exit(0)

    model = model.float()
    model.to(device)
    model.eval()


    ### test the model
    result_joint = np.zeros((nof, noj, 4, 4))
    input_marker_position = np.zeros((nof, nom, 4))
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # running_loss = 0.0
            frame_idx = data['frame_idx'].numpy()
            b_joint = np.copy(data['joint_transform'].numpy())       # b_joint (batch_size, noj, 4, 4)

            batch_size = np.size(frame_idx, axis=0)

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
            inputs = torch.from_numpy(inputs).float()
            inputs = inputs.to(device)

            # Forward
            outputs = model(inputs)       # (batch_size, noj*3*4)

            # Denormalize Y
            outputs = outputs * y_std + y_mean

            # Compute loss 
            b_joint = np.delete(b_joint, 3, axis=2)     # Ground Truth : (batch_size, noj, 3, 4)
            b_joint = b_joint.reshape(batch_size, -1)   # (batch_size, noj*3*4)
            b_joint = utils.toTensor(b_joint, device)

            loss = utils.get_loss(outputs, b_joint, noj, device)
            print(f'batch {i} loss: {loss.item()}')

            # outputs to numpy
            outputs = outputs.cpu().numpy()
            outputs = np.reshape(outputs, (batch_size, noj, 3, 4))
            
            # Save the output
            for j in range(batch_size):
                for k in range(noj):
                    # make output homogeneous
                    joint = np.append(outputs[j][k], [[0, 0, 0, 1]], axis=0)   

                    # transform back to global
                    result_joint[frame_idx[j]][k] = np.matmul(F[j], joint) 

    test_time = time.time() - start
    print('Test done')
    print('Test time %.3f min\n' % (test_time/60))

    return motion, result_joint, input_marker_position


def getJointPos(motion, frame, jointIdx):
    jntTrf = motion[frame][jointIdx]
    return [jntTrf[0][3], jntTrf[1][3], jntTrf[2][3]]


class Canvas(app.Canvas):

    def __init__(self, opts, vert, frag, motion, result_joint, input_marker_position):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        ps = self.pixel_scale

        # Create vertices
        n = 100
        data = np.zeros(n, [('a_id', np.float32)])
        for i in range(n):
            data[i][0] = np.float32(i)

        self.frame = 0
        self.x = 0
        self.y = 0
        self.z = 5
        self.program = gloo.Program(vert, frag)
        
        self.view = translate((self.x, self.y, -self.z))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()

        self.program.bind(gloo.VertexBuffer(data))
        
        # initialize uniform
        self.program['u_model'] = self.model
        self.program['u_view']  = self.view
        self.program['u_size']  = 5 / self.z
        self.program['u_rad'] = np.float32(1.0)
        for i in range(n):
            self.program[f'u_pos[{i}]'] = [0, 0, 0]
        
        self.theta = 0
        self.phi = 0

        gloo.set_state('translucent', clear_color='white')

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.run = True

        self.show()
        self.slowDown = 0

        self.init_pose = np.eye(4, 4)


        ### get data
        self.motion = motion
        self.input_marker_position = input_marker_position
        self.result_joint = result_joint
        # markers, _ = lbs_import.readLBS("./data/LBS_test.txt")

        # number of frames, joints, markers
        self.nof = np.size(self.motion, axis=0)
        self.noj = np.size(self.motion, axis=1)
        self.nom = np.size(self.input_marker_position, axis=1)

        ### save visualize options
        self.show_gt_joints = opts.show_gt_joints
        self.show_input_markers =  opts.show_input_markers


    def on_key_press(self, event):
        if event.text == ' ':
            # if self.timer.running:
            #     self.timer.stop()
            # else:
            #     self.timer.start()
            self.run = not self.run

    def on_timer(self, event):
        self.program['u_model'] = self.model
        
        if self.run:
            if self.slowDown > 2:
                self.frame = self.frame + 1
                if self.frame >= self.nof:
                    self.frame = 0
                self.slowDown = 0
            else :
                self.slowDown = self.slowDown + 1
        
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.z -= event.delta[1]
        self.z = max(2, self.z)
        self.view = translate((self.x, self.y, -self.z))

        self.program['u_view'] = self.view
        self.program['u_size'] = 5 / self.z
        #self.update()
        
    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 2:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            self.x = self.x + 0.001 * (x1 - x0)
            self.y = self.y - 0.001 * (y1 - y0)
            
            self.view = translate((self.x, self.y, -self.z))
            self.program['u_view'] = self.view
        elif event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]

            #self.theta += (y1 - y0)
            self.phi += (x1 - x0)
            self.model = np.dot(rotate(self.theta, (1, 0, 0)),
                                rotate(self.phi, (0, 1, 0)))
            # self.translate_center(X1 - X0, Y1 - Y0)

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection


    # Called evey frame
    def on_draw(self, event):
        gloo.clear()

        # drawing scale parameter
        scale = 0.01

        ##################### Draw Ground Truth Joints #####################
        if self.show_gt_joints:
            self.program['u_color'] = [0, 0, 0]
            for i in range(self.noj): 
                ji = getJointPos(self.motion, self.frame, i)
                uniform_name = f'u_pos[{i}]'
                self.program[uniform_name] = [scale * ji[0], scale * ji[1], scale * ji[2]]
            
            self.program['u_rad'] = np.float32(1.0)
            self.program['u_num'] = np.float32(self.noj) # noj: number of joint
            self.program.draw('points')


        #################### Draw Input Markers #####################
        if self.show_input_markers:
            self.program['u_color'] = [1, 0, 0] # color (RGB)
            cnt = 0
            for m in range(self.nom):
                pos = self.input_marker_position[self.frame][m]
                
                uniform_name = f'u_pos[{cnt}]'
                self.program[uniform_name] = [scale * pos[0], scale * pos[1], scale * pos[2]]
                cnt = cnt + 1

            self.program['u_rad'] = np.float32(1.0)
            self.program['u_num'] = np.float32(self.nom) # noj: number of joints
            self.program.draw('points')
        
        ##################### Draw Output Joints #####################
        # always show output result
        self.program['u_color'] = [0, 1, 0]

        for i in range(self.noj):
            joint = self.result_joint[self.frame][i]
            ji = [joint[0][3], joint[1][3], joint[2][3]]
            uniform_name = f'u_pos[{i}]'
            self.program[uniform_name] = [scale * ji[0], scale * ji[1], scale * ji[2]]
        
        self.program['u_rad'] = np.float32(1.0)
        self.program['u_num'] = np.float32(self.noj) # noj: number of joint
        self.program.draw('points')



if __name__ == '__main__':
    opts = utils.infer_parser()
    print(f'\nReceived options:\n{opts}')

    file = open("./shader/model.vs")
    vert = file.read()
    file.close()

    file = open("./shader/model.fs")
    frag = file.read()
    file.close()

    ### test the network
    print('\nTesting the model...')
    motion, result_joint, input_marker_position = infer(opts)

    ### visualize the result
    print('Visualizing the result...')
    c = Canvas(opts, vert, frag, motion, result_joint, input_marker_position)
    app.run()