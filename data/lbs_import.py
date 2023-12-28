import numpy as np
import os

class Marker:
    def __init__(self):
        self.indices = np.array([0])
        self.weights = np.array([0.0])
        self.vpos = np.zeros((4, 1))
        self.name = ""
    
    def globalPosition(self, frame, offsets, motion):
        '''
        Get marker position in a frame

        Parameters:
            frame       -- current frame index                                  (int)
            offsets     -- skinning offset transform to get marker position     (dict {jntIndex: offsetMatrix})
            motion      -- motion info read from Motion.txt                     (nof, noj, 4, 4)

        Returns:
            pos         -- position of the marker   (4, 1)
        '''

        pos = np.zeros((4, 1))
        
        ### get number of effective joints
        jnum = np.size(self.indices, 0)

        for i in range(jnum):
            ### joint which affects this marker's position
            index = self.indices[i]
            weight = self.weights[i]
            
            ### skinning offset matrix, and joint transform of this joint
            offsetMat = offsets[index]
            globalJntTrf = motion[frame][index]

            ### LBS 
            trf = np.matmul(globalJntTrf, offsetMat)
            
            ### weighted sum
            pos += weight * np.matmul(trf, self.vpos)
                    
        return pos


def importMotion(mode, path='./data'):
    '''
    Retrieve motion text data from all subdirectory of 'path'

    Parameters:
        mode        -- 'train' import train data / 'test' import test data          (string)
        path        -- motion data directory path, default directory = ./data       (string)

    Returns:
        data        -- np array of motion   (nof, noj, 4, 4)
    '''

    if mode == 'train':
        path = path + '/train'
    elif mode == 'test':
        path = path + '/test'
    else:
        print('Invalid import mode')
        exit(0)


    ### Collect motions from txt files
    motions = []
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.txt'):
                data_path = os.path.join(dirpath, file)
                m = readMotion(data_path)
                motions.append(m)

            

    ### Convert to numpy
    data = np.array(motions[0])

    for i in range(1, len(motions)):
        d = np.array(motions[i])
        data = np.concatenate((data, d), axis=0)
    
    print(f'Imported {mode} data')
    print(f'{np.size(data, axis=0)} frames\n')
    return data



def readMotion(path, nof=-1):
    '''
    Get motion data from a single text file

    Parameters:
        path        -- motion text file path (string)
        nof         -- number of frames to parse

    Returns:
        motion      -- motion info read from motion files     (nof, noj, 4, 4)
    '''
    f = open(path, 'r')

    # nof x noj x 4 x 4
    motion = []

    # nof
    line = f.readline()
    line = line.strip()
    line = line.split(":")
    if nof == -1:
        nof = int(line[1])

    line = f.readline()

    for i in range(nof):
        line = f.readline()
        line = line.strip()
        line = line.split(",")
        line = list(map(float, line))
        line = np.reshape(line, (-1, 16))
        noj, matsize = np.shape(line)
        motion_f = []
        for mat in line:
            trf = np.reshape(mat, (4, 4))
            motion_f.append(trf)    
        motion.append(motion_f)
    f.close()
    return motion


def readLBS(path):
    '''
    Get marker info and offset transform from text file (LBS info)

    Parameters:
        path        -- marker & offset text file path (string)

    Returns:
        markers     -- marker info saved in Marker class                    (Marker list)
        offsets     -- skinning offset transform to get marker position     (dict {jntIndex: offsetMatrix})
    '''

    markers = []
    offsets = {}

    f = open(path, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        if line == "vertex id\n":
            marker = Marker()
            
            line = f.readline()        # vertex id
            
            f.readline()
            line = f.readline()         # marker name
            marker.name = line

            f.readline()
            line = f.readline()        # vertex position
            line = line.strip()
            line = line.split(",")
            marker.vpos = np.array(list(map(float, line)))
            marker.vpos = np.append(marker.vpos, [1.0], axis=0)
            marker.vpos = np.reshape(marker.vpos, (4, 1))

            f.readline()
            line = f.readline() # skinning indices
            line = line.strip()
            indices = line.split(",")
            marker.indices = np.array(list(map(int, indices)))
            
            f.readline()
            line = f.readline() # skinning weights
            line = line.strip()
            weights = line.split(",")
            marker.weights = np.array(list(map(float, weights)))
            markers.append(marker)

        elif line == "skinning offset transform\n":
            line = f.readline() # number of offset transforms
            line = line.strip()
            offsetMatNum = int(line)
            for _ in range(0, offsetMatNum):
                line = f.readline()
                offsetline = line.split(":")
                
                jntidx = int(offsetline[0])
                
                jntoffset = offsetline[1]
                jntoffset = jntoffset.strip()
                jntoffset = jntoffset.split(",")
                jntoffset = list(map(float, jntoffset))
                jntoffset = np.reshape(jntoffset, (4, 4))
                jntoffset = np.matrix(jntoffset)
                offsets[jntidx] = jntoffset
    f.close()
    return markers, offsets
    
def getMarkerGpos(motion, offsets, markers):
    '''
    Get global marker positions

    Parameters:
        motion      -- motion info read from motion files                   (nof, noj, 4, 4)
        offsets     -- skinning offset transform to get marker position     (dict {jntIndex: offsetMatrix})
        markers     -- markers info                                         (Marker list)

    Returns:
        marker_gpos -- global marker position in all frames                 (nof, nom, 4)
    
    '''
    nof = np.size(motion, axis=0)
    nom = len(markers)
    marker_gpos = np.zeros((nof, nom, 4))

    for f in range(nof):
        for m in range(len(markers)):
            marker_gpos[f][m] = markers[m].globalPosition(f, offsets, motion).T

    return marker_gpos
