import numpy as np


def getMarkerConfig(f_motion, f_marker_gpos):
    '''
    Get local offset from each marker to each joint per frame

    Parameters:
        f_motion          -- motion info of a frame read from Motion.txt    (noj, 4, 4)
        f_marker_gpos     -- global marker position in a frame              (nom, 4)

    Returns:
        f_Z   -- marker configuration Z relative to each joint              (nom, noj, 3)
    '''

    noj = np.size(f_motion, axis=0)
    nom = np.size(f_marker_gpos, axis=0)

    f_Z = np.zeros((nom, noj, 3))    

    inv = np.zeros((noj, 4, 4))
        
    # get inverse of every joint's tranformation matrix
    for j in range(noj):
        jntTrf = f_motion[j]
        inv[j] = np.linalg.inv(jntTrf)

    # compute local offset of this frame
    for m in range(nom):
        for j in range(noj):
            # get relative marker position by multiplying marker position to inverse of each joint 
            relativePos = np.matmul(inv[j], f_marker_gpos[m])
            relativePos = relativePos.T.squeeze()
        
            f_Z[m][j] = relativePos[:3]
        
    return f_Z    


def getWeightedOffset(f_Z, markers):
    '''
    Get pre-weighted local offset per frame

    Parameters:
        f_Z         -- marker configuration Z relative to each joint in a frame    (nom, noj, 3)
        markers     -- markers info                     (53,)

    Return:
        f_zHat      -- pre-weighted local offset    (nom, 3)
    '''

    nom = np.size(f_Z, axis=0)
    
    f_zHat = np.zeros((nom, 3))
    for m in range(nom):
        marker = markers[m]
        
        # number of effective joints
        noj = np.size(marker.indices, axis=0)
        
        # local offset of only effective joints (noj, 3)
        localOffset = f_Z[m][marker.indices]

        # compute weighted sum of local offsets
        for j in range(noj):
            f_zHat[m] += localOffset[j] * marker.weights[j]

    return f_zHat