import numpy as np
import os
from scipy.spatial.transform import Rotation as R



'''
Functions for rigid body
'''
def getRigidBody(motion, marker_gpos, markers):
    '''
    Extract rigid body,
    get mean location of the markers around the torso
    relative to Spine2 joint from all frames
    
    Parameters:
        motion          -- motion info read from motion files       (nof, noj, 4, 4)
        marker_gpos     -- global marker position in all frames     (nof, nom, 4)
        markers         -- markers info                             (Marker list)
    
    Returns:
        torso       -- indices of torso markers
        rigid_body  -- homogeneous mean position of the markers around the torso, relative to Spine2 joint      (nofTorso, 4)
    '''


    torso_name = ['RFSH', 'LFSH', 'CLAV', 'STRN', 'RMWT', 'RFWT', 'LFWT', 'LMWT', 'LBSH', 'RBSH', 'C7', 'T10', 'LBWT', 'RBWT']
    rigid_body = np.zeros((len(torso_name), 4))
    torso = []

    for j in range(len(torso_name)):
        markerName = torso_name[j]
        idx = [idx for idx, m in enumerate(markers) if markerName in m.name][0]     # index of current torso marker
        torso.append(idx)

    nof = np.size(motion, axis=0)

    for i in range(nof):
        # spine2 joint's tranformation matrix
        spine = motion[i][3]

        for j in range(len(torso_name)):
            idx = torso[j]
            pos = marker_gpos[i][idx]
            
            # get relative marker position by multiplying marker position with inverse of spine2
            relativePos = np.matmul(np.linalg.inv(spine), pos)
            relativePos = relativePos.T.squeeze()

            # add the resulting position
            rigid_body[j] = np.add(rigid_body[j], relativePos)

    # compute the mean location of each marker    
    rigid_body = rigid_body / nof

    return torso, rigid_body

def saveRigidBody(torso, rigid_body, path):
    '''
    Save computed rigid body in text file
    
    Parameters:
        torso       -- indices of torso markers
        rigid_body  -- homogeneous mean position of the markers around the torso, relative to Spine2 joint      (nofTorso, 4)
        path        -- path to save rigid body, including file name
    
    Returns:
        None
    '''
    f = open(path, 'w')

    # write torso marker indices
    marker_idx = ' '.join(map(str, torso))
    f.write("torso\n" + marker_idx + "\n\n" + "rigid body\n")

    # write rigid body
    np.savetxt(f, rigid_body, fmt='%.8f')

    f.close()

def loadRigidBody(path):
    '''
    Load computed rigid body from text file
    
    Parameters:
        path        -- path to rigid body text file (string)
    
    Returns:
        torso       -- indices of torso markers
        rigid_body  -- homogeneous mean position of the markers around the torso, relative to Spine2 joint      (nofTorso, 4)
    '''

    f = open(path, 'r')
    
    line = f.readline()     # torso
    line = f.readline()     # list of names
    torso = line.split()
    torso = list(map(int, torso))
    

    f.readline()
    line = f.readline()         # rigid body
    rigidBody = np.loadtxt(f)   # numpy array of rigid body
    rigidBody[:,3] = 1

    return torso, rigidBody




'''
Functions for ICP
'''
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    
    Parameters:
        A   -- Nxm numpy array of corresponding points 
        B   -- Nxm numpy array of corresponding points 
    
    Returns:
        T   -- (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R   -- mxm rotation matrix
        t   -- mx1 translation vector
    '''

    # number of torso markers may differ because of occluded markers    
    assert A.shape == B.shape       # (# of markers not occluded, 3)

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A # 10*3
    BB = B - centroid_B # 10*3

    # rotation matrix
    H = np.dot(AA.T, BB) # 3*10 & 10*3 -> 3*3
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t        # (4,4)

    return T, R, t

def getDistance(src, dst):
    '''
    Compute distance between labeled torso markers (given in order)
    except for occluded markers

    Parameters:
        src     -- points from rigid body, (number of torso markers, 3)  
        dst     -- points from marker data, (number of torso markers, 3) 
    
    Returns:
        distances   -- Euclidean distances between each marker pair (src & dst)
        indices     -- indices of markers considered when computing best fit tranformation (not occluded marker)
    '''

    assert src.shape == dst.shape

    # number of torso marker
    nofT = src.shape[0]

    distances = np.zeros(nofT)
    indices = []

    for i in range(nofT):
        # if the marker is occluded, continue to next marker
        if np.count_nonzero(dst[i]) == 0:
            continue
        distances[i] = np.linalg.norm(src[i]-dst[i])
        indices.append(i)

    return distances, indices

def icp(source, destination, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method
    finds local reference frame that maps source points on to destination points 

    Parameters:
        source          -- Nxm numpy array of source m dimensional points (rigid body)
        destination     -- Nxm numpy array of destination m dimensional points (torso markers)
        init_pose       -- (m+1)x(m+1) homogeneous transformation
        max_iterations  -- exit algorithm after max_iterations
        tolerance       -- convergence criteria

    Returns:
        T           -- final homogeneous transformation that maps source on to destination      (4,4)
        distances   -- Euclidean distances (errors) of the corresponding points
        i           -- number of iterations to converge
    '''

    assert source.shape == destination.shape
    # get number of dimensions
    m = source.shape[1]
    # given points are already homogeneous
    m = m-1

    # copy them to maintain the originals
    src = np.copy(source.T)
    dst = np.copy(destination.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the distance between markers with same name
        distances, indices = getDistance(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        # only use markers in indices
        T,_,_ = best_fit_transform(src[:m,indices].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(source[:,:m], src[:m,:].T)

    return T, distances, i




'''
Functions for local reference frame F
'''
def getLocalReferenceFrame(pos, path):
    '''
    Compute local reference frame F for a frame using icp algorithm

    Parameters:
        pos         -- global marker position in each frame    (nom, 4)
        path        -- path to local reference frame text file (string)
    
    Returns:
        F   -- local reference frame F for this frame     (4, 4)
    '''

    torso, rigidBody = loadRigidBody(path)

    # destination markers for rigid body
    destination = np.zeros((len(torso), 4))

    # get homogeneous destination points
    for j in range(len(torso)):
        # get marker index of a torso marker
        idx = torso[j]
        destination[j] = pos[idx]
    destination[:, 3] = 1

    # using icp
    F, _, _ = icp(rigidBody, destination, init_pose=None, max_iterations=100, tolerance=0.001)

    return F

def saveLocalReferenceFrame(F, path):
    '''
    Save full local reference frame in text file

    Parameters:
        F           -- full local reference frame F      (nof, 4, 4)
        path        -- path to save local reference frmae, including file name (string)

    Returns:
        None

    '''
    
    f = open(path, 'w')

    nof = np.size(F, axis=0)
    F = np.reshape(F, (nof, -1))

    np.savetxt(f, F, fmt='%.8f')

    f.close()

def loadLocalReferenceFrame(path):
    '''
    Load computed local reference frame from text file

    Parameters:
        path        -- path to local reference frame text file (string)

    Returns:
        F           -- full local reference frame F      (nof, 4, 4)
    '''

    f = open(path, 'r')
    F = np.loadtxt(f)
    nof = np.size(F, axis=0)
    F = np.reshape(F, (nof, 4, 4))

    return F

def corruptLocalReferenceFrame(f_F, sig, betaR, betaT):
    '''
    Corrupt local reference frame to deal with incorrect rigid body alignment
    
    Parameters:
        f_F             -- local reference frame F for this frame   (4, 4)
        sig             -- parameter to sample noise probability
        betaR           -- magnitude for shifting (Rotation by degree)
        betaT           -- magnitude for shifting (Translation)
    
    Returns:
        corrupt_F     -- corrupted single local reference frame     (4, 4)
    '''

    # Sample probability whether to corrupt F or not
    alpha = np.random.normal(0, sig)

    # Sample using clipped probabilities if this F should be corrupted or not
    prob = min(abs(alpha), 2*sig)
    corrupt = np.random.binomial(1, prob)

    # (4, 4)
    corrupt_F = np.copy(f_F)

    if corrupt == 1:
        # Sample rotation noise  
        # Sample the magnitude by which to shift rotation
        magR_X = np.random.uniform(-1*betaR, betaR)  
        magR_Y = np.random.uniform(-1*betaR, betaR)  
        magR_Z = np.random.uniform(-1*betaR, betaR)
        r = R.from_euler('xyz', [magR_X, magR_Y, magR_Z], degrees=True)
        
        # Multiply original rotation by noise
        noise = r.as_matrix()                       # (3, 3)
        rotation = corrupt_F[0:3, 0:3]              # get rotation part of F
        noise = np.matmul(noise, rotation)
        corrupt_F[0:rotation.shape[0], 0:rotation.shape[1]] = noise

        # Sample translation noise
        # Sample the magnitude by which to shift translation
        magT_X = np.random.uniform(-1*betaT, betaT)
        magT_Y = np.random.uniform(-1*betaT, betaT)
        magT_Z = np.random.uniform(-1*betaT, betaT)
        corrupt_F[0][3] = corrupt_F[0][3] + magT_X
        corrupt_F[1][3] = corrupt_F[1][3] + magT_Y
        corrupt_F[2][3] = corrupt_F[2][3] + magT_Z

    return corrupt_F
