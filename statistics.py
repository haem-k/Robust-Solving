import numpy as np
import time
import os

import utils
from preprocess import local_reference, marker_config
from models import resnet_model
from data import lbs_import, configure_dataset 

if __name__ == "__main__":
    opts = utils.stats_parser()

    # paths to save rigid body, local reference frame, statistics
    rigidbody_name = "rigidbody_" + opts.stats_name + ".txt"
    localref_name = "localref_" + opts.stats_name + ".txt"
    statistics_name = "statistics_" + opts.stats_name + ".txt"

    rigidbody_path = os.path.join(opts.stats_dir, rigidbody_name)
    localref_path = os.path.join(opts.stats_dir, localref_name)
    statistics_path = os.path.join(opts.stats_dir, statistics_name)

    # path to LBS file
    lbs_path = os.path.join(opts.lbs_dir, opts.lbs_name)

    # import all motion data
    motion = lbs_import.importMotion('train')
    # motion = lbs_import.readMotion('./data/samba_dance/Motion.txt')
    markers, offsets = lbs_import.readLBS(lbs_path)

    nom = len(markers)               
    nof = np.size(motion, axis=0)       
    noj = np.size(motion, axis=1)        


    print('Preprocessing this dataset...')
    start = time.time()

    # get global marker position in every frames
    marker_gpos = lbs_import.getMarkerGpos(motion, offsets, markers)

    # compute rigid body
    torso, rigid_body = local_reference.getRigidBody(motion, marker_gpos, markers)
    local_reference.saveRigidBody(torso, rigid_body, rigidbody_path)

    # local reference frame
    F = np.zeros((nof, 4, 4))

    # marker configuration
    Z = np.zeros((nof, nom, noj, 3))

    # pre-weighted local offset
    zHat = np.zeros((nof, nom, 3))

    # local marker position
    marker_lpos = np.zeros((nof, nom, 4))

    # local joint transform
    jnttrf_local = np.zeros((nof, noj, 4, 4))

    # For each frame,
    for f in range(nof):
        # find local reference frame by fitting rigidbody
        F[f] = local_reference.getLocalReferenceFrame(marker_gpos[f], rigidbody_path)

        # compute local offset from marker to each joint
        Z[f] = marker_config.getMarkerConfig(motion[f], marker_gpos[f])
        zHat[f] = marker_config.getWeightedOffset(Z[f], markers)
        
        # transform marker position and joint trf to local space
        marker_lpos[f] = utils.getLocalMarkerPos(marker_gpos[f], F[f])
        jnttrf_local[f] = utils.getLocalJntTrf(motion[f], F[f])

    local_reference.saveLocalReferenceFrame(F, localref_path)


    processing_time = time.time() - start
    print('Done')
    print('Preprocessing time %.3f min\n' % (processing_time/60))



    ### Compute statistics
    print('Saving statistics...')
    
    y_mean = np.mean(jnttrf_local, axis=0)   # (noj, 4, 4)
    y_std = np.std(jnttrf_local, axis=0)     # (noj, 4, 4)

    y_mean = np.delete(y_mean, 3, axis=1)       # (noj, 3, 4)
    y_std = np.delete(y_std, 3, axis=1)         # (noj, 3, 4)

    y_mean = y_mean.ravel()                     # (noj*3*4, )
    y_std = y_std.ravel()                       # (noj*3*4, )

    x_mean = np.mean(marker_lpos, axis=0)       # (nom, 4)
    x_std = np.std(marker_lpos, axis=0)         # (nom, 4)
    x_std[x_std < 0.0001] = 0.0001        # Threshold standard deviation

    x_mean = np.delete(x_mean, 3, axis=1)       # (nom, 3)
    x_std = np.delete(x_std, 3, axis=1)         # (nom, 3)

    z_mean = np.mean(Z, axis=0)                 # (nom, noj, 3)
    z_std = np.std(Z, axis=0)                   # (nom, noj, 3)

    z_mean = np.reshape(z_mean, (nom, -1))
    z_std = np.reshape(z_std, (nom, -1))

    zHat_mean = np.mean(zHat, axis=0)           # (nom, 3)
    zHat_std = np.std(zHat, axis=0)             # (nom, 3)
    zHat_std[zHat_std < 0.0001] = 0.0001        # Threshold standard deviation

    ### Save statistics
    utils.saveStatistics(nof, noj, nom, x_mean, x_std, y_mean, y_std, z_mean, z_std, zHat_mean, zHat_std, statistics_path)
    print('Done')
    