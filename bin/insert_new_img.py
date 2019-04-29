import logging
import argparse
import os
import cv2
import numpy as np
import copy
import time

from opensfm import commands
from opensfm.dataset import DataSet
from opensfm.reconstruction import resect, grow_reconstruction
from opensfm.types import Shot, Pose
import opensfm.matching as matching

logger = logging.getLogger(__name__)

import shutil
import json
import pymap3d




def insert_new_img(data_dir):
    """
    Insert new images into point cloud using image matching.
    New images must be saved in <data_dir>/images, along with the original images used to create an existing
    reconstruction with openSFM

    :param data_dir: directory of data
    :return: None; updates reconstruction files
    """
    # Load data set
    if not os.path.exists(data_dir):
        raise IOError("Could not locate input directory {}".format(data_dir))
    data = DataSet(data_dir)

    # Create lists of the new and original images
    new_img, orig_img = find_new_img(data)
    if len(new_img) == 0:
        print("no new images found")
        return

    # Compute features (will only run for new images)
    args = argparse.Namespace(dataset=data_dir)
    mod = {module.Command().name: module for module in commands.opensfm_commands}
    mod['extract_metadata'].Command().run(args=args)  # read exif data
    mod['detect_features'].Command().run(args=args)  # compute features

    # Match features of new images to all images
    mod['match_features'].Command().run(args=args, new_img=new_img)
    mod['create_tracks'].Command().run(args=args, new_img=new_img)  # modify match-to-track stitching

    recon = data.load_reconstruction()
    for rec in recon:
        rec.cameras = data.load_camera_models()  # TODO determine where/how to insert new cameras - LH 8/3/18
    graph = data.load_tracks_graph()

    rec = recon[0]  # TODO determine the correct index for recon - LH 8/3/18
    for img in new_img:
        if img not in graph:  # only process images that registered in tracking
            print("Could not track image {}".format(img))
            continue
        rec, _ = grow_reconstruction(data=data, graph=graph, images={img}, gcp=None, reconstruction=rec)

    data.save_reconstruction(recon)  # save out modified cloud


def localize_shots(shots, ref_lla):
    res = {'shot_poses':[]}
    for shot in shots:
        translation = shot.pose.get_origin()
        e, n, u = translation
        print(('Converting to LLA using ref_lla', ref_lla))
        translation_lla = pymap3d.enu2geodetic(e, n, u, ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        pose_  = {'rotation': list(map(float, shot.pose.rotation)),
                  'translation': list(map(float, translation_lla))}
        res['shot_poses'].append(pose_)
    return res


def insert_img_cloud(data_dir, overwrite_shots=True, new_img=None):
    """
    Insert new images into point cloud using point cloud matching.
    New images must be saved in <data_dir>/images, along with the original images used to create an existing
    reconstruction with openSFM

    :param data_dir: directory of data
    :param overwrite_shots: bool value indicating if the original shots in the cloud should be overwritten. If False,
    the original shots are kept, and the new ones appended
    :param new_img: list of images to insert into cloud
    :return: None; updates reconstruction files
    """
    # Load data set
    if not os.path.exists(data_dir):
        raise IOError("Could not locate input directory {}".format(data_dir))
    data = DataSet(data_dir)

    # # Create lists of the new and original images
    # if new_img is None:
    #     new_img, _ = find_new_img(data)
    # elif isinstance(new_img, list):
    #     # set of original images
    #     lost_img = set(new_img) - set(data.images())  # images that could not be found
    #     if len(lost_img) > 0:
    #         raise IOError("Could not locate images {}".format(lost_img))
    # else:
    #     raise IOError("Input new_img must be a list")
    # if len(new_img) == 0:
    #     print "no new images found"
    #     return

    # Compute features (will only run for new images)
    args = argparse.Namespace(dataset=data_dir)
    mod = {module.Command().name: module for module in commands.opensfm_commands}
    mod['extract_metadata'].Command().run(args=args)  # read exif data
    print args, new_img
    mod['detect_features'].Command().run(args=args, images=new_img)  # compute features

    # Load reconstruction
    recon = data.load_reconstruction()

    # Insert new images
    recon_idx = 0  # TODO find out how to handle multiple clouds
    print("reconstruction has {} points".format(len(recon[recon_idx].points)))
    shots = []
    for img in new_img:
        t0 = time.time()
        shot = match_to_cloud(data, recon, recon_idx, img)
        print("runtime = {}".format(time.time() - t0))
        shots.append(shot)

    if overwrite_shots: # Remove all shots except the new ones
        print('Overwrite shots enabled')
        recon[recon_idx].shots = {img: shot for img, shot in recon[recon_idx].shots.items() if img in new_img}
        recon = [recon[recon_idx]]  # remove other clouds

    ref_lla = json.load(open(data_dir + os.sep + 'reference_lla.json'))

    # data.save_reconstruction(recon)  # save out modified cloud
    return localize_shots(shots, ref_lla)



def find_new_img(data):
    """
    Create lists of the new and original images in data set
    :param data:
    :return new_img:
    :return orig_img:
    """

    # Check inputs
    if not isinstance(data, DataSet):
        raise IOError("Input data must be a DataSet objet")

    # Search for new images
    new_img, orig_img = [], []
    for img in data.images():
        if data.feature_exists(img):
            orig_img.append(img)
        else:
            new_img.append(img)

    return new_img, orig_img


def match_to_cloud(data, recon, recon_idx, img, est_intrinsics=False):
    """
    Estimates the camera placement of an input image, and inserts this shot data into the reconstruction

    :param data: OpenSFM Dataset object
    :param recon: list of OpenSfM Reconstruction objects
    :param recon_idx: index of reconstruction
    :param img: name of input image
    :param est_intrinsics: bool value indicating if
    :return: None; modifies recon
    """
    # TODO remove use of recon_idx. We should match to entire cloud - LH  9/17/18

    # Load image features
    pt_img, f_img, _ = data.load_features(img)

    # Extract cloud points and features
    pts = recon[recon_idx].get_all_points()
    f_cld = np.array([pt.descriptor for pt in pts], dtype=f_img.dtype)
    xyz = np.array([pt.coordinates for pt in pts], dtype='float32')

    # TODO add checks to make sure that cloud and image features are the same type - LH 8/29/18
    # Load FLANN indices
    use_flann = data.config['matcher_type'] == 'FLANN'
    idx_img = data.load_feature_index(img, f_img) if use_flann else None
    idx_cld = data.load_reconstruction_feature_index(recon_idx, f_cld) if use_flann else None

    # Compute point matches with modified Lowe ratio
    # matches_init = matching.match_symmetric(fi=f_img, indexi=idx_img, pi=pt_img[:, :2],
    #                                         fj=f_cld, indexj=idx_cld, pj=xyz, config=data.config)
    # Compute point matches with original Lowe ratio test
    matches_init = matching.match_symmetric(fi=f_img, indexi=idx_img, fj=f_cld, indexj=idx_cld, config=data.config)

    # Extract camera data
    exif_data = data.load_exif(img)
    cam_list = data.load_camera_models()  # add cameras to reconstruction
    camera = cam_list[exif_data['camera']]

    # Use robust matching to prune down matches
    # TODO consider matching against many shots - LH 8/29/18
    for shot in list(recon[recon_idx].shots.values()):  # extract a shot
        if shot.id != img:
            break
    pt_cld = shot.project_many(points=xyz)
    matches = matching.robust_match(p1=pt_img, p2=pt_cld, camera1=camera, camera2=shot.camera,
                                    matches=matches_init, config=data.config)
    data.save_matches(image=img, matches={recon_idx: matches})  # save matches to file

    # VVVVV DBG VVVV
    # from bin.plot_matches import plot_matches
    # plot_matches(im1=data.load_image(img), im2=data.load_image(shot.id),
    #              p1=pt_img[matches[:, 0]], p2=pt_cld[matches[:, 1]])
    # ^^^^ DBG ^^^^^

    if len(matches) == 0:
        print("No feature matches made. Could not insert {}".format(img))
        return  # no matches made. stop processing

    # Extract 2D-3D point correspondences
    img_pts = pt_img[matches[:, 0], :2]
    obj_pts = xyz[matches[:, 1]].astype('float32')

    # Initialize camera model
    cam_mat = camera.get_K()
    dist_coeffs = np.array([camera.k1, camera.k2, 0.0, 0.0])

    # Estimate camera position
    shot, err = camera_align_outlier(obj_pts=obj_pts, img_pts=img_pts, cam_mat=cam_mat, dist_coeffs=dist_coeffs,
                                     rvec0=shot.pose.rotation, tvec0=shot.pose.translation)
    # Determine if the estimated camera is upside-down.
    view_dir = shot.viewing_direction()  # unit vector in the direction the camera is pointing
    cam_loc = shot.pose.get_origin()  # location of camera
    dist = np.dot(np.mean(obj_pts, axis=0) - cam_loc, view_dir)  # Distance from cam to cloud (in viewing direction)
    if dist < 0:  # Camera points away from cloud; modify the position
        shot.pose.set_origin(cam_loc + 2 * dist * view_dir)  # move the camera to the other side of the cloud
        # Re-estimate camera  TODO this doesn't work well. Find out why - LH 9/19/18
        # shot = camera_align_outlier(obj_pts=obj_pts, img_pts=img_pts, cam_mat=cam_mat, dist_coeffs=dist_coeffs,
        #                             rvec0=shot.pose.rotation, tvec0=shot.pose.translation)
    # Add shot to reconstruction
    if err > data.config['max_insert_err']:
        print("Image {} reprojection error {} too high to insert into cloud".format(img, err))  # do not insert image
    else:
        print("Adding image {} to cloud with reprojection error {}".format(img, err))  # do not insert image
        shot.id = img
        shot.camera = camera
        recon[recon_idx].shots[img] = shot  # add shot
        recon[recon_idx].cameras[camera.id] = camera  # add camera

    return shot


def camera_align_outlier(obj_pts, img_pts, cam_mat, dist_coeffs, rvec0=None, tvec0=None, ninlier=10, niter=100):
    """
    Perform camera calibration using robust outlier detection

    :param obj_pts: Nx3 array of (x,y,z) coordinates for N object points
    :param img_pts: Nx2 array of (x,y) pixel coordinates corresponding to each of N object points
    :param cam_mat: 3x3 intrinsic matrix of camera
    :param dist_coeffs: ndarray of distortion coefficients
    :param niter: number of Monte Carlo iterations
    :param ninlier: number of inliers to use for alignment
    :param rvec0: 3-element array initial guess for camera orientation
    :param tvec0: 3-element array initial guess for camera position
    :return shot: a Shot object of the estimated camera pose
    """
    # Check inputs
    if not isinstance(obj_pts, np.ndarray) or obj_pts.ndim != 2 or obj_pts.shape[1] != 3:
        raise IOError("Input obj_pts must be an Nx3 ndarray")
    npts = obj_pts.shape[0]
    if not isinstance(img_pts, np.ndarray) or img_pts.ndim != 2 or img_pts.shape[1] != 2:
        raise IOError("Input img_pts must be an Nx2 ndarray")
    if img_pts.shape[0] != npts:
        raise IOError("Input dimension mismatch")
    if not isinstance(cam_mat, np.ndarray) or cam_mat.ndim != 2 or cam_mat.shape != (3, 3):
        raise IOError("Input cam_mat must be a 3x3 ndarray")
    if not isinstance(dist_coeffs, np.ndarray) or dist_coeffs.ndim != 1 or len(dist_coeffs) < 4:
        raise IOError("Input dist_coeffs must be a 4 to 6 element ndarray")

    min_err, best_idx, best_rvec, best_tvec = np.inf, None, None, None  # initialize best inlier set and error
    if npts <= ninlier:  # too few points to use Monte Carlo. Estimate camera directly
        _, best_rvec, best_tvec = cv2.solvePnP(objectPoints=obj_pts, imagePoints=img_pts, cameraMatrix=cam_mat,
                                               distCoeffs=dist_coeffs,
                                               rvec=copy.deepcopy(rvec0), tvec=copy.deepcopy(tvec0),
                                               useExtrinsicGuess=(rvec0 is not None) and (tvec0 is not None))
        niter = 0  # Skip the Monte Carlo iterations below

    for _ in range(niter):  # loop over iterations
        # Generate random inlier/outlier set
        in_idx = np.random.choice(npts, ninlier, replace=False)  # generate inlier set
        mask_in = np.zeros(npts, dtype=bool)  # initialize mask
        mask_in[in_idx] = True  # generate inlier mask
        obj_in, img_in = obj_pts[mask_in], img_pts[mask_in]  # inlier object and image points
        obj_out, img_out = obj_pts[~mask_in], img_pts[~mask_in]  # outlier object and image points

        # Calibrate camera with inlier set
        _, rvec, tvec = cv2.solvePnP(objectPoints=obj_in, imagePoints=img_in, cameraMatrix=cam_mat,
                                       distCoeffs=dist_coeffs, rvec=copy.deepcopy(rvec0), tvec=copy.deepcopy(tvec0),
                                       useExtrinsicGuess=(rvec0 is not None) and (tvec0 is not None))
        # Reproject points
        proj_mat = np.dot(cam_mat, np.hstack((cv2.Rodrigues(rvec)[0], np.reshape(tvec, (3, 1)))))  # projection mat
        obj_out_homog = np.vstack((obj_out.T, np.ones(npts - ninlier)))  # convert to homogenous coordinates
        proj_out_homog = np.dot(proj_mat, obj_out_homog)  # project points to image plane
        proj_out = np.vstack((proj_out_homog[0] / proj_out_homog[2],  # convert from homogenous coordinates
                              proj_out_homog[1] / proj_out_homog[2])).T

        # Note that the openCV method below computes reprojection with lens distortion, but does not account for points
        # being behind the camera
        # proj_out, _ = cv2.projectPoints(objectPoints=obj_out, rvec=rvec, tvec=tvec,
        #                                 cameraMatrix=cam_mat, distCoeffs=dist_coeffs)  # project points to camera

        # Compute reprojection error
        proj_err = np.linalg.norm(np.squeeze(proj_out) - img_out, axis=1) # reprojection error
        proj_err[proj_out_homog[2] < 0] = np.inf  # If a point lies behind the camera, set the error to infinity
        med_err = np.median(proj_err)  # median error

        if med_err < min_err:  # update best inlier set
            min_err, best_idx, best_rvec, best_tvec = med_err, in_idx, rvec, tvec
    # TODO include step with more inlier selection - LH 8/17/18

    # Create shot object
    shot = Shot()
    shot.pose = Pose(rotation=np.squeeze(best_rvec), translation=np.squeeze(best_tvec))

    # Compute reprojection error
    proj_pts, _ = cv2.projectPoints(objectPoints=obj_pts, rvec=best_rvec, tvec=best_tvec,
                                    cameraMatrix=cam_mat, distCoeffs=dist_coeffs)  # project points to camera
    err = np.median(np.linalg.norm(np.squeeze(proj_pts) - img_pts, axis=1))
    return shot, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add new image to cloud.')
    parser.add_argument('--data', type=str, help='path to input data set')
    parser.add_argument('--overwrite_shots', default='False', help='flag to overwrite cameras in cloud')
    parser.add_argument('--new_img', type=str, default=None, help='name of image to process')
    args = parser.parse_args()


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    shutil.copy(args.new_img, args.data + 'images' + os.sep)
    fname = args.new_img.split(os.sep)[-1]

    shots = insert_img_cloud(data_dir=args.data, overwrite_shots=str2bool(args.overwrite_shots),
                             new_img=[fname] if args.new_img else None)

    print('Saving pose estimates')
    json.dump(shots, open('.'.join(fname.split('.')[:-1]) + '_shots.json', 'w'), sort_keys=True, indent=4)