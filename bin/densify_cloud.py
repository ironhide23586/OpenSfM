import os
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors

from opensfm.dataset import DataSet
from opensfm import commands
from opensfm.matching import match_symmetric, match_lowe_bf
from opensfm.reconstruction import triangulate_shot_features, paint_reconstruction
from opensfm.types import Reconstruction


mod = {module.Command().name: module for module in commands.opensfm_commands}


def densify_cloud(data_dir, epipolar_thresh=0.04):
    """
    Add point features to an already-created point cloud. New points are derived from the epipolar constraint of
    calibrated cameras
    :param data_dir: directory of data
    :param epipolar_thresh: threshold for distance a point must lie from the epipolar line. Threshold is for distances
    in normalized image coordinates, so the threshold is a fraction of the image size
    :return: None; modifies files
    """

    # Load data set
    if not os.path.exists(data_dir):
        raise IOError("Could not locate input directory {}".format(data_dir))
    data = DataSet(data_dir)

    # Load reconstruction
    recon = data.load_reconstruction()
    # Generate list of images in reconstruction
    all_img = []
    for partial in recon:
        all_img += partial.shots.keys()

    # Extract all features
    feat_pts, feat_desc = {}, {}  # initialize feature points and descriptors
    for img in all_img:
        feat_pts[img], feat_desc[img], _ = data.load_features(img)

    # Extract camera shot data
    shots = {img: get_shot(recon, img) for img in all_img}

    # Extract pairs by finding nearby shots in the reconstruction
    cam_pos = np.array([shots[img].pose.get_origin() for img in all_img])  # position of each camera
    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=data.config['matching_gps_neighbors'] + 1, algorithm='kd_tree').fit(cam_pos)
    _, nearest_idx = nbrs.kneighbors(cam_pos)
    pairs = {}
    for nn, img in enumerate(all_img):
        img_pairs = []
        for mm in nearest_idx[nn]:  # loop through neighbors of img
            nbr = all_img[mm]
            if nbr in pairs and img in pairs[nbr] or nbr == img:
                continue  # this pair has already been visited; skip it
            img_pairs.append(nbr)  # add neighbor to list
        if len(img_pairs) > 0:
            pairs[img] = img_pairs  # add pairs for this image

    # Modify reconstruction
    data.config['matcher_type'] = 'BRUTEFORCE'  # Use bruteforce matching with the epipolar constraint
    for img0 in pairs:  # loop over images
        matches = {}
        for img1 in pairs[img0]:  # loop over paired images
            fund_mat = shots[img0].get_fundamental_mat(shots[img1])  # compute fundamental matrix
            pts1 = np.hstack((feat_pts[img1][:, :2], np.ones((len(feat_pts[img1]), 1))))  # homogenous feat pts in img1
            matches[img1] = []
            for feat0_idx, (pt0, desc0) in enumerate(zip(feat_pts[img0], feat_desc[img0])):
                # Find the epipolar line in img1 corresponding to pt0. Matching point lies close to this line
                l1 = fund_mat.dot(np.hstack((pt0[:2], 1)))
                l1 /= np.linalg.norm(l1[:2])  # normalize as suggested in: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html

                # Find points close to this line
                mask = np.abs(pts1.dot(l1)) < epipolar_thresh

                # Match desciptors only among close points
                desc1 = feat_desc[img1][mask]  # extract descriptors of close features
                if len(desc1) == 0:
                    continue
                # Add matches
                mask_match = match_lowe_bf(np.reshape(desc0, (1, len(desc0))), desc1, config=data.config)
                feat1_idx = np.where(mask)[0][mask_match[:, 1]]  # feature indices from img1 that match
                matches[img1] += [(feat0_idx, f1) for f1 in feat1_idx]

                # VVV DBG VVVV
                # if len(mask_match) > 0:
                #     import matplotlib.pyplot as plt
                #     from opensfm import features
                #     im0, im1 = data.load_image(img0), data.load_image(img1)
                #     h0, w0, _ = im0.shape
                #     h1, w1, _ = im1.shape
                #     p0 = features.denormalized_image_coordinates(np.reshape(pt0, (1, 4)), w0, h0)
                #     p1 = features.denormalized_image_coordinates(pts1, w1, h1)
                #
                #     plt.figure()
                #     plt.imshow(im0)
                #     plt.plot(p0[0, 0], p0[0, 1], 'g.', markersize=15)
                #
                #     plt.figure()
                #     plt.imshow(im1)
                #     plt.plot(p1[mask, 0], p1[mask, 1], 'r.')
                #     plt.plot(p1[feat1_idx, 0], p1[feat1_idx, 1], 'go')
                #     plt.plot(p1[~mask, 0], p1[~mask, 1], 'b.')
                #
                #     plt.show()
                # ^^^^ DBG ^^^
            matches[img1] = np.array(matches[img1])  # convert to ndarray

        # Write out matches
        data.save_matches(img0, matches)

    # Create an updated tracks graph
    args = argparse.Namespace(dataset=data_dir)
    mod['create_tracks'].Command().run(args=args)
    graph = data.load_tracks_graph()

    # Create new reconstruction with the original camera estimates and the new point locations
    recon = Reconstruction()
    for img in all_img:
        if img not in graph:
            continue  # TODO maybe we should modify all_img to only include images in the graph - LH 9/14/18
        recon.add_shot(shots[img])
        recon.add_camera(shots[img].camera)
        triangulate_shot_features(graph, recon, img, data.config)  # add triangulated points to reconstruction

    paint_reconstruction(data, graph, recon)
    data.save_reconstruction([recon])


def get_shot(recon, img):
    if not isinstance(recon, list):
        raise IOError("Input recon must be a list")
    for partial in recon:  # loop over partial reconstructions
        shot = partial.get_shot(img)  # extract shot from this partial
        if shot is not None:
            break
    return shot


if __name__ == "__main__":
    import yaml
    from shutil import copyfile

    parser = argparse.ArgumentParser(description='Convert cloud to ORB')
    parser.add_argument('--data', type=str, help='path to input data set')
    parser.add_argument('--feat_type', type=str, default='ORB', help='type of feature')
    parser.add_argument('--lowes_ratio', type=float, default=0.65, help="Lowe's matching parameter")
    parser.add_argument('--matching_gps_neighbors', type=int, default=5, help="num of spatial neighbors for matching")
    parser.add_argument('--max_dist', type=float, default=50, help="maximum distance for a projected point")
    args = parser.parse_args()

    # Rename original input files so they are not re-used or overwritten
    data = DataSet(args.data)
    if os.path.exists(data._feature_path()):  # move features
        os.rename(data._feature_path(), '{}_orig'.format(data._feature_path()))
    if os.path.exists(data._matches_path()):  # move matches
        os.rename(data._matches_path(), '{}_orig'.format(data._matches_path()))
    if os.path.exists(data._tracks_graph_file()):  # move tracks graph
        os.rename(data._tracks_graph_file(), '{}_orig.csv'.format(os.path.splitext(data._tracks_graph_file())[0]))
    recon_path = data._reconstruction_file(None)
    if os.path.exists(recon_path):  # copy reconstruction file (read, overwritten during densification)
        copyfile(src=recon_path, dst='{}_orig.json'.format(os.path.splitext(recon_path)[0]))

    # Change config parameters
    data.config['feature_type'] = args.feat_type  # set feature type
    if args.feat_type == 'ORB':
        data.config['matcher_type'] = 'BRUTEFORCE'  # ORB features use bruteforce matching
        data.config['feature_min_frames'] = 20000  # extract ORB features densely
    data.config['lowes_ratio'] = args.lowes_ratio
    data.config['matching_gps_neighbors'] = args.matching_gps_neighbors
    data.config['triangulation_max_dist'] = args.max_dist

    # Write out config file
    if os.path.exists(data._config_path()):
        copyfile(src=data._config_path(), dst='{}_orig.yaml'.format(os.path.splitext(data._config_path())[0]))
    with open(data._config_path(), 'w') as outfile:
        yaml.dump(data.config, outfile, default_flow_style=False)

    # Re-compute features
    mod['detect_features'].Command().run(args=argparse.Namespace(dataset=args.data))

    # Run densification
    densify_cloud(data_dir=args.data)
