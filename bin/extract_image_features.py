"""Module extracts ORB features from an input image and writes to .npz file.  All code lifted from openSfM;
namely files io.py, features.py and detect_features.py"""
import os
import cv2
import numpy as np


def detect_features(img, nfeat, hahog_params=None):
    """
    Extracts ORB features from image

    :param img: MxNx3 RGB pixel array
    :param nfeat: desired number of features
    :param hahog_params: parameters for HAHOG calculation (if None, ORB features used)
    :return pts: Kx4 array of (x, y, scale, angle) data for each of K key points
    (x, y) is the position of the feature in normalized image coordinates (values in [0, 1])
    scale of the feature, in pixels
    angle is the orientation of the feature, in degrees
    :return desc: Kx32 array of 32-element ORB descriptors at each of K key points
    """
    if hahog_params is None:  # Extract ORB features
        detector = cv2.ORB_create(nfeatures=nfeat)  # define ORB detector
        pts = detector.detect(img)  # find keypoints
        pts, desc = detector.compute(img, pts)  # compute desciptors at key points
        pts = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in pts])  # reformat points to array
    else:  # Extract HAHOG features
        from opensfm import csfm  # This is a C library that is part of the opensfm installation
        pts, desc = csfm.hahog(img.astype(np.float32) / 255,  # VlFeat expects pixel values between 0, 1
                               peak_threshold=hahog_params['hahog_peak_threshold'],
                               edge_threshold=hahog_params['hahog_edge_threshold'],
                               target_num_features=nfeat,
                               use_adaptive_suppression=hahog_params['use_adapt_sup'])
        if hahog_params['feature_root']:
            desc = np.sqrt(desc)
            uchar_scaling = 362  # x * 512 < 256  =>  sqrt(x) * 362 < 256
        else:
            uchar_scaling = 512

        if hahog_params['hahog_normalize_to_uchar']:
            desc = (uchar_scaling * desc).clip(0, 255).round()

    # Normalize image coordinates
    pts[:, :2] = normalized_image_coordinates(pts[:, :2], width=img.shape[1], height=img.shape[0])

    # Return keypoints sorted by scale
    idx = np.argsort(pts[:, 2])  # points[:, 2] is the keypoint scale
    return pts[idx], desc[idx]


def normalized_image_coordinates(pixel_coords, width, height):
    """
    Normalize input (x, y) coordinates by the longest image side length

    :param pixel_coords: Kx4 array of (x, y, scale, angle) data for each of K key points
    :param width: image width
    :param height: image height
    :return: Kx2 array of normalized (x, y) coordinates for each of K key points
    """
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p


def imread(img_path, max_size):
    """
    Read image pixels from file

    :param img_path: path to image file
    :param max_size: desired maximum side length of image
    :return: MxN grayscale pixel array of resized image
    """
    if not os.path.exists(img_path):
        raise IOError("Could not find file {}".format(img_path))
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION  # options for file read
    img_rgb = cv2.imread(img_path, flags)[:, :, ::-1]  # Read BGR image and conver to RGB
    return cv2.cvtColor(resized_image(img_rgb, max_size), cv2.COLOR_RGB2GRAY)  # resize image and convert to grayscale


def resized_image(img, max_size=-1):
    """
    Resize img to indicated process size

    :param img: M0xN0x3 RGB pixel array
    :param max_size: int value indicating the maximum side length of the image (in pixels)
    :return: MxNx3 RGB pixel array of resized image
    """

    # Check inputs
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise IOError("Input img must be MxNx3 ndarray")
    if not isinstance(max_size, int):
        raise IOError("Input max size must be an int value")

    # Resize image
    h, w, _ = img.shape
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Add new image to cloud.')
    # Define IO parameters
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    parser.add_argument('--feat_type', type=str, required=True, choices=['ORB', 'HAHOG'], help='Type of feature to use')
    parser.add_argument('--output', type=str, required=True, help='Path to output .npz file')

    # Define optional parameters
    parser.add_argument('--feature_process_size', type=int, default=2048,
                        help='Resize the image if its size is larger than specified. Set to -1 for original size')
    parser.add_argument('--feature_min_frames', type=int, default=4000,
                        help='If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.')

    # Define HOG-specific parameters
    parser.add_argument('--hahog_peak_threshold', type=float, default=1e-5, help='parameter for HAHOG features')
    parser.add_argument('--hahog_edge_threshold', type=int, default=10, help='parameter for HAHOG features')
    parser.add_argument('--hahog_normalize_to_uchar', type=bool, default=False, help='parameter for HAHOG features')
    parser.add_argument('--feature_use_adaptive_suppression', type=bool, default=False,
                        help='parameter for HAHOG features')
    parser.add_argument('--feature_root', type=bool, default=True, help='True = apply square root mapping to features')
    args = parser.parse_args()

    # Read image
    image = imread(img_path=args.image, max_size=args.feature_process_size)

    # Set feature parameters
    hahog_params = None if args.feat_type == 'ORB' else {'hahog_peak_threshold': args.hahog_peak_threshold,
                                                         'hahog_edge_threshold': args.hahog_edge_threshold,
                                                         'hahog_normalize_to_uchar': args.hahog_normalize_to_uchar,
                                                         'use_adapt_sup': args.feature_use_adaptive_suppression,
                                                         'feature_root': args.feature_root}
    # Extract features
    points, descriptors = detect_features(img=image, nfeat=args.feature_min_frames, hahog_params=hahog_params)

    # Write output file
    dtype = np.uint8 if args.feat_type == 'ORB' else np.float32
    np.savez_compressed(args.output, points=points.astype(np.float32), descriptors=descriptors.astype(dtype))
