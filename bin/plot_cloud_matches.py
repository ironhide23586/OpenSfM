import argparse
import numpy as np
import matplotlib.pyplot as plt

from opensfm import dataset
from plot_matches import plot_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot matches between images')
    parser.add_argument('dataset',
                        help='path to the dataset to be processed')
    parser.add_argument('--image',
                        help='show tracks for a specific')
    parser.add_argument('--ref_image',
                        help='show tracks to points viewed at this image')
    args = parser.parse_args()

    # Load data
    data = dataset.DataSet(args.dataset)
    images = data.images()
    if args.image not in images:
        raise IOError("Did not find input {} in image set {}".format(args.image, args.dataset))
    if args.ref_image not in images:
        raise IOError("Did not find input {} in image set {}".format(args.ref_image, args.dataset))
    img_in = data.load_image(args.image)  # input image
    img_ref = data.load_image(args.ref_image)  # reference image (raw pixels)

    # Load matches
    matches = data.load_matches(args.image)
    idx = np.where([isinstance(k, int) for k in matches.keys()])[0]  # find the reconstruction index
    if len(idx) == 0:
        raise IOError("Input image {} was not registered to the point cloud".format(args.image))
    recon_idx = idx[0]
    matches = matches[recon_idx]

    # Load features
    feat, _, _ = data.load_features(args.image)

    # Load point cloud
    recon = data.load_reconstruction()[recon_idx]
    xyz = np.array([pt.coordinates for pt in recon.get_all_points()], dtype='float32')

    # Create point cloud image
    img_cld = recon.display_point_cloud(args.ref_image)  # render cloud from reference

    # Extract positions of matches
    shot = recon.shots[args.ref_image]
    pt_cld = shot.project_many(xyz[matches[:, 1]])  # find match poitns at reference
    pt_img = feat[matches[:, 0], :2]
    # Mask points that fall out of image points
    mask = np.all((pt_cld > -0.5) & (pt_cld < 0.5), axis=1)
    pt_cld, pt_img = pt_cld[mask], pt_img[mask]

    # Plot the matches to the rendered point cloud at reference image
    plt.figure()
    plot_matches(img_in, img_cld, pt_img, pt_cld)

    # Plot matches to the raw reference image
    plt.figure()
    plot_matches(img_in, img_ref, pt_img, pt_cld)
    plt.show()
