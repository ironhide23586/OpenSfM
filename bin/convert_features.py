import os
import cv2
import six
import numpy as np
import argparse

from opensfm.dataset import DataSet
from opensfm import features
from opensfm import commands

def convert_features(data_dir):
    """
    Converts the point cloud features to ORB

    :param data_dir: directory of data
    :return: None; updates reconstruction files
    """
    # Load data set
    if not os.path.exists(data_dir):
        raise IOError("Could not locate input directory {}".format(data_dir))
    data = DataSet(data_dir)
    data.config['feature_type'] = 'ORB'  # set to orb features

    # Load tracks graph
    graph = data.load_tracks_graph()

    for img_name in data.images():
        if img_name not in graph:
            continue  # image was not tracked
        img = data.load_image(img_name)  # read image
        ht, wd, _ = img.shape  # image size

        edges, feat_ary = [], []  # initialize lists
        for track_idx in graph[img_name]:
            # Extract tracked features in this image
            edge = graph.get_edge_data(track_idx, img_name)
            edges.append(edge)
            # Extract formatted array of normalized coordinates (x, y, scale, angle)
            feat_ary.append((edge['feature'][0], edge['feature'][1], edge['scale'], edge['orientation']))

        # Create features with extracted points
        p, descriptor, clr = features.extract_features(img, data.config, points=np.array(feat_ary))

        # Re-set the descriptors of each feature edges in the graph
        for edge, desc in zip(edges, descriptor):
            edge['descriptor'] = desc.tolist()

    # Save track graph
    data.save_tracks_graph(graph)

    # Update the descriptors in the reconstruction
    recon = data.load_reconstruction()
    for partial in recon:  # loop through partial reconstructions
        for trk_id, pt in partial.points.iteritems():  # loop through points in partial
            # TODO determine if we should duplicate points - LH 8/27/18
            pt.descriptor = six.next(six.itervalues(graph[trk_id]))['descriptor']  # re-set descriptor
    data.save_reconstruction(recon)  # save out modified cloud
