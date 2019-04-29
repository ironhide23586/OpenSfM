import numpy as np
import cv2
import pyopengv
import networkx as nx
import logging
from collections import defaultdict
from itertools import combinations

from six import iteritems

from opensfm import context
from opensfm import multiview
from opensfm.unionfind import UnionFind


logger = logging.getLogger(__name__)


# pairwise matches
def match_lowe(index, f2, config):
    """Match features and apply Lowe's ratio filter.

    Args:
        index: flann index if the first image
        f2: feature descriptors of the second image
        config: config parameters
    """
    search_params = dict(checks=config['flann_checks'])
    results, dists = index.knnSearch(f2, 2, params=search_params)
    squared_ratio = config['lowes_ratio']**2  # Flann returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    matches = list(zip(results[good, 0], good.nonzero()[0]))
    return np.array(matches, dtype=int)

def match_symmetric(fi, indexi, fj, indexj, config, pi=None, pj=None):
    """Match in both directions and keep consistent matches.

    Args:W
        fi: feature descriptors of the first image
        indexi: flann index if the first image
        fj: feature descriptors of the second image
        indexj: flann index of the second image
        config: config parameters
        pi: locations of features in first image (only supported for bruteforce matching)
        pj: locations of features in second image (only supported for bruteforce matching)
    """
    if config['matcher_type'] == 'FLANN':
        matches_ij = [(a, b) for a, b in match_lowe(indexi, fj, config)]
        matches_ji = [(b, a) for a, b in match_lowe(indexj, fi, config)]
    else:
        matches_ij = [(a, b) for a, b in match_lowe_bf(f1=fi, f2=fj, p2=pj, config=config)]
        matches_ji = [(b, a) for a, b in match_lowe_bf(f1=fj, f2=fi, p2=pi, config=config)]

    matches = set(matches_ij).intersection(set(matches_ji))
    return np.array(list(matches), dtype=int)


def _convert_matches_to_vector(matches):
    """Convert Dmatch object to matrix form."""
    matches_vector = np.zeros((len(matches), 2), dtype=np.int)
    k = 0
    for mm in matches:
        matches_vector[k, 0] = mm.queryIdx
        matches_vector[k, 1] = mm.trainIdx
        k = k+1
    return matches_vector


def match_lowe_bf(f1, f2, config, p2=None):
    """Bruteforce matching and Lowe's ratio filtering. Optionally, the positions, p2, of features in the second image
    may be input. In this case, the second-closeset match for Lowe's ratio is a feature which is located sufficiently
    far from the closeset match

    Args:
        f1: feature descriptors of the first image
        f2: feature descriptors of the second image
        p2: feature locations in the second image (if None, then
        config: config parameters
    """
    pts_in = p2 is not None  # bool indicating if points were input
    NNBR = 10 if pts_in else 2  # number  of neighbors to extract from KNN matching

    if pts_in:
        _, ndim = np.shape(p2)  # number of dimensions of the spatial points
        THRESH = 5.0 if ndim == 3 else 0.03  # TODO add to config file - LH 9/17/18

    assert(f1.dtype.type == f2.dtype.type)
    if (f1.dtype.type == np.uint8):
        matcher_type = 'BruteForce-Hamming'
    else:
        matcher_type = 'BruteForce'
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matches = matcher.knnMatch(f1, f2, k=NNBR)  # extract 10 best matches

    ratio = config['lowes_ratio']
    good_matches = []
    for match in matches:
        if not (match and len(match) >= 2):
            continue
        if pts_in:  # discard features that are too close to the first
            # compute spatial distance between the top matching feature and all other features
            feat_idx = _convert_matches_to_vector(match)[:, 1]
            spatial_dist = np.linalg.norm(p2[feat_idx] - p2[feat_idx[0]], axis=1)

            # ignore features that are close
            mask = spatial_dist >= THRESH
            mask[0] = True  # keep the top match
            match = np.array(match)[mask]
            if len(match) < 2:
                continue  # too few matches remaining
        m, n = match[:2]
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    good_matches = _convert_matches_to_vector(good_matches)
    return np.array(good_matches, dtype=int)


def robust_match_fundamental(p1, p2, matches, config):
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()

    FM_RANSAC = cv2.FM_RANSAC if context.OPENCV3 else cv2.cv.CV_FM_RANSAC
    threshold = config['robust_matching_threshold']
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
    inliers = mask.ravel().nonzero()

    if F is None or F[2, 2] == 0.0:
        return []

    return matches[inliers]


def _compute_inliers_bearings(b1, b2, T, threshold=0.01):
    R = T[:, :3]
    t = T[:, 3]
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = multiview.vector_angle_many(br1, b1) < threshold
    ok2 = multiview.vector_angle_many(br2, b2) < threshold
    return ok1 * ok2


def robust_match_calibrated(p1, p2, camera1, camera2, matches, config):
    """Filter matches by estimating the Essential matrix via RANSAC."""

    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    threshold = config['robust_matching_calib_threshold']
    T = multiview.relative_pose_ransac(
        b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000, 0.999)

    inliers = _compute_inliers_bearings(b1, b2, T, threshold)

    return matches[inliers]


def robust_match(p1, p2, camera1, camera2, matches, config):
    """Filter matches by fitting a geometric model.

    If cameras are perspective without distortion, then the Fundamental
    matrix is used.  Otherwise, we use the Essential matrix.
    """
    if (camera1.projection_type == 'perspective'
            and camera1.k1 == 0.0 and camera1.k2 == 0.0
            and camera2.projection_type == 'perspective'
            and camera2.k1 == 0.0 and camera2.k2 == 0.0):
        return robust_match_fundamental(p1, p2, matches, config)
    else:
        return robust_match_calibrated(p1, p2, camera1, camera2, matches, config)


def _good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


def create_tracks_graph(features, colors, descriptors, matches, config, tracks_graph=nx.Graph()):
    """
    Link matches into tracks

    :param features: dictionary, keyed by image, to a list of feature coords, scale and orientation (x,y,scale,theta)
    :param colors: dictionary, keyed by image, to a list of feature pixel color (r,g,b)
    :param descriptors: dictionary, keyed by image, to a list of feature descriptors
    :param matches: dictionary, keyed by image pair tuple, to a list of matched feature indices
    :param config: dictionary of config settings
    :param tracks_graph: nx.Graph object (can be from existing track)
    :return: updated tracks_graph
    """
    logger.debug('Merging features onto existing tracks' if len(tracks_graph) else 'Merging features into tracks')

    # Re-create union-find for original graph
    uf = UnionFind()
    max_trk = -1
    feat_trk = {}  # mapping from feature to the track index
    for trk in tracks_graph.node:
        if not tracks_graph.node[trk]['bipartite']:
            continue  # skip image nodes
        # Extract an exhaustive list of (image, feature) pairs within the track
        feat_set = tuple([(img, tracks_graph.get_edge_data(img, trk)['feature_id']) for img in tracks_graph[trk]])
        uf.union(*feat_set)  # Update union find
        for feat in feat_set:
            feat_trk[feat] = trk
        max_trk = max(max_trk, int(trk))  # update the max track index

    # Append new matches to union-find
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    # Stitch features matches into tracks
    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    # Create list of tracks
    tracks = []
    for t in sets.values():
        if not _good_track(t, config['min_track_length']):
            continue  # skip short tracks

        # Assign track_id to this track
        track_id = None
        for feat in t:
            if feat in feat_trk:
                track_id = feat_trk[feat]
                break
        if track_id is None:  # This is a new track, assign it a new index
            max_trk += 1  # increment index
            track_id = str(max_trk)  # assign track ID
        tracks.append((track_id, t))  # append this track to list of tracks
    logger.debug('Good tracks: {}'.format(len(tracks)))

    for track_id, track in tracks:
        for image, featureid in track:
            if (image not in features) or tracks_graph.has_edge(image, track_id):
                continue
            x, y, scale, theta = features[image][featureid]  # parse the feature data
            r, g, b = colors[image][featureid]
            tracks_graph.add_node(image, bipartite=0)
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(image,
                                  str(track_id),
                                  feature=(x, y),
                                  feature_id=featureid,
                                  feature_color=(float(r), float(g), float(b)),
                                  descriptor=tuple(descriptors[image][featureid]),
                                  scale=scale, orientation=theta)

    return tracks_graph

def tracks_and_images(graph):
    """List of tracks and images in the graph."""
    tracks, images = [], []
    for n in graph.nodes(data=True):
        if n[1]['bipartite'] == 0:
            images.append(n[0])
        else:
            tracks.append(n[0])
    return tracks, images


def common_tracks(graph, im1, im2):
    """List of tracks observed in both images.

    Args:
        graph: tracks graph
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1, t2 = graph[im1], graph[im2]
    tracks, p1, p2 = [], [], []
    for track in t1:
        if track in t2:
            p1.append(t1[track]['feature'])
            p2.append(t2[track]['feature'])
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


def all_common_tracks(graph, tracks, include_features=True, min_common=50):
    """List of tracks observed by each image pair.

    Args:
        graph: tracks graph
        tracks: list of track identifiers
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    track_dict = defaultdict(list)
    for track in tracks:
        track_images = sorted(graph[track].keys())
        for im1, im2 in combinations(track_images, 2):
            track_dict[im1, im2].append(track)

    common_tracks = {}
    for k, v in iteritems(track_dict):
        if len(v) < min_common:
            continue
        im1, im2 = k
        if include_features:
            p1 = np.array([graph[im1][tr]['feature'] for tr in v])
            p2 = np.array([graph[im2][tr]['feature'] for tr in v])
            common_tracks[im1, im2] = (v, p1, p2)
        else:
            common_tracks[im1, im2] = v
    return common_tracks
