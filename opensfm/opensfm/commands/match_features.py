import logging
from itertools import combinations, product
from timeit import default_timer as timer

from os.path import splitext
import numpy as np
import scipy.spatial as spatial

from opensfm import dataset
from opensfm import geo
from opensfm import io
from opensfm import log
from opensfm import matching
from opensfm.context import parallel_map


logger = logging.getLogger(__name__)


class Command:
    name = 'match_features'
    help = 'Match features between image pairs'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args, new_img=None):
        """
        Runs feature matching for an input dataset

        :param args: specifies dataset using argparse.namespace
        :param new_img: a list of new images to process. If None, then all images are processed.
        :return: None; writes to file
        """
        data = dataset.DataSet(args.dataset)
        all_img = data.images()
        if new_img is None:
            new_img = all_img  # assume no images have been processed if none are input
        exifs = {im: data.load_exif(im) for im in all_img}

        # Define pairing function
        pairs, preport = match_candidates_from_metadata(images=all_img, exifs=exifs, data=data, new_img=new_img,
                                                        special_match_method=data.config['special_matching_method'])

        # Ignore pairs not involving new images
        pairs = {img: pairs[img] for img in new_img}
        num_pairs = sum(len(c) for c in pairs.values())
        logger.info('Matching {} image pairs'.format(num_pairs))

        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs
        ctx.p_pre, ctx.f_pre = load_preemptive_features(data)
        args = list(match_arguments(pairs, ctx))

        start = timer()
        processes = ctx.data.config['processes']
        parallel_map(match, args, processes)
        end = timer()
        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('match_features: {0}\n'.format(end - start))
        self.write_report(data, preport, pairs, end - start)

    def write_report(self, data, preport, pairs, wall_time):
        pair_list = []
        for im1, others in pairs.items():
            for im2 in others:
                pair_list.append((im1, im2))

        report = {
            "wall_time": wall_time,
            "num_pairs": len(pair_list),
            "pairs": pair_list,
        }
        report.update(preport)
        data.save_report(io.json_dumps(report), 'matches.json')


class Context:
    pass


def load_preemptive_features(data):
    p, f = {}, {}
    if data.config['preemptive_threshold'] > 0:
        logger.debug('Loading preemptive data')
        for image in data.images():
            try:
                p[image], f[image] = \
                    data.load_preemtive_features(image)
            except IOError:
                p, f, c = data.load_features(image)
                p[image], f[image] = p, f
            preemptive_max = min(data.config['preemptive_max'],
                                 p[image].shape[0])
            p[image] = p[image][:preemptive_max, :]
            f[image] = f[image][:preemptive_max, :]
    return p, f


def has_gps_info(exif):
    return (exif and
            'gps' in exif and
            'latitude' in exif['gps'] and
            'longitude' in exif['gps'])


def has_here_true_info(exif):
    return (exif and has_gps_info(exif) and
            'img_direction' in exif and exif['img_direction'] is not None and
            'gps_time' in exif and exif['gps_time'] != 0)


def match_candidates_by_distance(images, exifs, reference, max_neighbors, max_distance):
    """Find candidate matching pairs by GPS distance."""
    if max_neighbors <= 0 and max_distance <= 0:
        return set()
    max_neighbors = max_neighbors or 99999999
    max_distance = max_distance or 99999999.
    k = min(len(images), max_neighbors + 1)

    points = np.zeros((len(images), 3))
    for i, image in enumerate(images):
        gps = exifs[image]['gps']
        alt = gps.get('altitude', 2.0)
        points[i] = geo.topocentric_from_lla(
            gps['latitude'], gps['longitude'], alt,
            reference['latitude'], reference['longitude'], reference['altitude'])

    tree = spatial.cKDTree(points)

    pairs = set()
    for i, image in enumerate(images):
        distances, neighbors = tree.query(
            points[i], k=k, distance_upper_bound=max_distance)
        for j in neighbors:
            if i != j and j < len(images):
                pairs.add(tuple(sorted((images[i], images[j]))))
    return pairs


def match_here_candidates_by_distance(drives, exifs, reference, max_neighbors, max_distance):
    """
    Find nearest neighbors for Here True drive images, subject to these criteria:
    1. Neighbor images must be from different drives
    2. Neighbor images must be facing approximately the same direction

    :param drives: list of image lists by drive
    :param exifs: dictionary of extracted EXIF data
    :param reference: (lat, lon, alt) of origin
    :param max_neighbors: maximum number of neighbors
    :param max_distance: maximum allowable distance between neighbors
    :return: set of selected image pairs
    """
    # Create single list of images
    images, drive_id = np.array([]), np.array([], dtype=int)
    for id, drive in enumerate(drives):
        images = np.append(images, drive)
        drive_id = np.append(drive_id, id * np.ones(len(drive), dtype=int))

    npts = len(images)
    points = np.zeros((npts, 3))
    for ii, image in enumerate(images):
        gps = exifs[image]['gps']
        alt = gps.get('altitude', 2.0)
        points[ii] = geo.topocentric_from_lla(gps['latitude'], gps['longitude'], alt,
                                              reference['latitude'], reference['longitude'], reference['altitude'])

    # Compute distance between all image pairs
    dist_mat = np.zeros((npts, npts))
    for ii in range(npts):
        for jj in range(ii, npts):
            if drive_id[ii] == drive_id[jj]:
                dist_mat[ii, jj] = np.inf  # disable pairing between images of the same drives
            elif image_direction_sep(img0=images[ii], img1=images[jj], exifs=exifs) > 45:
                dist_mat[ii, jj] = np.inf  # disable pairing between images pointing different directions
            else:
                dist_mat[ii, jj] = np.linalg.norm(points[ii] - points[jj])
    dist_mat[dist_mat > max_distance] = np.inf  # cap the distance between neighbors
    dist_mat += dist_mat.T  # account for distance symnmetry


    # Find the nearest neighbors
    pairs = set()  # initialize set of pairs
    idx = np.argsort(dist_mat, axis=1)
    for ii, img in enumerate(images):
        for jj in idx[ii, :max_neighbors]:  # Add max_neighbors closest neighbors
            # Do not add pair if the distance is inf (i.e. doesn't meet the pointing and drive criteria
            if not np.isinf(dist_mat[ii, jj]):
                pairs.add(tuple(sorted((images[ii], images[jj]))))
    return pairs



def match_candidates_by_time(images, exifs, max_neighbors, use_capture_time=True):
    """Find candidate matching pairs by time difference."""
    if max_neighbors <= 0:
        return set()
    k = min(len(images), max_neighbors + 1)

    times = np.zeros((len(images), 1))
    for i, image in enumerate(images):
        times[i] = exifs[image]['capture_time' if use_capture_time else 'gps_time']

    tree = spatial.cKDTree(times)

    pairs = set()
    for i, image in enumerate(images):
        distances, neighbors = tree.query(times[i], k=k)
        for j in neighbors:
            if i != j and j < len(images):
                pairs.add(tuple(sorted((images[i], images[j]))))
    return pairs


def match_candidates_by_order(images, max_neighbors):
    """Find candidate matching pairs by sequence order."""
    if max_neighbors <= 0:
        return set()
    n = (max_neighbors + 1) // 2

    pairs = set()
    for i, image in enumerate(images):
        a = max(0, i - n)
        b = min(len(images), i + n)
        for j in range(a, b):
            if i != j:
                pairs.add(tuple(sorted((images[i], images[j]))))
    return pairs


def match_candidates_from_metadata(images, exifs, data, special_match_method='none', new_img=[]):
    """
    Compute candidate matching pairs

    :param images: list of image names to process
    :param exifs: exif data processed from image files
    :param data: opensfm dataset
    :param new_img: subset of images to process first
    :return:
    """
    max_distance = data.config['matching_gps_distance']
    gps_neighbors = data.config['matching_gps_neighbors']
    time_neighbors = data.config['matching_time_neighbors']
    order_neighbors = data.config['matching_order_neighbors']
    special_match_method = special_match_method.lower()

    if not data.reference_lla_exists():
        data.invent_reference_lla()
    reference = data.load_reference_lla()

    if not all(map(has_gps_info, exifs.values())):
        if gps_neighbors != 0:
            logger.warn("Not all images have GPS info. "
                        "Disabling matching_gps_neighbors.")
        gps_neighbors = 0
        max_distance = 0

    images.sort()
    images = new_img + [img for img in images if img not in new_img]  # Place new images at the front of the processing

    if max_distance == gps_neighbors == time_neighbors == order_neighbors == 0:
        # All pair selection strategies deactivated so we match all pairs
        d = set()
        t = set()
        o = set()
        pairs = combinations(images, 2)
    elif special_match_method == 'none':
        d = match_candidates_by_distance(images, exifs, reference,
                                         gps_neighbors, max_distance)
        t = match_candidates_by_time(images, exifs, time_neighbors)
        o = match_candidates_by_order(images, order_neighbors)
        pairs = d | t | o
    elif special_match_method == 'video':
        images = np.array(images)
        gps_mask = np.array([has_gps_info(exifs[img]) for img in images])
        gps_images = images[gps_mask]
        no_gps_images = images[~gps_mask]

        d = match_candidates_by_distance(gps_images, exifs, reference,
                                         gps_neighbors, max_distance) if len(gps_images) else set()
        t = match_candidates_by_time(no_gps_images, exifs, time_neighbors)
        o = match_candidates_by_order(no_gps_images, order_neighbors)
        gps_no = set(product(gps_images, no_gps_images))  # compare GPS images to all other images
        pairs = d | t | o | gps_no
    elif special_match_method == 'here':
        # Sort images by time
        images = [img for img in images if has_here_true_info(exifs[img])]  # discard images without Here True data
        timestamp = np.array([exifs[i]['gps_time'] for i in images])  # extract timestamps
        idx = np.argsort(timestamp)
        images = np.array(images)[idx]
        timestamp = timestamp[idx]

        # Group images by drive based on their timestamp. Frames spaced by < 2 minutes are considered a single drive.
        # If more than 2 minutes between captures, then it is considered a different drive
        dt = np.hstack(([0], np.diff(timestamp)))  # time between frames
        dt_thresh = 2 * 60 * 1e6  # number of microseconds in 2 minutes
        drives = np.split(images, np.where(dt > dt_thresh)[0])

        t = set([])  # initialize drive
        for drive in drives:
            # Compute temporal neighbors among frames of this drive
            t0 = match_candidates_by_time(drive, exifs, time_neighbors)
            # Discard matches pointing opposite directions (angle of separation exceeds 135 degrees)
            t |= set([pair for pair in t0 if image_direction_sep(img0=pair[0], img1=pair[1], exifs=exifs) < 135])

        d = match_here_candidates_by_distance(drives, exifs, reference, gps_neighbors, max_distance)
        o = match_candidates_by_order(images, order_neighbors)
        pairs = d | t | o

    res = {im: [] for im in images}
    for im1, im2 in pairs:
        res[im1].append(im2)

    report = {
        "num_pairs_distance": len(d),
        "num_pairs_time": len(t),
        "num_pairs_order": len(o)
    }
    return res, report


def image_direction_sep(img0, img1, exifs):
    get_img_dir = lambda img: exifs[img]['img_direction'] * np.pi / 180  # convert image direction to radians
    # Convert image direction to unit vector in direction of pointing
    get_img_vec = lambda img: np.array([np.cos(get_img_dir(img)), np.sin(get_img_dir(img))])
    return np.arccos(np.dot(get_img_vec(img0), get_img_vec(img1))) * 180 / np.pi  # angle between vectors, in degrees


def match_arguments(pairs, ctx):
    for i, (im, candidates) in enumerate(pairs.items()):
        yield im, candidates, i, len(pairs), ctx


def match(args):
    """Compute all matches for a single image"""
    log.setup()

    im1, candidates, i, n, ctx = args
    logger.info('Matching {}  -  {} / {}'.format(im1, i + 1, n))

    config = ctx.data.config
    robust_matching_min_match = config['robust_matching_min_match']
    preemptive_threshold = config['preemptive_threshold']
    lowes_ratio = config['lowes_ratio']
    preemptive_lowes_ratio = config['preemptive_lowes_ratio']

    im1_matches = {}

    for im2 in candidates:
        # preemptive matching
        if preemptive_threshold > 0:
            t = timer()
            config['lowes_ratio'] = preemptive_lowes_ratio
            matches_pre = matching.match_lowe_bf(
                ctx.f_pre[im1], ctx.f_pre[im2], config)
            config['lowes_ratio'] = lowes_ratio
            logger.debug("Preemptive matching {0}, time: {1}s".format(
                len(matches_pre), timer() - t))
            if len(matches_pre) < preemptive_threshold:
                logger.debug(
                    "Discarding {0}/{1} based of preemptive matches {2} < {3}".format(
                        im1, im2, len(matches_pre), preemptive_threshold))
                continue

        # symmetric matching
        t = timer()
        p1, f1, c1 = ctx.data.load_features(im1)
        p2, f2, c2 = ctx.data.load_features(im2)

        if config['matcher_type'] == 'FLANN':
            i1 = ctx.data.load_feature_index(im1, f1)
            i2 = ctx.data.load_feature_index(im2, f2)
        else:
            i1 = None
            i2 = None

        matches = matching.match_symmetric(f1, i1, f2, i2, config)
        logger.debug('{} - {} has {} candidate matches'.format(
            im1, im2, len(matches)))
        if len(matches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue

        # robust matching
        t_robust_matching = timer()
        camera1 = ctx.cameras[ctx.exifs[im1]['camera']]
        camera2 = ctx.cameras[ctx.exifs[im2]['camera']]

        rmatches = matching.robust_match(p1, p2, camera1, camera2, matches,
                                         config)

        if len(rmatches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue
        im1_matches[im2] = rmatches
        logger.debug('Robust matching time : {0}s'.format(
            timer() - t_robust_matching))

        logger.debug("Full matching {0} / {1}, time: {2}s".format(
            len(rmatches), len(matches), timer() - t))
    ctx.data.save_matches(im1, im1_matches)
