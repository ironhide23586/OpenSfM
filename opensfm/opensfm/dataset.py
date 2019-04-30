# -*- coding: utf-8 -*-

import os
import json
import logging
import pickle
import gzip

import cv2
import numpy as np
import networkx as nx
import six
import pandas as pd

from opensfm import io
from opensfm import config
from opensfm import context


logger = logging.getLogger(__name__)


class DataSet:
    """Accessors to the main input and output data.

    Data include input images, masks, and segmentation as well
    temporary data such as feature and matches and the final
    reconstructions.

    All data is stored inside a single folder with a specific subfolder
    structure.

    It is possible to store data remotely or in different formats
    by subclassing this class and overloading its methods.
    """
    def __init__(self, data_path):
        """Init dataset associated to a folder."""
        self.data_path = data_path
        self._load_config()
        self._load_image_list()
        self._load_mask_list()

    def _config_path(self):
        return os.path.join(self.data_path, 'config.yaml')

    def _load_config(self):
        config_file = self._config_path()
        self.config = config.load_config(config_file)

    def _load_image_list(self):
        """Load image list from image_list.txt or list image/ folder."""
        image_list_file = os.path.join(self.data_path, 'image_list.txt')
        if os.path.isfile(image_list_file):
            with io.open_rt(image_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_image_list(lines)
        else:
            self._set_image_path(os.path.join(self.data_path, 'images'))

    def images(self):
        """List of file names of all images in the dataset."""
        return self.image_list

    def _image_file(self, image):
        """Path to the image file."""
        return self.image_files[image]

    def open_image_file(self, image):
        """Open image file and return file object."""
        return open(self._image_file(image), 'rb')

    def load_image(self, image):
        """Load image pixels as numpy array.

        The array is 3D, indexed by y-coord, x-coord, channel.
        The channels are in RGB order.
        """
        return io.imread(self._image_file(image))

    def _undistorted_image_path(self):
        return os.path.join(self.data_path, 'undistorted')

    def _undistorted_image_file(self, image):
        """Path of undistorted version of an image."""
        return os.path.join(self._undistorted_image_path(), image + '.jpg')

    def load_undistorted_image(self, image):
        """Load undistorted image pixels as a numpy array."""
        return io.imread(self._undistorted_image_file(image))

    def save_undistorted_image(self, image, array):
        io.mkdir_p(self._undistorted_image_path())
        cv2.imwrite(self._undistorted_image_file(image), array[:, :, ::-1])

    def _load_mask_list(self):
        """Load mask list from mask_list.txt or list masks/ folder."""
        mask_list_file = os.path.join(self.data_path, 'mask_list.txt')
        if os.path.isfile(mask_list_file):
            with open(mask_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_mask_list(lines)
        else:
            self._set_mask_path(os.path.join(self.data_path, 'masks'))

    def load_mask(self, image):
        """Load image mask if it exists, otherwise return None."""
        if image in self.mask_files:
            mask_path = self.mask_files[image]
            mask = cv2.imread(mask_path)
            if mask is None:
                raise IOError("Unable to load mask for image {} "
                              "from file {}".format(image, mask_path))
            if len(mask.shape) == 3:
                mask = mask.max(axis=2)
        else:
            mask = None
        return mask

    def _undistorted_mask_path(self):
        return os.path.join(self.data_path, 'undistorted_masks')

    def _undistorted_mask_file(self, image):
        """Path of undistorted version of a mask."""
        return os.path.join(self._undistorted_mask_path(), image + '.png')

    def undistorted_mask_exists(self, image):
        """Check if the undistorted mask file exists."""
        return os.path.isfile(self._undistorted_mask_file(image))

    def load_undistorted_mask(self, image):
        """Load undistorted mask pixels as a numpy array."""
        return io.imread(self._undistorted_mask_file(image))

    def save_undistorted_mask(self, image, array):
        """Save the undistorted image mask."""
        io.mkdir_p(self._undistorted_mask_path())
        cv2.imwrite(self._undistorted_mask_file(image), array)

    def _segmentation_path(self):
        return os.path.join(self.data_path, 'segmentations')

    def _segmentation_file(self, image):
        return os.path.join(self._segmentation_path(), image + '.png')

    def load_segmentation(self, image):
        """Load image segmentation if it exitsts, otherwise return None."""
        segmentation_file = self._segmentation_file(image)
        if os.path.isfile(segmentation_file):
            segmentation = cv2.imread(segmentation_file)
            if len(segmentation.shape) == 3:
                segmentation = segmentation.max(axis=2)
        else:
            segmentation = None
        return segmentation

    def _undistorted_segmentation_path(self):
        return os.path.join(self.data_path, 'undistorted_segmentations')

    def _undistorted_segmentation_file(self, image):
        """Path of undistorted version of a segmentation."""
        return os.path.join(self._undistorted_segmentation_path(), image + '.png')

    def undistorted_segmentation_exists(self, image):
        """Check if the undistorted segmentation file exists."""
        return os.path.isfile(self._undistorted_segmentation_file(image))

    def load_undistorted_segmentation(self, image):
        """Load an undistorted image segmentation."""
        segmentation = cv2.imread(self._undistorted_segmentation_file(image))
        if len(segmentation.shape) == 3:
            segmentation = segmentation.max(axis=2)
        return segmentation

    def save_undistorted_segmentation(self, image, array):
        """Save the undistorted image segmentation."""
        io.mkdir_p(self._undistorted_segmentation_path())
        cv2.imwrite(self._undistorted_segmentation_file(image), array)

    def segmentation_ignore_values(self, image):
        """List of label values to ignore.

        Pixels with this labels values will be masked out and won't be
        processed when extracting features or computing depthmaps.
        """
        return self.config.get('segmentation_ignore_values', [])

    def load_segmentation_mask(self, image):
        """Build a mask from segmentation ignore values.

        The mask is non-zero only for pixels with segmentation
        labels not in segmentation_ignore_values.
        """
        ignore_values = self.segmentation_ignore_values(image)
        if not ignore_values:
            return None

        segmentation = self.load_segmentation(image)
        if segmentation is None:
            return None

        return self._mask_from_segmentation(segmentation, ignore_values)

    def load_undistorted_segmentation_mask(self, image):
        """Build a mask from the undistorted segmentation.

        The mask is non-zero only for pixels with segmentation
        labels not in segmentation_ignore_values.
        """
        ignore_values = self.segmentation_ignore_values(image)
        if not ignore_values:
            return None

        segmentation = self.load_undistorted_segmentation(image)
        if segmentation is None:
            return None

        return self._mask_from_segmentation(segmentation, ignore_values)

    def _mask_from_segmentation(self, segmentation, ignore_values):
        mask = np.ones(segmentation.shape, dtype=np.uint8)
        for value in ignore_values:
            mask &= (segmentation != value)
        return mask

    def load_combined_mask(self, image):
        """Combine binary mask with segmentation mask.

        Return a mask that is non-zero only where the binary
        mask and the segmentation mask are non-zero.
        """
        mask = self.load_mask(image)
        smask = self.load_segmentation_mask(image)
        return self._combine_masks(mask, smask)

    def load_undistorted_combined_mask(self, image):
        """Combine undistorted binary mask with segmentation mask.

        Return a mask that is non-zero only where the binary
        mask and the segmentation mask are non-zero.
        """
        mask = None
        if self.undistorted_mask_exists(image):
            mask = self.load_undistorted_mask(image)
        smask = None
        if self.undistorted_segmentation_exists(image):
            smask = self.load_undistorted_segmentation_mask(image)
        return self._combine_masks(mask, smask)

    def _combine_masks(self, mask, smask):
        if mask is None:
            if smask is None:
                return None
            else:
                return smask
        else:
            if smask is None:
                return mask
            else:
                return mask & smask

    def _depthmap_path(self):
        return os.path.join(self.data_path, 'depthmaps')

    def _depthmap_file(self, image, suffix):
        """Path to the depthmap file"""
        return os.path.join(self._depthmap_path(), image + '.' + suffix)

    def raw_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'raw.npz'))

    def save_raw_depthmap(self, image, depth, plane, score, nghbr, nghbrs):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'raw.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score, nghbr=nghbr, nghbrs=nghbrs)

    def load_raw_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'raw.npz'))
        return o['depth'], o['plane'], o['score'], o['nghbr'], o['nghbrs']

    def clean_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'clean.npz'))

    def save_clean_depthmap(self, image, depth, plane, score):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'clean.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score)

    def load_clean_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'clean.npz'))
        return o['depth'], o['plane'], o['score']

    def pruned_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'pruned.npz'))

    def save_pruned_depthmap(self, image, points, normals, colors, labels):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'pruned.npz')
        np.savez_compressed(filepath,
                            points=points, normals=normals,
                            colors=colors, labels=labels)

    def load_pruned_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'pruned.npz'))
        return o['points'], o['normals'], o['colors'], o['labels']

    def _is_image_file(self, filename):
        extensions = {'jpg', 'jpeg', 'png', 'tif', 'tiff', 'pgm', 'pnm', 'gif'}
        return filename.split('.')[-1].lower() in extensions

    def _set_image_path(self, path):
        """Set image path and find all images in there"""
        self.image_list = []
        self.image_files = {}
        if os.path.exists(path):
            for name in os.listdir(path):
                name = six.text_type(name)
                if self._is_image_file(name):
                    self.image_list.append(name)
                    self.image_files[name] = os.path.join(path, name)

    def _set_image_list(self, image_list):
        self.image_list = []
        self.image_files = {}
        for line in image_list:
            path = os.path.join(self.data_path, line)
            name = os.path.basename(path)
            self.image_list.append(name)
            self.image_files[name] = path

    def _set_mask_path(self, path):
        """Set mask path and find all masks in there"""
        self.mask_files = {}
        for image in self.images():
            filepath = os.path.join(path, image + '.png')
            if os.path.isfile(filepath):
                self.mask_files[image] = filepath

    def _set_mask_list(self, mask_list_lines):
        self.mask_files = {}
        for line in mask_list_lines:
            image, relpath = line.split(None, 1)
            path = os.path.join(self.data_path, relpath.strip())
            self.mask_files[image.strip()] = path

    def _exif_path(self):
        """Return path of extracted exif directory"""
        return os.path.join(self.data_path, 'exif')

    def _exif_file(self, image):
        """
        Return path of exif information for given image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._exif_path(), image + '.exif')

    def load_exif(self, image):
        """
        Return extracted exif information, as dictionary, usually with fields:

        ================  =====  ===================================
        Field             Type   Description
        ================  =====  ===================================
        width             int    Width of image, in pixels
        height            int    Height of image, in pixels
        focal_prior       float  Focal length (real) / sensor width
        ================  =====  ===================================

        :param image: Image name, with extension (i.e. 123.jpg)
        """
        with io.open_rt(self._exif_file(image)) as fin:
            return json.load(fin)

    def save_exif(self, image, data):
        io.mkdir_p(self._exif_path())
        with io.open_wt(self._exif_file(image)) as fout:
            io.json_dump(data, fout)

    def exif_exists(self, image):
        return os.path.isfile(self._exif_file(image))

    def feature_type(self):
        """Return the type of local features (e.g. AKAZE, SURF, SIFT)"""
        feature_name = self.config['feature_type'].lower()
        if self.config['feature_root']:
            feature_name = 'root_' + feature_name
        return feature_name

    def _feature_path(self):
        """Return path of feature descriptors and FLANN indices directory"""
        return os.path.join(self.data_path, "features")

    def _feature_file(self, image):
        """
        Return path of feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '.npz')

    def _save_features(self, filepath, image, points, descriptors, colors=None):
        io.mkdir_p(self._feature_path())
        feature_type = self.config['feature_type']
        if ((feature_type == 'AKAZE' and self.config['akaze_descriptor'] in ['MLDB_UPRIGHT', 'MLDB'])
                or (feature_type == 'HAHOG' and self.config['hahog_normalize_to_uchar'])
                or (feature_type == 'ORB')):
            feature_data_type = np.uint8
        else:
            feature_data_type = np.float32
        np.savez_compressed(filepath,
                            points=points.astype(np.float32),
                            descriptors=descriptors.astype(feature_data_type),
                            colors=colors)

    def features_exist(self, image):
        return os.path.isfile(self._feature_file(image))

    def load_features(self, image):
        feature_type = self.config['feature_type']
        s = np.load(self._feature_file(image))
        if feature_type == 'HAHOG' and self.config['hahog_normalize_to_uchar']:
            descriptors = s['descriptors'].astype(np.float32)
        else:
            descriptors = s['descriptors']
        return s['points'], descriptors, s['colors'].astype(float)

    def save_features(self, image, points, descriptors, colors):
        self._save_features(self._feature_file(image), image, points, descriptors, colors)

    def feature_index_exists(self, image):
        return os.path.isfile(self._feature_index_file(image))

    def feature_exists(self, image):
        return os.path.isfile(self._feature_file(image))

    def _feature_index_file(self, image):
        """
        Return path of FLANN index file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '.flann')

    def load_feature_index(self, image, features):
        index = context.flann_Index()
        index.load(features, self._feature_index_file(image))
        return index

    def load_reconstruction_feature_index(self, recon_idx, features):
        index = context.flann_Index()
        index.load(features, self._reconstruction_index_file(recon_idx))
        return index

    def save_feature_index(self, image, index):
        index.save(self._feature_index_file(image))

    def _preemptive_features_file(self, image):
        """
        Return path of preemptive feature file (a short list of the full feature file)
        for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '_preemptive' + '.npz')

    def load_preemtive_features(self, image):
        s = np.load(self._preemptive_features_file(image))
        return s['points'], s['descriptors']

    def save_preemptive_features(self, image, points, descriptors):
        self._save_features(self._preemptive_features_file(image), image, points, descriptors)

    def _matches_path(self):
        """Return path of matches directory"""
        return os.path.join(self.data_path, 'matches')

    def _matches_file(self, image):
        """File for matches for an image"""
        return os.path.join(self._matches_path(), '{}_matches.pkl.gz'.format(image))

    def matches_exists(self, image):
        return os.path.isfile(self._matches_file(image))

    def load_matches(self, image):
        with gzip.open(self._matches_file(image), 'rb') as fin:
            matches = pickle.load(fin)
        return matches

    def save_matches(self, image, matches):
        io.mkdir_p(self._matches_path())
        with gzip.open(self._matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)

    def find_matches(self, im1, im2):
        if self.matches_exists(im1):
            im1_matches = self.load_matches(im1)
            if im2 in im1_matches:
                return im1_matches[im2]
        if self.matches_exists(im2):
            im2_matches = self.load_matches(im2)
            if im1 in im2_matches:
                if len(im2_matches[im1]):
                    return im2_matches[im1][:, [1, 0]]
        return []

    def _tracks_graph_file(self, filename=None):
        """Return path of tracks file"""
        return os.path.join(self.data_path, filename or 'tracks.csv')

    def load_tracks_graph(self, filename=None):
        """Return graph (networkx data structure) of tracks"""
        return load_tracks_graph(self._tracks_graph_file(filename))

    def save_tracks_graph(self, graph, filename=None):
        save_tracks_graph(file_name=self._tracks_graph_file(filename), graph=graph)

    def load_undistorted_tracks_graph(self):
        return self.load_tracks_graph('undistorted_tracks.csv')

    def save_undistorted_tracks_graph(self, graph):
        return self.save_tracks_graph(graph, 'undistorted_tracks.csv')

    def _reconstruction_file(self, filename):
        """Return path of reconstruction file"""
        # print '---->', self.data_path, filename
        return os.path.join(self.data_path, filename or 'reconstruction.json')

    def _reconstruction_index_file(self, idx):
        """Return path of reconstruction index"""
        return os.path.join(self.data_path, 'reconstruction{}.json.flann'.format(idx))

    def _bound_box_file(self, filename):
        """Return path of bounding box file"""
        return os.path.join(self.data_path, filename or 'bound_box.txt')

    def reconstruction_exists(self, filename=None):
        return os.path.isfile(self._reconstruction_file(filename))

    def load_reconstruction(self, filename=None):
        with open(self._reconstruction_file(filename)) as fin:
            print '----->', fin
            reconstructions = io.reconstructions_from_json(json.load(fin))
        return reconstructions

    def save_reconstruction(self, reconstruction, filename=None, minify=False, ref_lla=None):
        with io.open_wt(self._reconstruction_file(filename)) as fout:
            io.json_dump(io.reconstructions_to_json(reconstruction, ref_lla=ref_lla), fout, minify)

    def save_reconstruction_flann(self, flann_idx, recon_idx):
        """
        Save reconstruction feature FLANN to file
        :param flann_idx: FLANN indexing for features
        :param recon_idx: index of reconstruction
        :return: None; saves to file
        """
        flann_idx.save(self._reconstruction_index_file(recon_idx))

    def save_bounding_box(self, reconstruction, filename=None):
        """Save bounding box to file"""
        import pymap3d as p3d  # TODO use the native tools of opensfm - LH 10/19/18

        # Found bounding box
        min_pt, max_pt = np.inf * np.ones(3), -np.inf * np.ones(3)
        for partial in reconstruction:
            min_part, max_part = partial.get_bound_box()
            min_pt = np.min((min_pt, min_part), axis=0)
            max_pt = np.max((max_pt, max_part), axis=0)

        # Convert bounding box to lat/lon
        ref_lla = self.load_reference_lla()
        min_lat, min_lon, _ = p3d.enu2geodetic(e=min_pt[0], n=min_pt[1], u=min_pt[1], lat0=ref_lla['latitude'],
                                               lon0=ref_lla['longitude'], h0=ref_lla['altitude'])
        max_lat, max_lon, _ = p3d.enu2geodetic(e=max_pt[0], n=max_pt[1], u=max_pt[1], lat0=ref_lla['latitude'],
                                               lon0=ref_lla['longitude'], h0=ref_lla['altitude'])
        with open(self._bound_box_file(filename), 'w+') as fout:
            fout.write(','.join(map(str, [min_lon, min_lat, max_lon, max_lat])))

    def load_undistorted_reconstruction(self):
        return self.load_reconstruction(
            filename='undistorted_reconstruction.json')

    def save_undistorted_reconstruction(self, reconstruction):
        return self.save_reconstruction(
            reconstruction, filename='undistorted_reconstruction.json')

    def _reference_lla_path(self):
        return os.path.join(self.data_path, 'reference_lla.json')

    def invent_reference_lla(self, images=None):
        lat, lon, alt = 0.0, 0.0, 0.0
        wlat, wlon, walt = 0.0, 0.0, 0.0
        if images is None: images = self.images()
        for image in images:
            d = self.load_exif(image)
            if 'gps' in d and 'latitude' in d['gps'] and 'longitude' in d['gps']:
                w = 1.0 / max(0.01, d['gps'].get('dop', 15))
                lat += w * d['gps']['latitude']
                lon += w * d['gps']['longitude']
                wlat += w
                wlon += w
                if 'altitude' in d['gps']:
                    alt += w * d['gps']['altitude']
                    walt += w
        if wlat: lat /= wlat
        if wlon: lon /= wlon
        if walt: alt /= walt
        reference = {'latitude': lat, 'longitude': lon, 'altitude': 0}  # Set altitude manually.
        self.save_reference_lla(reference)
        return reference

    def save_reference_lla(self, reference):
        with io.open_wt(self._reference_lla_path()) as fout:
            io.json_dump(reference, fout)

    def load_reference_lla(self):
        with io.open_rt(self._reference_lla_path()) as fin:
            return io.json_load(fin)

    def reference_lla_exists(self):
        return os.path.isfile(self._reference_lla_path())

    def _camera_models_file(self):
        """Return path of camera model file"""
        return os.path.join(self.data_path, 'camera_models.json')

    def load_camera_models(self):
        """Return camera models data"""
        with io.open_rt(self._camera_models_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models(self, camera_models):
        """Save camera models data"""
        with io.open_wt(self._camera_models_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _camera_models_overrides_file(self):
        """Path to the camera model overrides file."""
        return os.path.join(self.data_path, 'camera_models_overrides.json')

    def camera_models_overrides_exists(self):
        """Check if camera overrides file exists."""
        return os.path.isfile(self._camera_models_overrides_file())

    def load_camera_models_overrides(self):
        """Load camera models overrides data."""
        with io.open_rt(self._camera_models_overrides_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models_overrides(self, camera_models):
        """Save camera models overrides data"""
        with io.open_wt(self._camera_models_overrides_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _exif_overrides_file(self):
        """Path to the EXIF overrides file."""
        return os.path.join(self.data_path, 'exif_overrides.json')

    def exif_overrides_exists(self):
        """Check if EXIF overrides file exists."""
        return os.path.isfile(self._exif_overrides_file())

    def load_exif_overrides(self):
        """Load EXIF overrides data."""
        with io.open_rt(self._exif_overrides_file()) as fin:
            return json.load(fin)

    def profile_log(self):
        "Filename where to write timings."
        return os.path.join(self.data_path, 'profile.log')

    def _report_path(self):
        return os.path.join(self.data_path, 'reports')

    def load_report(self, path):
        """Load a report file as a string."""
        with open(os.path.join(self._report_path(), path)) as fin:
            return fin.read()

    def save_report(self, report_str, path):
        """Save report string to a file."""
        filepath = os.path.join(self._report_path(), path)
        io.mkdir_p(os.path.dirname(filepath))
        with io.open_wt(filepath) as fout:
            return fout.write(report_str)

    def _navigation_graph_file(self):
        "Return the path of the navigation graph."
        return os.path.join(self.data_path, 'navigation_graph.json')

    def save_navigation_graph(self, navigation_graphs):
        with io.open_wt(self._navigation_graph_file()) as fout:
            io.json_dump(navigation_graphs, fout)

    def _ply_file(self, filename):
        return os.path.join(self.data_path, filename or 'reconstruction.ply')

    def save_ply(self, reconstruction, filename=None,
                 no_cameras=False, no_points=False):
        """Save a reconstruction in PLY format."""
        ply = io.reconstruction_to_ply(reconstruction, no_cameras, no_points)
        with io.open_wt(self._ply_file(filename)) as fout:
            fout.write(ply)

    def _ground_control_points_file(self):
        return os.path.join(self.data_path, 'gcp_list.txt')

    def ground_control_points_exist(self):
        return os.path.isfile(self._ground_control_points_file())

    def load_ground_control_points(self):
        """Load ground control points.

        It uses reference_lla to convert the coordinates
        to topocentric reference frame.
        """
        exif = {image: self.load_exif(image) for image in self.images()}

        with open(self._ground_control_points_file()) as fin:
            return io.read_ground_control_points_list(
                fin, self.load_reference_lla(), exif)

    def image_as_array(self, image):
        logger.warning("image_as_array() is deprecated. Use load_image() instead.")
        return self.load_image(image)

    def undistorted_image_as_array(self, image):
        logger.warning("undistorted_image_as_array() is deprecated. "
                       "Use load_undistorted_image() instead.")
        return self.load_undistorted_image(image)

    def mask_as_array(self, image):
        logger.warning("mask_as_array() is deprecated. Use load_mask() instead.")
        return self.load_mask(image)


def load_tracks_graph(file_name):
    g = nx.Graph()  # initialize graph
    df = pd.read_csv(file_name)  # load csv data to pandas dataframe
    pt_values = ['image', 'track_id', 'feat_id', 'x', 'y', 'scale', 'orinetation', 'red', 'green', 'blue']  # .csv data

    # Combine columns
    desc_len = len(df.columns.values) - len(pt_values)  # length of descriptor
    args = [df[str(ky)] for ky in xrange(desc_len)]
    df['desc'] = zip(*args)  # bundle tuple for descriptors
    df['xy'] = zip(df.x, df.y)  # bundle tuple for feature (x,y)
    df['color'] = zip(df.red, df.green, df.blue)  # bundle tuple for feature color

    for idx, trk_pt in df.iterrows():  # copy all
        # TODO This is too similar to the code in matching.create_tracks_graph(). Create subroutine - LH 8/16/18
        g.add_node(trk_pt['image'], bipartite=0)
        g.add_node(trk_pt['track_id'], bipartite=1)
        g.add_edge(trk_pt['image'], trk_pt['track_id'], feature=trk_pt.xy, feature_id=trk_pt.feat_id,
                   feature_color=trk_pt.color, descriptor=trk_pt.desc, scale=trk_pt.scale,
                   orientation=trk_pt.orientation)
    return g


def save_tracks_graph(file_name, graph):
    """
    Writes track graph to .csv file

    :param file_name: file to write to
    :param graph: track graph
    :return: None; writes to file
    """
    if not isinstance(graph, nx.Graph):
        raise IOError("Input graph must be a Graph object")

    # Initilize data array of named columns
    track_data = {'image': [], 'track_id': [], 'feat_id': [], 'x': [], 'y': [], 'scale': [], 'orientation': [],
                  'red': [], 'green': [], 'blue': []}
    desc = []  # initialize a list of descriptors
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                track_data['image'].append(image)
                track_data['track_id'].append(track)
                track_data['feat_id'].append(data['feature_id'])
                track_data['scale'].append(data['scale'])
                track_data['orientation'].append(data['orientation'])
                xx, yy = data['feature']
                rr, gg, bb = data['feature_color']
                track_data['x'].append(xx)
                track_data['y'].append(yy)
                # TODO consider converting color from float to int - LH 8/14/18
                track_data['red'].append(rr)
                track_data['green'].append(gg)
                track_data['blue'].append(bb)
                desc.append(data['descriptor'])
    # Create dataframe from the two data structures
    df = pd.concat((pd.DataFrame(track_data), pd.DataFrame(np.array(desc))), axis=1)
    df.to_csv(file_name, index=False)  # write to file