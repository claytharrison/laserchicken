"""Calculate echo ratio.

See https://github.com/eEcoLiDAR/eEcoLiDAR/issues/21
"""

import numpy as np

from collections import OrderedDict

from laserchicken.feature_extractor.base_feature_extractor import FeatureExtractor
from laserchicken.utils import get_attributes_per_neighborhood
from laserchicken.keys import point_classes

#used to set value returned for ratios when there are no points in the neighborhood
DIVIDE_BY_ZERO_VALUE = np.nan

def _to_unmasked_array(masked_array):
    """Creates a 'normal' numpy array from a masked array, inputting nans for masked values."""
    data = masked_array.data
    data[masked_array.mask] = np.nan
    return data


class FilteredBandRatioFeatureExtractor(FeatureExtractor):
    """Feature extractor for the point density."""

    def __init__(self, lower_limit, upper_limit, data_key='z', attribute_key = 'classification', attribute_values = 'all'):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.data_key = data_key
        self.attribute_key = attribute_key
        if attribute_values == 'all':
            self.attribute_values = list(OrderedDict.fromkeys(point_classes.keys()))
        else:
            self.attribute_values = list(OrderedDict.fromkeys(attribute_values))

    def requires(self):
        """
        Get a list of names of the point attributes that are needed for this feature extraction.

        For simple features, this could be just x, y, and z. Other features can build on again
        other features to have been computed first.

        :return: List of feature names
        """
        return []

    def provides(self):
        """
        Get a list of names of the feature values.

        This will return as many names as the number feature values that will be returned.
        For instance, if a feature extractor returns the first 3 Eigen values, this method
        should return 3 names, for instance 'eigen_value_1', 'eigen_value_2' and 'eigen_value_3'.

        :return: List of feature names
        """
        n_points_band = 'band_count_'
        if self.lower_limit is not None:
            n_points_band += str(self.lower_limit) + '_'
        n_points_band += self.data_key
        if self.upper_limit is not None:
            n_points_band += '_' + str(self.upper_limit)
            
        n_points_band_class = 'band_count_'
        if self.lower_limit is not None:
            n_points_band_class += str(self.lower_limit) + '_'
        n_points_band_class += self.data_key
        if self.upper_limit is not None:
            n_points_band_class += '_' + str(self.upper_limit)
        if self.attribute_values is not None:
            if self.attribute_values == list(OrderedDict.fromkeys(point_classes.keys())):
                n_points_band_class += '_all_classes'
            else:
                for pt_class in list(OrderedDict.fromkeys([point_classes[x] for x in self.attribute_values])):
                    n_points_band_class += '_' + pt_class
                    
        ratio_band = 'band_ratio_'
        if self.lower_limit is not None:
            ratio_band += str(self.lower_limit) + '_'
        ratio_band += self.data_key
        if self.upper_limit is not None:
            ratio_band += '_' + str(self.upper_limit)
                    
        ratio_band_class = 'band_ratio_'
        if self.lower_limit is not None:
            ratio_band_class += str(self.lower_limit) + '_'
        ratio_band_class += self.data_key
        if self.upper_limit is not None:
            ratio_band_class += '_' + str(self.upper_limit)
        if self.attribute_values is not None:
            if self.attribute_values == list(OrderedDict.fromkeys(point_classes.keys())):
                ratio_band_class += '_all_classes'
            else:
                for pt_class in list(OrderedDict.fromkeys([point_classes[x] for x in self.attribute_values])):
                    ratio_band_class += '_' + pt_class
            
        
        return_names = [n_points_band, n_points_band_class, ratio_band, ratio_band_class]
        return return_names

    def extract(self, point_cloud, neighborhoods, target_point_cloud, target_index, volume_description):
        """
        Extract the feature value(s) of the point cloud at location of the target.

        :param point_cloud: environment (search space) point cloud
        :param neighborhoods: array of array of indices of points within the point_cloud argument
        :param target_point_cloud: point cloud that contains target point
        :param target_index: index of the target point in the target point cloud
        :param volume_description: volume object that describes the shape and size of the search volume
        :return: feature value
        """
        supported_volumes = ['infinite cylinder', 'cell']
        if volume_description.TYPE not in supported_volumes:
            raise ValueError('The volume must be a cylinder')

        attribute = get_attributes_per_neighborhood(point_cloud, neighborhoods, [self.data_key])
        z = attribute[:, 0, :]
        n_total_points = attribute.shape[2]
        n_masked_points_per_neighborhood = attribute.mask[:, 0, :].sum(axis=1)
        n_points_per_neighborhood = -n_masked_points_per_neighborhood + n_total_points
        is_point_below_upper_limit = z < self.upper_limit if self.upper_limit else np.ones_like(z)
        is_point_above_lower_limit = z > self.lower_limit if self.lower_limit else np.ones_like(z)
        
        attribute2 = get_attributes_per_neighborhood(point_cloud, neighborhoods, [self.attribute_key])
        z2 = attribute2[:, 0, :]
        is_point_in_classes = np.isin(z2, self.attribute_values) if self.attribute_values else np.ones_like(z2)
        
        n_points_within_band = np.sum(is_point_below_upper_limit * is_point_above_lower_limit, axis=1)
        n_points_within_band_within_classes = np.sum(is_point_below_upper_limit * is_point_above_lower_limit * is_point_in_classes, axis=1)

        #convert to float64 to avoid errors when dividing by zero, etc
        n_points_within_band = n_points_within_band.astype(np.float64)
        n_points_within_band_within_classes = n_points_within_band_within_classes.astype(np.float64)
        n_points_per_neighborhood = n_points_per_neighborhood.astype(np.float64)

        band_to_neighborhood_ratio = np.divide(n_points_within_band, n_points_per_neighborhood, out = np.full_like(n_points_within_band, DIVIDE_BY_ZERO_VALUE), where = n_points_per_neighborhood!=0)
        band_within_classes_to_neighborhood_ratio = np.divide(n_points_within_band_within_classes, n_points_per_neighborhood, out = np.full_like(n_points_within_band_within_classes, DIVIDE_BY_ZERO_VALUE), where = n_points_per_neighborhood!=0)

        return _to_unmasked_array(n_points_within_band), _to_unmasked_array(n_points_within_band_within_classes), _to_unmasked_array(band_to_neighborhood_ratio),_to_unmasked_array(band_within_classes_to_neighborhood_ratio)

    def get_params(self):
        """
        Return a tuple of parameters involved in the current feature extractor object.

        Needed for provenance.
        """
        return (self.lower_limit, self.upper_limit, self.data_key, self.attribute_key, self.attribute_values)
