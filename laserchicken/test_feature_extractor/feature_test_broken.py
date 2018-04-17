"""Broken feature extractor."""
from laserchicken.feature_extractor.abc import AbstractFeatureExtractor


class TestBrokenFeatureExtractor(AbstractFeatureExtractor):
    """Feature extractor that fails to add the feature in promises to provide to target."""

    @classmethod
    def requires(cls):
        return []

    @classmethod
    def provides(cls):
        return ['test_broken']

    def extract(self, sourcepc, neighborhood, targetpc, targetindex, volume):
        pass