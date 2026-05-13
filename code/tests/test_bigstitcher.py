"""
Tests for the BigStitcher algorithm module
"""

import unittest

from aind_smartspim_stitch.algorithms.bigstitcher import (get_estimated_downsample, get_max_shifts,
                                                          get_stitching_dict)


class TestGetEstimatedDownsample(unittest.TestCase):
    """Tests for get_estimated_downsample"""

    def test_standard_2x_downsample(self):
        """Voxel resolution (1.8, 1.8, 2.0), target (7.2, 7.2, 8.0) → level 2"""
        level = get_estimated_downsample([1.8, 1.8, 2.0], phase_corr_res=(7.2, 7.2, 8.0))
        self.assertEqual(level, 2)

    def test_zero_downsample(self):
        """Equal resolutions → level 0"""
        level = get_estimated_downsample([8.0, 8.0, 8.0], phase_corr_res=(8.0, 8.0, 8.0))
        self.assertEqual(level, 0)

    def test_asymmetric_axes_takes_max(self):
        """Different ratios per axis → max level wins"""
        # X: 8/2=4 → log2=2; Y: 8/2=4 → log2=2; Z: 4/2=2 → log2=1 → max=2
        level = get_estimated_downsample([2.0, 2.0, 2.0], phase_corr_res=(8.0, 8.0, 4.0))
        self.assertEqual(level, 2)

    def test_raises_when_corr_res_smaller(self):
        """Raises ValueError if phase_corr_res < voxel_resolution in any axis"""
        with self.assertRaises(ValueError):
            get_estimated_downsample([8.0, 8.0, 8.0], phase_corr_res=(4.0, 8.0, 8.0))


class TestGetMaxShifts(unittest.TestCase):
    """Tests for get_max_shifts"""

    def test_basic_shifts(self):
        """Returns a tuple of three ints"""
        shifts = get_max_shifts(shape=(20, 1600, 2000), overlap=0.1, pyramid_level=1)
        self.assertIsInstance(shifts, tuple)
        self.assertEqual(len(shifts), 3)
        for s in shifts:
            self.assertIsInstance(s, int)

    def test_overlap_zero_uses_min_shift(self):
        """With overlap=0 every dim gets min_shift + room"""
        shifts = get_max_shifts(
            shape=(20, 1600, 2000), overlap=0.0, pyramid_level=1, min_shift=10, room=5
        )
        self.assertEqual(shifts, (15, 15, 15))

    def test_raises_invalid_overlap(self):
        """Raises ValueError for overlap outside [0, 1]"""
        with self.assertRaises(ValueError):
            get_max_shifts(shape=(20, 1600, 2000), overlap=1.5, pyramid_level=1)


class TestGetStitchingDict(unittest.TestCase):
    """Tests for get_stitching_dict"""

    def test_structure(self):
        """Dictionary has expected keys and values"""
        d = get_stitching_dict("test_specimen", "/fake/path.xml", downsample=3)
        self.assertEqual(d["session_id"], "test_specimen")
        self.assertEqual(d["dataset_xml"], "/fake/path.xml")
        self.assertTrue(d["do_phase_correlation"])
        self.assertFalse(d["do_detection"])
        params = d["phase_correlation_params"]
        self.assertEqual(params["downsample"], 3)
        self.assertEqual(params["max_shift_in_x"], 30)
        self.assertEqual(params["max_shift_in_y"], 30)
        self.assertEqual(params["max_shift_in_z"], 30)


if __name__ == "__main__":
    unittest.main()
