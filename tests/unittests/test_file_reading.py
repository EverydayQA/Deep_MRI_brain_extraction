import unittest
from Deep_MRI_brain_extraction.utils import file_reading


class TestFileReading(unittest.TestCase):
    """
    Test file_reading.py
    """

    def test_get_path(self):
        path = file_reading.get_path('/tmp/abc/test.nii')
        self.assertEqual(path, '/tmp/abc')

    def test_get_filename(self):
        path = file_reading.get_filename('/tmp/abc/test.nii')
        self.assertEqual(path, 'test.nii')
        path = file_reading.get_filename('/tmp/abc/test.nii', remove_trailing_ftype=True)
        self.assertEqual(path, 'test')

    def test_mkdir(self):
        pass

    def test_load_nifti(self):
        pass

    def test_save_nifti(self):
        # (fname, data, affine=None, header=None)
        pass

    def test_load_h5(self):
        pass

    def test_save_h5(self):
        # (fname, data, compress=1, fast_compression=1):
        pass

    def test_save_text(self):
        pass

    def test_load_file(self):
        pass
