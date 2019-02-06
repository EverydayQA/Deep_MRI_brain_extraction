import unittest
import mock
import os
from utils import file_reading


class TestFileReading(unittest.TestCase):
    """
    Test file_reading.py
    """

    def test_get_path(self):
        path = file_reading.get_path('/tmp/abc/test.nii')
        self.assertEqual(path, '/tmp/abc')

    def test_get_filename(self):
        fname = file_reading.get_filename('/tmp/abc/test.nii')
        self.assertEqual(fname, 'test.nii')

        fname = file_reading.get_filename('/tmp/abc/test.nii', remove_trailing_ftype=True)
        self.assertEqual(fname, 'test')
        fname = file_reading.get_filename('/tmp/abc/test.nii.gz', remove_trailing_ftype=True)
        self.assertEqual(fname, 'test.nii')

    def test_mkdir(self):
        with mock.patch('os.makedirs') as mock_makedirs:
            with mock.patch('os.path.exists') as mock_exists:
                mock_exists.return_value = False
                file_reading.mkdir('/tmp/predicts/abc.nii.gz')
                mock_makedirs.assert_called_once_with('/tmp/predicts')
                self.assertFalse(os.path.isdir('/tmp/predicts'))

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
