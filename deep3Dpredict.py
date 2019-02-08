"""
This software is an implementation of

Deep MRI brain extraction: A 3D convolutional neural network for skull stripping

You can download the paper at http://dx.doi.org/10.1016/j.neuroimage.2016.01.024

If you use this software for your projects please cite:

Kleesiek and Urban et al, Deep MRI brain extraction: A 3D convolutional neural network for skull stripping,
NeuroImage, Volume 129, April 2016, Pages 460-469.

The MIT License (MIT)

Copyright (c) 2016 Gregor Urban, Jens Kleesiek

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import sys
import argparse
import time
import numpy as np
import utils.data_and_CV_handler as data_and_CV_handler
from pprint import pprint
import utils.Segmentation_trainer as Segmentation_trainer
import utils.Segmentation_predictor as Segmentation_predictor


class Deep3dPredict(object):
    """
    Start with predict
    """

    def __init__(self, argv, *args_tuple, **kwargs):
        self.argv = argv
        self.args_tuple = args_tuple
        self.args, self.extra = self.predict_parser().parse_known_args(self.argv)
        self.kwargs = kwargs
        pprint(self.argv)
        pprint(self.args)
        pprint(self.extra)
        pprint(self.args_tuple)
        pprint(self.kwargs)

    @property
    def list_training_data(self):
        """
        lists of strings that specify the locations of training data files (each file must contain one 3D or 4D-volume of type float32; the 4th dimension containins the channels)
        """
        return []

    @property
    def list_training_labels(self):
        """
        lists of strings that specify the locations of training labels files (each file must contain one 3D or 4D-volume of type int; the 4th dimension containins the channels)
        """
        return []

    @property
    def list_test_data(self):
        """
        lists of strings that specify the locations of test data files (each file must contain one 3D or 4D-volume of type float32; the 4th dimension containins the channels)
        """
        return self.data

    @property
    def data(self):
        data = self.findall(self.tolist(self.args.data))
        pprint(data)
        assert len(data) > 0, 'Could not find the data. Please either pass all paths to the individual files or place them in a single folder and pass the path to this folder as "-i" argument'
        assert self.args.format in ['nifti', 'h5', 'numpy'], 'Argument "format" must be nifti, h5, or numpy'
        return data

    @property
    def save_name(self):
        """
        str, name of the folder (and files) for saving/loading the trained network parameters
        """
        save_name = self.tolist(self.args.name)
        save_name = self.filter_saves(save_name)
        print('using model-parameters: {}'.format(save_name))
        return save_name

    def filter_saves(self, path_or_file):
        candidates = self.findall(path_or_file)
        matches = []
        for c in candidates:
            if '.save' in c:
                matches.append(c)
                if 'end_' in c:
                    return c
        if len(matches) == 0:
            raise ValueError('The provided save file/directory does not contain any saved model (file ending in .save)')
        return matches[-1]

    @property
    def n_labels_pred_per_dim(self):
        """
        16 or 32
        # this number should be as large as possible to increase the speed-efficiency when making predictions
        # the only limit is the RAM of the GPU which will manifest as memory allocation errors

        # 32
        # n_labels_pred_per_dim = n_labels_pred_per_dim

        """
        return self.args.gridsize

    @property
    def output_path(self):
        output_path = str(self.args.output)
        if len(output_path) and output_path.replace('\\', '/')[-1] != '/':
            output_path += '/'
        return output_path

    @property
    def apply_cc_filtering(self):
        return bool(self.args.cc)

    @property
    def output_filetype(self):
        if self.args.format:
            return self.args.format
        return 'h5'

    @property
    def save_prob_map(self):
        """
        False by default
        """
        return self.args.prob

    def predict(self):
        """
        This is the runner for the brain mask prediction project.
        It will either train the CNN or load the trained network and predict the test set.
        """
        assert len(self.list_training_data) == len(self.list_training_labels)

        # CNN specification:
        cnn, patchCreator = Segmentation_trainer.Build3D(self.nnet_args_build3d, **self.kwargs_build3d)
        cnn.LoadParameters(self.save_name)

        t0 = time.clock()
        Segmentation_predictor.predict_all(cnn, patchCreator, apply_cc_filtering=self.apply_cc_filtering,
                                           save_as=self.output_path, output_filetype=self.output_filetype,
                                           save_prob_map=self.save_prob_map)
        t1 = time.clock()
        print("Predicted all in {} seconds".format(t1 - t0))

    @property
    def nnet_args_build3d(self):

        # number of classes in the data set -  2 means binary classification.
        n_classes = 2

        nnet_args = {}
        nnet_args["filter_sizes"] = [4, 5, 5, 5, 5, 5, 5, 1]

        # this indicates where max-pooling is used ( a value of 1 means no pooling)
        nnet_args["pooling_factors"] = [2, 1, 1,  1, 1, 1, 1, 1]

        # nnet_args["nof_filters"]     = [1, 1, 1, 1, 1, 1, 1,   n_classes] # number of different filters in each layer:
        nnet_args["nof_filters"] = [16, 24, 28, 34, 42, 50, 50, n_classes]  # number of different filters in each layer:

        nnet_args["nof_filters"] = [int(np.ceil(self.network_size_factor * x)) for x in nnet_args["nof_filters"][:-1]] + [nnet_args["nof_filters"][-1]]
        return nnet_args

    @property
    def kwargs_build3d(self):
        d3d = {}
        n_labels_per_batch = self.n_labels_pred_per_dim**(3)
        d3d['n_labels_per_batch'] = n_labels_per_batch
        d3d['notrain'] = True
        bDropoutEnabled = 0
        d3d['bDropoutEnabled'] = bDropoutEnabled
        patch_depth = 1

        d3d['patch_depth'] = patch_depth
        input_to_cnn_depth = patch_depth
        d3d['input_to_cnn_depth'] = input_to_cnn_depth

        override_data_set_filenames = {"train_data": self.list_training_data,
                                       "test_data": self.list_test_data,
                                       "train_labels": self.list_training_labels}

        d3d['override_data_set_filenames'] = override_data_set_filenames
        num_patches_per_batch = 1

        d3d['num_patches_per_batch'] = num_patches_per_batch
        d3d['actfunc'] = "relu"
        d3d['data_init_preserve_channel_scaling'] = 0
        d3d['data_clip_range'] = self.args.data_clip_range
        use_fragment_pooling = 0
        d3d['use_fragment_pooling'] = use_fragment_pooling
        d3d['auto_threshold_labels'] = self.auto_threshold_labels
        return d3d

    @property
    def network_size_factor(self):
        """
        1
        """
        return float(self.args.CNN_width_scale)

    @property
    def auto_threshold_labels(self):
        """
        False
        """
        return False

    def findall(self, paths):
        """
        locate and return all files in the paths (list of directory/file names)
        """
        rlist = []
        for x in paths:
            rlist += data_and_CV_handler.list_files(x) if data_and_CV_handler.os.path.isdir(x) else [x]
        return rlist

    def tolist(self, x):
        return x if isinstance(x, list) else [x]

    def predict_parser(self):
        parser = argparse.ArgumentParser(description='Main module to apply an already trained 3D-CNN to segment data')

        parser.add_argument('-data', type=str, nargs='+', required=True, help='Any number and combination of paths to files or folders that will be used as input-data for training the CNN')
        parser.add_argument('-name', default='OASIS_ISBR_LPBA40__trained_CNN.save', type=str,  help='name of the trained/saved CNN model (can be either a folder or .save file)')
        parser.add_argument('-output', default='predictions/', type=str, help='output path for the predicted brain masks')

        parser.add_argument('-cc', default=True, type=bool,  help='Filter connected components: removes all connected components but the largest two (i.e. background and brain) [default=True]')
        parser.add_argument('-format', default='nifti', type=str,  help='File saving format for predictions. Options are "h5", "nifti", "numpy" [default=nifti]')
        parser.add_argument('-prob', default=1, type=bool,  help='save probability map as well')
        parser.add_argument('-gridsize', default=32, type=int,  help='size of CNN output grid (optimal: largest possible divisor of the data-volume axes that still fits into GPU memory). This setting heavily affects prediction times: larger values are better. Values that are too large will cause a failure due to too little GPU-memory.')

        parser.add_argument('-data_clip_range', default=None, type=float, nargs=2, help='[Mostly for single-channel data] Clip all pixel-values outside of the given range (important if values of volumes have very different ranges!) -- Must be identical to the setting used during training!')
        parser.add_argument('-CNN_width_scale', default=1, type=float, help='Scale factor for the layer widths of the CNN; values larger than 1 will increase the total network size beyond the default size, but be careful to not exceed your GPU memory. -- Must be identical to the setting used during training!')
        return parser


def main():
    deep = Deep3dPredict(sys.argv[1:])
    deep.predict()


if __name__ == '__main__':
    main()
