import unittest
import deep3Dpredict as predict


class TestDeep3Dpredict(unittest.TestCase):

    def test_predict(self):
        pass

    def test_filter_saves(self):
        pass

    def test_findall(self):
        pass

    def test_tolist(self):
        pass

    def test_predict_parser(self):
        """
        -data DATA [DATA ...]
        -name NAME            name of the trained/saved CNN model (can be either a
        -output OUTPUT        output path for the predicted brain masks
        -cc CC                Filter connected components: removes all connected
        -format FORMAT        File saving format for predictions. Options are "h5",
        -prob PROB            save probability map as well
        -gridsize GRIDSIZE    size of CNN output grid (optimal: largest possible
        -data_clip_range DATA_CLIP_RANGE DATA_CLIP_RANGE
        -CNN_width_scale CNN_WIDTH_SCALE
        """
        argv = ['deepPredict.py', '-gridsize', str(16), '-data', '~/test.nii']
        parser = predict.predict_parser()
        args = parser.parse_args(argv[1:])
        self.assertEqual(args.data, ['~/test.nii'])
        self.assertEqual(args.name, 'OASIS_ISBR_LPBA40__trained_CNN.save')
        self.assertEqual(args.gridsize, 16)
