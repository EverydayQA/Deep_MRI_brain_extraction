
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
import os
from os import listdir as _listdir
from os.path import isfile as _isfile, join as _join


class DataMatch(object):
    """
    Filtering Data with 4 conditions startwith/endswith/contains/contains_not
    """

    def __init__(self, endswith=None, contains=None, startswith=None, contains_not=None):
        self.endswith = endswith
        self.contains = contains
        self.startswith = startswith
        self.contains_not = contains_not

    def list_directories(self, dir_paths):
        paths = []
        if not isinstance(dir_paths, list):
            if os.path.isdir(dir_paths):
                dir_paths = [dir_paths]
            else:
                raise Exception('dir_paths {} is not list'.format(dir_paths))
        for path in dir_paths:
            items = self.get_subdirs(path)
            paths.extend(items)
        # should they really be sorted?
        paths.sort()
        return paths

    def filter_list(self, string_list):
        items = []
        for item in string_list:
            if self.keep_str(item):
                items.append(item)
        return items

    def list_files(self, dir_paths):
        """
        Get files in each path in paths without walk into deeper, just os.listdir
        abspath()
        apply filter
        """
        files = []
        if not isinstance(dir_paths, list):
            if os.path.isdir(dir_paths):
                dir_paths = [dir_paths]
            else:
                raise Exception('dir_paths {} is not list'.format(dir_paths))
        for path in dir_paths:
            imgs = self.files_in_path(path)
            files.extend(imgs)
        files.sort()
        return files

    def get_subdirs(self, path):
        """
        Get files in the path, with conditions
        # e.g. '/home/nkrasows/phd/data/graham/Neurons/4dBinNeuronVolume/h5/'
        """
        if not os.path.isdir(path):
            return []
        paths = []
        path = os.path.abspath(path)
        for item in os.listdir(path):
            if not os.path.isdir(item):
                # skip file
                continue
            if self.keep_str(item):
                item = os.path.join(path, item)
                paths.append(item)
        return paths

    def files_in_path(self, path):
        """
        Get files in the path, with conditions
        # e.g. '/home/nkrasows/phd/data/graham/Neurons/4dBinNeuronVolume/h5/'
        """
        if not os.path.isdir(path):
            return []
        files = []
        path = os.path.abspath(path)
        for img in os.listdir(path):
            if os.path.isdir(img):
                # skip dir
                continue
            base = os.path.basename(img)
            if self.keep_str(base):
                files.append(img)
        return files

    def keep_str(self, base):
        """
        Assume basename match
        """
        if not self.keep_with_contains(base):
            return False
        if not self.keep_with_contains_not(base):
            return False
        if not self.keep_with_endswith(base):
            return False
        if not self.keep_with_startswith(base):
            return False
        return True

    def keep_with_startswith(self, img):
        """
        """
        if not self.startswith:
            return True
        if img.startswith(self.startswith):
            return True
        return False

    def keep_with_endswith(self, img):
        if not self.endswith:
            return True
        if img.endswith(self.endswith):
            return True
        return False

    def keep_with_contains(self, img):
        if not self.contains:
            return True
        if self.contains in img:
            return True
        return False

    def keep_with_contains_not(self, img):
        if not self.contains_not:
            return True
        if self.contains_not in img:
            return False
        return True


def list_files(dir_paths, endswith=None, contains=None, startswith=None, contains_not=None):
    """ endswith may be a sting like '.jpg' """
    files = []
    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]
    for path in dir_paths:  # '/home/nkrasows/phd/data/graham/Neurons/4dBinNeuronVolume/h5/',
        try:
            gg= [ (_join(path,f) if path!="." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]
            files += gg
        except:
            print("path",path,"invalid")
    files.sort()
    return files


def filter_list(string_list,endswith=None, contains=None, startswith=None, contains_not=None):
    return [ f for f in string_list if (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]


def list_directories(dir_paths, endswith=None, contains=None, startswith=None, contains_not=None):
    """ endswith may be a sting like '.jpg' """
    files=[]
    N_OK=0
    if type(dir_paths)!=type([]):
        dir_paths=[dir_paths]
    for path in dir_paths:
        try:
            gg= [ (_join(path,f) if path!="." else f) for f in _listdir(path) if _isfile(_join(path,f))==False and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]
            files+=gg
            N_OK+=1
        except:
            print("path <",path,"> is invalid")
    if N_OK==0:
        print('list_directories():: All paths were invalid!')
        raise ValueError()
    files.sort()
    return files




def LPBA40_data(location):
    all_ = list_directories(location)
    list_training_data_=[]
    list_training_labels_=[]
    for x in all_:
        candid = list_files(x,endswith=".nii.gz",startswith="S")
        dat = filter_list(candid, endswith="mri.nii.gz")
        lab = filter_list(candid, endswith="mask.nii.gz")
        assert len(dat)==1
        dat=dat[0]
        assert len(lab)==1
        lab=lab[0]
        list_training_data_.append(dat)
        list_training_labels_.append(lab)
    assert len(list_training_labels_) == 40
    assert len(list_training_data_) == len(list_training_labels_)
    return list_training_data_, list_training_labels_

class TraingDataLabels(object):
    """
    Centralize all data models
    """

    def oasis_kwargs(self):
        d = {}
        d['labels_location'] = None
        d['data_subdirs'] = ['disc1', 'disc2']
        d['endswith_labels'] = 'hardmask.nii.gz'
        d['endswith_data'] = 't88_gfc.hdr'
        d['processed_data_subdir'] = '/PROCESSED/MPRAGE/T88_111'
        return d

    def oasis_data(self):
        oasis = DataLabels(**self.oasis_kwargs())
        return oasis.list_training_labels, oasis.list_training_data

    def ibsr_kwargs(self):
        d = {}
        d['labels_location'] = None
        d['data_subdirs'] = None
        d['endswith_labels'] = 'mask.nii.gz'
        d['endswith_data'] = 'ana.nii.gz'
        d['processed_data_subdir'] = None
        return d

        def ibsr_data():
            ibsr = DataLabels(**self.ibsr.kwargs())
            return ibsr.list_training_labels, ibsr.list_training_data


def IBSR_data(location):

    all_ = list_directories(location)

    list_training_data_=[]
    list_training_labels_=[]
    for x in all_:
        candid = list_files(x,endswith=".nii.gz",startswith="IB")
        dat = filter_list(candid, endswith="ana.nii.gz")
        lab = filter_list(candid, endswith="mask.nii.gz")
        assert len(dat)==1
        dat=dat[0]
        assert len(lab)==1
        lab=lab[0]
        list_training_data_.append(dat)
        list_training_labels_.append(lab)
    assert len(list_training_labels_) == 18
    assert len(list_training_data_) == len(list_training_labels_)
    return list_training_data_, list_training_labels_



def ID_check(list_training_data_, list_training_labels_):
    IDs_data = ['.'.join(x.replace('\\','/').split('/')[-1].split('_')[:2]) for x in list_training_data_]
    IDs_labels = ['.'.join(x.replace('\\','/').split('/')[-1].split('_')[:2]) for x in list_training_labels_]
    for a,b in zip(IDs_data, IDs_labels):
        assert a==b, 'training data/labels are shuffled and do not match!: '+a+' <>  '+b

class DataLabels(object):
    """
    Flexible enough to define user's data location
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def labels_location(self):
        return self.kwargs.get('labels_location', None)

    @property
    def data_subdirs(self):
        """
        ['disc1', disc2']
        """
        return self.kwargs.get('data_subdirs', None)

    def dirs_training_data(self):
        dd = DataMatch()
        if not self.subdirs:
            return dd.list_directories(self.labels_location)
        dirs = []
        for subdir in self.data_subdirs:
            path = os.path.join(self.location_path, subdir)
            subdirs = dd.list_direcoties(path)
            dirs.extend(subdirs)
        return dirs

    @property
    def endswith_labels(self):
        """
        hardmask.nii.gz
        """
        return self.kwargs.get('endswith_label', None)

    def list_training_labels(self):
        dm = DataMatch(endswith=self.endswith_labels)
        return dm.list_files(self.labels_location)

    @property
    def endswith_data(self):
        """
        t88_gfc.hdr
        """
        return self.kwargs.get('endswith_data', None)

    @property
    def processed_data_subdir(self):
        """
        /PROCESSED/MPRAGE/T88_111
        """
        return self.kwargs.get('processed_data_subdir', None)

    def list_training_data(self):
        files = []
        for path in self.dirs_training_data():
            df = DataMatch(endswith=self.endswith_data)
            if self.processed_data_subdir:
                path = os.path.join(path, self.processed_data_subdir)
            items = df.list_files(path)
            files.extend(items)
        return files


def OASIS_data(location, labels_location = None):
    """
    labels_location: if None, then <location> will be used
    """
    if labels_location is None:
        labels_location = location

    list_training_labels_ = list_files(labels_location,endswith='hardmask.nii.gz')

    dirs = list_directories(location+"/disc1") + list_directories(location+"/disc2")
    list_training_data_=[]
    for d in dirs:
        dat = list_files(d+"/PROCESSED/MPRAGE/T88_111",endswith="t88_gfc.hdr")
        assert len(dat)==1
        dat=dat[0]
        list_training_data_.append(dat)
    assert len(list_training_data_)  ==77, len(list_training_data_)
    assert len(list_training_labels_)==77, len(list_training_labels_)
    assert len(list_training_data_) == len(list_training_labels_)
    ID_check(list_training_data_, list_training_labels_)
    return list_training_data_, list_training_labels_


def Tumor_data_JensCustomCreated():
    list_training_data_ = list_files("/home/share/brain_mask/tumor_data_h5", endswith='t1ce.nii.gz')
    list_training_labels_ = list_files("/home/share/brain_mask/tumor_data_h5", endswith='_human_mask.nii.gz')
    assert len(list_training_data_) == len(list_training_labels_)
    ID_check(list_training_data_, list_training_labels_)
    return list_training_data_, list_training_labels_


def get_CrossVal_part(list_training_data, list_training_labels, CV_index, CV_total_folds = 2):
    """ Splits data into training data/labels and test set.

        Inputs:
            CV_index: int from 0 to <CV_total_folds> - 1 (selects current split)

            CV_total_folds: Total number of CV folds, i.e. this must remain constant while CV_index changes from 0 up to <CV_total_folds> - 1

        returns:
            a dictionary with keys: 'list_test_data', 'list_training_data', 'list_training_labels'
            """
    N = len(list_training_data)
    cross_val_n_per_test= int(N*1./CV_total_folds)

    offset = CV_index*cross_val_n_per_test # must be calculated BEFORE the next if check !!!!!!!!!

    if CV_index == CV_total_folds - 1:
        cross_val_n_per_test = N - CV_index*cross_val_n_per_test # all remaining examples

    list_test_data = list_training_data[offset: offset + cross_val_n_per_test]
    list_training_data = list_training_data[:offset]  + list_training_data[offset + cross_val_n_per_test:]
    list_training_labels = list_training_labels[:offset]  + list_training_labels[(offset + cross_val_n_per_test):]
    assert len(list_training_data) == len(list_training_labels)
    return {'list_test_data': list_test_data, 'list_training_data': list_training_data, 'list_training_labels': list_training_labels}
