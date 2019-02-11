from nilearn import datasets
# import datasets module and use `fetch_atlas_basc_multiscale_2015` function
# import plotting module and use `plot_roi` function, since the maps are in 3D
from nilearn import plotting

"""
Visualizing multiscale functional brain parcellations
=====================================================

This example shows how to download and fetch brain parcellations of
multiple networks using :func:`nilearn.datasets.fetch_atlas_basc_multiscale_2015`
and visualize them using plotting function :func:`nilearn.plotting.plot_roi`.

We show here only three different networks of 'symmetric' version. For more
details about different versions and different networks, please refer to its
documentation.
"""
# Let us use a Nifti file that is shipped with nilearn


def plot_nii(nii_file):
    plotting.plot_img(nii_file)
    plotting.show()


def plot_atlas():

    ###############################################################################
    # Retrieving multiscale group brain parcellations
    # -----------------------------------------------
    from nilearn.datasets import MNI152_FILE_PATH
    print('Path to MNI152 template: %r' % MNI152_FILE_PATH)

    parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')

    # We show here networks of 64, 197, 444
    networks_64 = parcellations['scale064']
    networks_197 = parcellations['scale197']
    networks_444 = parcellations['scale444']

    ###############################################################################
    # Visualizing brain parcellations
    # -------------------------------

    # The coordinates of all plots are selected automatically by itself
    # We manually change the colormap of our choice
    plotting.plot_roi(networks_64, cmap=plotting.cm.bwr,
                      title='64 regions of brain clusters')

    plotting.plot_roi(networks_197, cmap=plotting.cm.bwr,
                      title='197 regions of brain clusters')

    plotting.plot_roi(networks_444, cmap=plotting.cm.bwr_r,
                      title='444 regions of brain clusters')

    plotting.show()


def main():
    from utils.data_and_CV_handler import DataMatch
    dm = DataMatch(endswith='.nii.gz')
    niis = dm.find_nii('./')
    print(niis)
    for nii in niis:
        plot_nii(nii)
    if not niis:
        print('\n*** cannot find any nii files to plot')


if __name__ == '__main__':
    main()
