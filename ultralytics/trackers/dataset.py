from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import torchreid
from torchreid.data import ImageDataset


class CustomDataset(ImageDataset):
    dataset_dir = 'new_dataset'

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        train = ...
        query = ...
        gallery = ...

        super(CustomDataset, self).__init__(train, query, gallery, **kwargs)


torchreid.data.register_image_dataset('new_dataset', CustomDataset)

# use your own dataset only
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='new_dataset'
)
