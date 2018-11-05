import os
import time
import numpy as np
import imgaug

from config import Config
import utils
import model as modellib

import pathlib as pl

from partnet.config import Config as PNConfig
import partnet.dataset as pd
import partnet.util as pu

class ChairConfig(Config):

    NAME = "chairs"

    IMAGES_PER_GPU = 2

    # Number of part classes
    NUM_CLASS = 1 + 13

    # Use RGB-D data
    USE_DEPTH = True
    USE_NORMAS = True


class ChairDataset(utils.Dataset):
    def __init__(self, cfg):
        self.cfg_ = cfg        
        # load the chair labels
        self.id_to_label_ = {}
        self.label_to_id_ = {}
        with open(str(self.cfg_.label_path),"r") as f:
            for line in f.readlines():
                label, i = line.strip().split(" ")
                self.id_to_label_[i] = label
                self.label_to_id_[label] = i
                self.class_info.append({"source": "chairs",
                                        "id": i,
                                        "name": label})
        super(ChairDataset,self).__init__()
    
    def load_chairs(self, dataset):
        self.dataset_ = dataset
        samples = self.dataset_.samples_list(["chair"])        
        i = 0        
        for mdl,frm in samples:
            info = {"id": i,
                    "source": "chairs",
                    "path": frm[0],
                    "frame_id": pu.frame_num(frm[0]),
                    "depth": frm[1],
                    "instance": frm[2],
                    "masks": frm[4]}
            self.image_info.append(info)
        
        
    def load_mask(self, image_id):
        pass
