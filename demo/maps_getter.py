# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import numpy as np
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, _PanopticPrediction

from .predictor import VisualizationDemo


class MapsGetter(VisualizationDemo):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        A helper object to get raw map(s) (panoptic, instance, etc.) from the model.

        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        super(MapsGetter, self).__init__(cfg, instance_mode, parallel)

    def get_maps(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            maps (dict): a dictionary of maps.
        """
        maps = []
        predictions = self.predictor(image)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            self.pred = _PanopticPrediction(panoptic_seg, segments_info)
            sem_map, ins_map = self.panoptic_seg2sem_map(panoptic_seg), \
                               self.panoptic_seg2ins_map(panoptic_seg)
            maps = {"sem_map": sem_map, "ins_map": ins_map}
        else:
            if "sem_seg" in predictions:
                raise NotImplementedError()
            if "instances" in predictions:
                raise NotImplementedError()

        return maps

    def panoptic_seg2sem_map(self, panoptic_seg):
        n_classes = len(self.metadata.stuff_classes)
        class2color = {self.metadata.stuff_class[i]: i for i in range(n_classes)}
        canvas = np.zeros(list(panoptic_seg.shape))

        for mask, sinfo in self.pred.semantic_masks():
            color = class2color[self.metadata.stuff_class[sinfo["category_id"]]]
            canvas += mask.astype(np.int) * color

        return canvas

    def panoptic_seg2ins_map(self, panoptic_seg):
        color = 0
        canvas = np.zeros(list(panoptic_seg.shape))
        for mask, _ in self.pred.instance_masks():
            canvas += mask.astype(np.int) * color
            color += 1

        return canvas
