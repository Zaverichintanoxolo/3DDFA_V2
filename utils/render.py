# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('../')

import cv2
import numpy as np

from sim3dr.Sim3DR import RenderPipeline
from sim3dr.utils.functions import plot_image
from .tddfa_util import _to_ctype

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(ver_lst, tri, wfp):
    overlap = np.zeros((720, 1280, 3), dtype=np.uint8)
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        res = render_app(ver, tri, overlap)
        
    if wfp is not None:
        cv2.imwrite(wfp, res)
        
    return res
