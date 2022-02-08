from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import get_face_alignment_net, HighResolutionNet
from .hrnet6 import get_face_alignment_net6, HighResolutionNet6
__all__ = ['HighResolutionNet', 'get_face_alignment_net','get_face_alignment_net6','HighResolutionNet6']
