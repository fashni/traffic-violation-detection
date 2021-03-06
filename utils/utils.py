import glob
import re
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numba import njit

def bgr8_to_jpeg(value, quality=75):
  return bytes(cv2.imencode('.jpg', value)[1])

def rgb8_to_jpeg(value, quality=75):
  return bytes(cv2.imencode('.jpg', value[:, :, ::-1])[1])

def hex2bgr(hex):
  return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

def hex2rgb(hex):
  return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def imresize(img, width=None, height=None, interp=cv2.INTER_AREA):
  if img is None:
    return
  h, w = img.shape[:2]
  if width is None and height is None:
    return img
  if width is not None and height is not None:
    dim = (width, height)
  else:
    r = height/h if width is None else width/w
    dim = (int(r*w), int(r*h))
  return cv2.resize(img, dim, interpolation=interp)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
  """
  description: Plots one bounding box on image img,
  param: 
    x:      a box likes [x1,y1,x2,y2]
    img:    a opencv image object
    color:  color to draw rectangle, such as (0,255,0)
    label:  str
    line_thickness: int
  return:
    no return

  """
  tl = (
    line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
  )  # line/font thickness
  color = color or [random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
      img,
      label,
      (c1[0], c1[1] - 2),
      0,
      tl / 3,
      [225, 255, 255],
      thickness=tf,
      lineType=cv2.LINE_AA,
    )

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
  im_raw = im.copy()
  # Resize and pad image while meeting stride-multiple constraints
  h, w = shape = im.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
    new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better val mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scaleFill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
      im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  new_h, new_w = top+im.shape[0]+bottom, left+im.shape[1]+right
  new_pad_h, new_pad_w = (new_shape[0]-new_h)//2, (new_shape[1]-new_w)//2
  top, bottom = top+new_pad_h, bottom+new_pad_h
  left, right = left+new_pad_w, right+new_pad_w
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  im = im.astype(np.float32)
  # Normalize to [0,1]
  im /= 255.0
  # HWC to CHW format:
  im = np.transpose(im, [2, 0, 1])
  # CHW to NCHW format
  im = np.expand_dims(im, axis=0)
  # Convert the image to row-major order, also known as "C order":
  im = np.ascontiguousarray(im)
  return im, im_raw, h, w

@lru_cache(maxsize=None)
def get_line_pts(x, y, l, a, w, h):
  dx = 0.5*l*np.cos(np.deg2rad(a))
  dy = 0.5*l*np.sin(np.deg2rad(a))
  return (int((x-dx)*w), int((y-dy)*h)), (int((x+dx)*w), int((y+dy)*h))

@njit
def reduce_boxes(boxroi, roi_tl):
  roi_x, roi_y = roi_tl
  boxroi[:, ::2] -= roi_x
  boxroi[:, 1::2] -= roi_y

@njit
def expand_boxes(boxroi, roi_tl):
  roi_x, roi_y = roi_tl
  boxroi[:, ::2] += roi_x
  boxroi[:, 1::2] += roi_y

@njit
def is_intersect_line(line1, line2):
  P1, P2 = line1[:2], line1[2:]
  P3, P4 = line2[:2], line2[2:]
  A, B, C = P2-P1, P3-P4, P1-P3

  den = A[1]*B[0] - A[0]*B[1]
  num_a = B[1]*C[0] - B[0]*C[1]
  if den > 0:
      if num_a < 0 or num_a > den:
          return False
  elif num_a > 0 or num_a < den:
      return False

  num_b = A[0]*C[1] - A[1]*C[0]
  if den > 0:
      if num_b < 0 or num_b > den:
          return False
  elif num_b > 0 or num_b < den:
      return False

  return True

@njit
def get_line_segments(bbox):
  line1 = bbox.copy()
  line2 = bbox.copy()
  line3 = bbox.copy()
  line4 = bbox.copy()
  line1[2] = line1[0]
  line2[3] = line2[1]
  line3[1] = line3[3]
  line4[0] = line4[2]
  return np.vstack((line1, line2, line3, line4))

@njit
def is_crossing_stop_line(bbox_id, line_pos):
  nbox = bbox_id.shape[0]
  crossed = []
  for i in range(nbox):
    segments = get_line_segments(bbox_id[i, :4])
    cross = False
    for j in range(segments.shape[0]):
      cross |=  is_intersect_line(line_pos, segments[j])
    crossed.append(cross)

  return np.array(crossed)

def increment_path(path, exist_ok=False, sep='', mkdir=False):
  # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
  path = Path(path)
  if path.exists() and not exist_ok:
    path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 1  # increment number
    path = Path(f"{path}{sep}{n}{suffix}")  # increment path
  if mkdir:
    path.mkdir(parents=True, exist_ok=True)  # make directory
  return path
