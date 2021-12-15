import copy
import random
import threading
import time

import cv2
import numpy as np
import onnxruntime
import pandas as pd
from numba import njit, jit
# import pycuda.autoinit
# import pycuda.driver as cuda
# import tensorrt as trt
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from .utils import plot_one_box


@njit
def bbox_iou(box1, box2, x1y1x2y2=True):
  if not x1y1x2y2:
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
  else:
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

  # Get the coordinates of the intersection rectangle
  inter_rect_x1 = np.maximum(b1_x1, b2_x1)
  inter_rect_y1 = np.maximum(b1_y1, b2_y1)
  inter_rect_x2 = np.minimum(b1_x2, b2_x2)
  inter_rect_y2 = np.minimum(b1_y2, b2_y2)
  # Intersection area
  inter_rect_x = inter_rect_x2 - inter_rect_x1 + 1
  inter_rect_y = inter_rect_y2 - inter_rect_y1 + 1
  inter_area = np.minimum(inter_rect_x.max(), np.maximum(inter_rect_x, 0)) * \
               np.minimum(inter_rect_y.max(), np.maximum(inter_rect_y, 0))
  # inter_area = np.clip(inter_rect_x, 0, None) * \
  #              np.clip(inter_rect_y, 0, None)

  # Union Area
  b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

  iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

  return iou


@njit
def xywh2xyxy(x, origin_h, origin_w, input_h, input_w):
  """
  description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  param:
    x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    origin_h:   height of original image
    origin_w:   width of original image
    input_h:    height of processed image
    input_w:    width of processed image
  return:
    y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
  """
  y = np.zeros_like(x)
  r_w = input_w / origin_w
  r_h = input_h / origin_h
  if r_h > r_w:
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
    y /= r_w
  else:
    y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    y /= r_h

  return y


class Yolo(object):
  """
  description: Base class for Yolo inference, including preprocessing and postprocessing ops.
  """

  def __init__(self, classes=None, conf=0.5, iou=0.4):
    self.conf = conf
    self.iou = iou
    self.load_class_names(classes)

  def setup_model(self, *args, **kwargs):
    raise NotImplementedError

  def infer(self, *args, **kwargs):
    raise NotImplementedError

  def destroy(self, *args, **kwargs):
    raise NotImplementedError

  def load_class_names(self, classes):
    self.classes = classes or [f'class{i}' for i in range(1000)]
    self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

  def preprocess_image(self, raw_bgr_image):
    """
    description: Convert BGR image to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
      raw_bgr_image: numpy array, OpenCV image
      imgsz: list of two integers [h, w], target size
    return:
      image:  the processed image
      image_raw: the original image
      h: original height
      w: original width
    """
    # return letterbox(raw_bgr_image, (self.input_h, self.input_w), scaleup=False)
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = self.input_w / w
    r_h = self.input_h / h
    if r_h > r_w:
      tw = self.input_w
      th = int(r_w * h)
      tx1 = tx2 = 0
      ty1 = int((self.input_h - th) / 2)
      ty2 = self.input_h - th - ty1
    else:
      tw = int(r_h * w)
      th = self.input_h
      tx1 = int((self.input_w - tw) / 2)
      tx2 = self.input_w - tw - tx1
      ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
      image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, image_raw, h, w

  def post_process(self, output, origin_h, origin_w):
    """
    description: postprocess the prediction
    param:
      output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
      origin_h:   height of original image
      origin_w:   width of original image
    return:
      result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
      result_scores: finally scores, a numpy, each element is the score correspoing to box
      result_classid: finally classid, a numpy, each element is the classid correspoing to box
    """
    # Do nms
    boxes = self.non_max_suppression(output, origin_h, origin_w, self.input_h, self.input_w, conf_thres=self.conf, nms_thres=self.iou)

    # check and filter box with w=1 and h=1 (line)
    boxes = self.check_boxes(boxes)

    # return boxes
    results = boxes if len(boxes) else np.array([])
    return results
  
  @staticmethod
  def get_detection_matrix(prediction, classes=None, max_boxes=6000):
    """
    description: Convert raw network output (batch, detect, 5 + n_cls) to nx6 matrix (xywh, conf, cls).
    param:
      prediction: numpy array, network output
      max_boxes: integer, maximum box sorted by conf
    return:
      output:  numpy array, nx6 matrix
    """
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
      if not x.shape[0]:
        continue
      # Compute conf
      x[:, 5:] *= x[:, 4:5]  # conf = cls_conf * obj_conf

      # Detections matrix nx6 (xywh, conf, cls)
      conf = x[:, 5:].max(1, keepdims=True)
      j =  np.expand_dims(x[:, 5:].argmax(1), axis=1)
      x = np.concatenate((x[:, :4], conf, j.astype(conf.dtype)), 1)#[conf.flatten() > 0.5]

      # filter by class
      if classes is not None:
        x = x[(x[:, 5:6] == np.array(classes)).any(1)]

      # check
      n = x.shape[0]  # number of boxes
      if not n:  # no boxes
        continue
      elif n > max_boxes:  # excess boxes
        x = x[x[:, 4].argsort()[::-1][:max_boxes]]  # sort by confidence

      output[xi] = x

    return np.array(output) if len(output) > 1 else output[0]

  @staticmethod
  def check_boxes(boxes):
    if len(boxes)==0:
      return boxes
    boxes[:, 2:4] -= boxes[:, :2] - 1
    boxes = boxes[(boxes[:, 2:4].astype(int) > 1).all(1)]
    boxes[:, 2:4] += boxes[:, :2] - 1
    return boxes
  
  @staticmethod
  @njit
  def non_max_suppression(prediction, origin_h, origin_w, input_h, input_w, conf_thres=0.5, nms_thres=0.4):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
      prediction: detections, (x1, y1, x2, y2, conf, cls_id)
      origin_h: original image height
      origin_w: original image width
      conf_thres: a confidence threshold to filter detections
      nms_thres: a iou threshold to filter detections
    return:
      boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH
    boxes = prediction[prediction[:, 4] >= conf_thres]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes[:, :4] = xywh2xyxy(boxes[:, :4], origin_h, origin_w, input_h, input_w)
    # clip the coordinates
    boxes[:, 0] = np.minimum(origin_w -1, np.maximum(boxes[:, 0], 0)) #np.clip(boxes[:, 0], 0, origin_w -1)
    boxes[:, 2] = np.minimum(origin_w -1, np.maximum(boxes[:, 2], 0)) #np.clip(boxes[:, 2], 0, origin_w -1)
    boxes[:, 1] = np.minimum(origin_h -1, np.maximum(boxes[:, 1], 0)) #np.clip(boxes[:, 1], 0, origin_h -1)
    boxes[:, 3] = np.minimum(origin_h -1, np.maximum(boxes[:, 3], 0)) #np.clip(boxes[:, 3], 0, origin_h -1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = np.zeros_like(boxes)
    idx = 0
    while boxes.shape[0]:
      large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
      label_match = boxes[0, -1] == boxes[:, -1]
      # Indices of boxes with lower confidence scores, large IOUs and matching labels
      invalid = large_overlap & label_match
      keep_boxes[idx] = boxes[0]
      boxes = boxes[~invalid]
      idx = idx+1

    return keep_boxes


class YoloInfer(Yolo):
  """
  description: Yolo inference using ONNX and Polygraphy (TensorRT Backend).
  """

  def __init__(self, model_path=None, classes=None, conf=0.5, iou=0.4):
    super(YoloInfer, self).__init__(classes, conf, iou)
    if model_path is None:
      return
    self.setup_model(model_path, classes)

  def setup_model(self, model_path, classes):
    self.load_class_names(classes)
    if model_path.endswith('.onnx'):
      self.backend = 'onnx'
      self.session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
    elif model_path.endswith('.engine') or model_path.endswith('.trt'):
      self.backend = 'trt'
      self.engine = EngineFromBytes(BytesFromPath(model_path))
      self.session = TrtRunner(self.engine)
      self.session.activate()
      self.input_h, self.input_w = self.session.get_input_metadata()['images'].shape[2:]
      self.input_dtype = self.session.get_input_metadata()['images'].dtype

  def infer(self, img, imgsz=None, classes=None, render=False, proc_time=False):
    threading.Thread.__init__(self)
    if self.backend=='onnx':
      if imgsz is None:
        self.input_h, self.input_w = img.shape[:2]
      if isinstance(imgsz, int):
        self.input_h = self.input_w = imgsz
      else:
        self.input_h, self.input_w = imgsz

    # preprocessing
    pre_start = time.perf_counter()
    input_image, image_raw, origin_h, origin_w = self.preprocess_image(img)
    pre_end = time.perf_counter()
    
    # inference
    infer_start = time.perf_counter()
    if self.backend=='onnx':
      y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: input_image})[0]
    else:
      out = self.session.infer(feed_dict={"images": input_image.astype(self.input_dtype)})
      y = copy.deepcopy(out['output'])
    infer_end = time.perf_counter()
    output = self.get_detection_matrix(y, classes=classes)

    # postprocessing
    post_start = time.perf_counter()
    results = self.post_process(output, origin_h, origin_w)
    post_end = time.perf_counter()
    # return results

    infer_result = InferResult(image_raw, results, self.classes, self.colors)
    if render:
      infer_result.render()
    if proc_time:
      return infer_result, (pre_end-pre_start, infer_end-infer_start, post_end-post_start)

    return infer_result

  def destroy(self):
    try:
      self.session.deactivate()
    except:
      del self.session


class YoloTRT(Yolo):
  """
  description: Yolo inference using Polygraphy (TensorRT Backend).
  """

  def __init__(self, model_path=None, classes=None, conf=0.5, iou=0.4):
    super(YoloTRT, self).__init__(classes, conf, iou)
    if model_path is None:
      return
    self.setup_model(model_path, classes)

  def setup_model(self, model_path, classes):
    self.load_class_names(classes)
    self.engine = EngineFromBytes(BytesFromPath(model_path))
    self.runner = TrtRunner(self.engine)
    self.runner.activate()
    self.input_h, self.input_w = self.runner.get_input_metadata()['images'].shape[2:]
    self.input_dtype = self.runner.get_input_metadata()['images'].dtype
    # self.input_h, self.input_w = input_size[2:]

  def infer(self, img, imgsz=None, classes=None, render=False, proc_time=False):
    threading.Thread.__init__(self)
    # preprocessing
    pre_start = time.perf_counter()
    input_image, image_raw, origin_h, origin_w = self.preprocess_image(img)
    pre_end = time.perf_counter()

    # inference
    infer_start = time.perf_counter()
    y = self.runner.infer(feed_dict={"images": input_image.astype(self.input_dtype)})
    infer_end = time.perf_counter()
    output = self.get_detection_matrix(copy.deepcopy(y['output']), classes=classes)

    # postprocessing
    post_start = time.perf_counter()
    results = self.post_process(output, origin_h, origin_w)
    post_end = time.perf_counter()
    # return results

    infer_result = InferResult(image_raw, results, self.classes, self.colors)
    if render:
      infer_result.render()
    if proc_time:
      return infer_result, (pre_end-pre_start, infer_end-infer_start, post_end-post_start)

    return infer_result

  def destroy(self):
    self.runner.deactivate()


class YoloONNX(Yolo):
  """
  description: Yolo inference using ONNX.
  """

  def __init__(self, model_path=None, classes=None, conf=0.5, iou=0.4):
    super(YoloONNX, self).__init__(classes, conf, iou)
    if model_path is None:
      return
    self.setup_model(model_path, classes)

  def setup_model(self, model_path, classes):
    self.load_class_names(classes)
    self.session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())

  def infer(self, img, imgsz=None, classes=None, render=False, proc_time=False):
    threading.Thread.__init__(self)
    if imgsz is None:
      self.input_h, self.input_w = img.shape[:2]
    if isinstance(imgsz, int):
      self.input_h = self.input_w = imgsz
    else:
      self.input_h, self.input_w = imgsz

    # preprocessing
    pre_start = time.perf_counter()
    input_image, image_raw, origin_h, origin_w = self.preprocess_image(img)
    pre_end = time.perf_counter()

    # inference
    infer_start = time.perf_counter()
    y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: input_image})[0]
    # return y
    infer_end = time.perf_counter()
    output = self.get_detection_matrix(y, classes=classes)

    # postprocessing
    post_start = time.perf_counter()
    results = self.post_process(output, origin_h, origin_w)
    post_end = time.perf_counter()

    infer_result = InferResult(image_raw, results, self.classes, self.colors)
    if render:
      infer_result.render()
    if proc_time:
      return infer_result, (pre_end-pre_start, infer_end-infer_start, post_end-post_start)

    return infer_result

  def destroy(self):
    del self.session


class YoloTRT2(YoloInfer):
  """
  description: Yolo inference using TensorRT.
  """

  def __init__(self, engine_file_path=None, classes=None, conf=0.5, iou=0.4):
    super(YoloTRT, self).__init__(classes, conf, iou)
    if engine_file_path is None:
      return
    self.setup_model(engine_file_path, classes)

  def setup_model(self, engine_file_path, classes):
    # Create a Context on this device,
    self.ctx = cuda.Device(0).make_context()
    self.load_class_names(classes)
    stream = cuda.Stream()
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)

    # Deserialize the engine from file
    with open(engine_file_path, "rb") as f:
      engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    outputs_shape = []

    for binding in engine:
      binding_shape = engine.get_binding_shape(binding)
      print('binding:', binding, binding_shape)
      size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
      dtype = trt.nptype(engine.get_binding_dtype(binding))
      # Allocate host and device buffers
      host_mem = cuda.pagelocked_empty(size, dtype)
      cuda_mem = cuda.mem_alloc(host_mem.nbytes)
      # Append the device buffer to device bindings.
      bindings.append(int(cuda_mem))
      # Append to the appropriate list.
      if engine.binding_is_input(binding):
        self.input_w = engine.get_binding_shape(binding)[-1]
        self.input_h = engine.get_binding_shape(binding)[-2]
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
      else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
        outputs_shape.append(binding_shape)

    # Store
    self.stream = stream
    self.context = context
    self.engine = engine
    self.host_inputs = host_inputs
    self.cuda_inputs = cuda_inputs
    self.host_outputs = host_outputs
    self.cuda_outputs = cuda_outputs
    self.bindings = bindings
    self.batch_size = engine.max_batch_size
    self.outputs_shape = outputs_shape

  def infer_img(self, img):
    res = self.infer(img, render=True)
    return res.img

  def infer(self, raw_image_generator, imgsz=None, classes=None, render=False, proc_time=False):
    threading.Thread.__init__(self)
    # Make self the active context, pushing it on top of the context stack.
    self.ctx.push()
    # Restore
    stream = self.stream
    context = self.context
    engine = self.engine
    host_inputs = self.host_inputs
    cuda_inputs = self.cuda_inputs
    host_outputs = self.host_outputs
    cuda_outputs = self.cuda_outputs
    bindings = self.bindings
    # Do image preprocess
    pre_start = time.perf_counter()
    input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image_generator)
    pre_end = time.perf_counter()

    # Copy input image to host buffer
    np.copyto(host_inputs[0], input_image.ravel())
    infer_start = time.perf_counter()
    # Transfer input data  to the GPU.
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    # Run inference.
    context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    # Synchronize the stream
    stream.synchronize()
    infer_end = time.perf_counter()
    # Remove any context from the top of the context stack, deactivating it.
    self.ctx.pop()
    # Here we use the first row of output in that batch_size = 1
    output = self.get_detection_matrix(host_outputs[0][0:6001], classes=classes)

    # Do postprocess
    post_start = time.perf_counter()
    results = self.post_process(output, origin_h, origin_w)
    post_end = time.perf_counter()

    infer_result = InferResult(image_raw, results, self.classes, self.colors)
    if render:
      infer_result.render()
    if proc_time:
      return infer_result, (pre_end-pre_start, infer_end-infer_start, post_end-post_start)

    return infer_result

  def destroy(self):
    # Remove any context from the top of the context stack, deactivating it.
    self.ctx.pop()

  def get_detection_matrix(self, output, classes=None):
    # Get the num of boxes detected
    num = int(output[0])
    # Reshape to a two dimentional ndarray
    x = np.reshape(output[1:], (-1, 6))[:num, :]
    # filter by class
    if classes is not None:
      x = x[(x[:, 5:6] == np.array(classes)).any(1)]
    return x


class InferResult(object):
  valid_forms = ['xyxy', 'xywh', 'ltwh', 'yolo']
  results = boxes = confs = classids = np.array([])
  _xyxy = _xywh = _ltwh = _yolo = np.array([])

  def __init__(self, raw_img, results, class_lbl=None, colors=None, ti=None):
    self.raw_img = raw_img
    self.xyxy = results
    if len(results):
      self.boxes, self.confs, self.classids = results[:, :4], results[:, 4], results[:, 5]
      self.ltwh = self.xyxy
      self.xywh = self.ltwh
      self.yolo = self.xywh
    self.classes = class_lbl or [f'class{i}' for i in range(1000)]
    self.colors = colors or [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

  @property
  def raw_img(self):
    return self._raw_img

  @raw_img.setter
  def raw_img(self, value):
    self._raw_img = value
    self.img = value.copy()

  @property
  def xyxy(self): # top-left, right-bottom
    return self._xyxy

  @xyxy.setter
  def xyxy(self, value):
    self._xyxy = value
    self.ltwh = value
    self.xywh = self.ltwh
    self.yolo = self.xywh

  @property
  def ltwh(self):  # x, y --> top-left
    self.ltwh = self.xyxy
    return self._ltwh

  @ltwh.setter
  def ltwh(self, value):
    self._ltwh = value.copy()
    if value.size > 0:
      self._ltwh[:, 2:4] -= self._ltwh[:, :2] - 1

  @property
  def xywh(self): # x, y --> center
    self.xywh = self.ltwh
    return self._xywh

  @xywh.setter
  def xywh(self, value):
    self._xywh = value.copy()
    if value.size > 0:
      self._xywh[:, :2] += self._xywh[:, 2:4]/2

  @property
  def yolo(self):
    self.yolo = self.xywh
    return self._yolo

  @yolo.setter
  def yolo(self, value):
    self._yolo = value.copy()
    if value.size > 0:
      h, w = self.raw_img.shape[:2]
      box = self._yolo[:, :4]
      box[:, ::2] /= w
      box[:, 1::2] /= h

  def get_class_name(self, cls_id):
    return self.classes[cls_id]

  def get_columns_name(self, form='xyxy'):
    conf_clsid = ['conf','class_id']
    if form not in self.valid_forms:
      return
    if form=='xyxy':
      return ['x1', 'y1', 'x2', 'y2'] + conf_clsid
    elif form=='xywh' or form=='yolo':
      return ['cx', 'cy', 'w', 'h']  + conf_clsid
    elif form=='ltwh':
      return ['x', 'y', 'w', 'h']  + conf_clsid
    else:
      return ['x', 'y', 'a', 'h']  + conf_clsid

  def pandas(self, form='xyxy'):
    if form not in self.valid_forms:
      print('unknown format, using xyxy')
      form = 'xyxy'
    boxes = vars(self)['_'+form]
    cols = self.get_columns_name(form)
    if boxes.size == 0:
      return pd.DataFrame(columns=cols)
    df = pd.DataFrame(boxes, columns=cols)
    df['class_id'] = df['class_id'].astype(np.int8)
    df['class_label'] = np.array(self.classes)[self.classids.astype(np.int8)]
    return df

  def render(self, classes=None):
    self.img = self.raw_img.copy()
    boxes = self.get_class_boxes(classes, form='xyxy')
    if boxes.size > 0:
      bb, confs, classids = boxes[:, :4], boxes[:, 4], boxes[:, 5]
      for box, conf, clsid in zip(bb, confs, classids):
        plot_one_box(box, self.img, color=self.colors[int(clsid)], label=f"{self.classes[int(clsid)]}:{conf:#.2f}")
    return self

  def crop(self, save=False):
    if hasattr(self, 'crops'):
      return self.crops
    self.crops = []
    for box, conf, cid in zip(self.boxes, self.confs, self.classids):
      x1, y1, x2, y2 = box.astype(np.int16)
      crop = {}
      crop['box'] = box
      crop['conf'] = conf
      crop['class_id'] = cid
      crop['class_label'] = self.classes[int(cid)]
      crop['img'] = self.raw_img[y1:y2, x1:x2]
      self.crops.append(crop)
    return self.crops

  def get_class_boxes(self, classes=None, form='xyxy'):
    if form not in self.valid_forms:
      print('invalid format, using xyxy.')
      form = 'xyxy'
    boxes = vars(self)['_'+form].copy()
    if classes is None:
      return boxes
    boxes = boxes[(boxes[:, 5:6] == np.array(classes)).any(1)]
    return boxes
