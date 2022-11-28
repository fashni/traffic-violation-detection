import atexit
import threading
import time

import cv2
import numpy as np
import traitlets


class Camera(traitlets.HasTraits):
  value = traitlets.Any()
  width = traitlets.Integer(default_value=224)
  height = traitlets.Integer(default_value=224)
  format = traitlets.Unicode(default_value='bgr8')
  running = traitlets.Bool(default_value=False)
  fps = traitlets.Float(default_value=30)

  def __init__(self, *args, **kwargs):
    super(Camera, self).__init__(*args, **kwargs)
    if self.format == 'bgr8':
      self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
    self._running = False

  def _read(self):
    """Blocking call to read frame from camera"""
    raise NotImplementedError

  def read(self):
    if self._running:
      raise RuntimeError('Cannot read directly while camera is running')
    self.value = self._read()
    return self.value

  def _capture_frames(self):
    while True:
      if not self._running:
        break
      self.value = self._read()
      time.sleep(1 / self.fps)

  @traitlets.observe('running')
  def _on_running(self, change):
    if change['new'] and not change['old']:
      # transition from not running -> running
      self._running = True
      self.thread = threading.Thread(target=self._capture_frames)
      self.thread.start()
    elif change['old'] and not change['new']:
      # transition from running -> not running
      self._running = False
      self.thread.join()


class URLStream(Camera):
  url = traitlets.Unicode(default_value='https://youtu.be/MNn9qKG2UFI') #https://youtu.be/eCQoTgxCCSg

  def __init__(self, *args, **kwargs):
    super(URLStream, self).__init__(*args, **kwargs)
    if 'youtube.com/' in self.url or 'youtu.be/' in self.url:  # if source is YouTube video
      import pafy
      self.url = pafy.new(self.url).getbestvideo(preftype="mp4").url
    # self.cap = cv2.VideoCapture(self._gst_str, cv2.CAP_GSTREAMER)
    self.cap = cv2.VideoCapture(self.url)
    assert self.cap.isOpened(), f'Failed to open {self.url}'
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
    re, self.value = self.cap.read()
    atexit.register(self.cap.release)

  def _read(self):
    self.cap.grab()
    re, image = self.cap.retrieve()
    if re:
      return image
    else:
      self.cap.open(self.url)
      return self.value * 0

  def release(self):
    if self._running:
      self._running = False
    self.unobserve_all()
    self.cap.release()


class CameraStream(Camera):
  capture_device = traitlets.Integer(default_value=0)
  capture_fps = traitlets.Integer(default_value=30)
  capture_width = traitlets.Integer(default_value=640)
  capture_height = traitlets.Integer(default_value=480)
  wbmode = traitlets.Integer(default_value=1)
  tnr_mode = traitlets.Integer(default_value=1)
  tnr_strength = traitlets.Integer(default_value=-1)
  ee_mode = traitlets.Integer(default_value=1)
  ee_strength = traitlets.Integer(default_value=-1)
  contrast = traitlets.Float(default_value=1.0)
  brightness = traitlets.Float(default_value=0.0)
  saturation = traitlets.Float(default_value=1.0)

  def __init__(self, *args, **kwargs):
    super(CameraStream, self).__init__(*args, **kwargs)
    self.cap = cv2.VideoCapture(self._gst_str, cv2.CAP_GSTREAMER)
    assert self.cap.isOpened(), 'Failed to open camera'
    self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or float(self.capture_fps)
    re, self.value = self.cap.read()
    atexit.register(self.cap.release)

  @property
  def _gst_str(self):
    r = self.width/self.height
    hh = self.capture_height
    ww = int(r*hh)

    top = 0
    bottom = hh - 1
    left = self.capture_width//2 - ww//2
    right = left + ww - 1
    # top = self.capture_height//2 - self.height//2
    # bottom = top + self.height-1
    # left = self.capture_width//2 - self.width//2
    # right = left + self.width-1
    # gainrange="16 16" ispdigitalgainrange="1 1"
    return 'nvarguscamerasrc sensor-id=%d wbmode=%d tnr-mode=%d tnr-strength=%d ee-mode=%d ee-strength=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv top=%d bottom=%d left=%d right=%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! videobalance contrast=%f brightness=%f saturation=%f ! appsink max-buffers=1 drop=true' % (
            self.capture_device, self.wbmode, self.tnr_mode, self.tnr_strength, self.ee_mode, self.ee_strength,
            self.capture_width, self.capture_height, self.capture_fps, top, bottom, left, right, self.width, self.height,
            self.contrast, self.brightness, self.saturation)

  def _read(self):
    self.cap.grab()
    re, image = self.cap.retrieve()
    if re:
      return image
    else:
      self.cap.open(self._gst_str, cv2.CAP_GSTREAMER)
      return self.value * 0

  def release(self):
    if self._running:
      self._running = False
    self.unobserve_all()
    self.cap.release()

  def change_properties(self, **kwargs):
    if self._running:
      raise RuntimeError('Cannot change properties while camera is running')
    self.release()
    self.__init__(**kwargs)


class FileStream(Camera):
  filepath = traitlets.Unicode(default_value='')
  def __init__(self, *args, **kwargs):
    super(FileStream, self).__init__(*args, **kwargs)
    if self.filepath:
      self.cap = cv2.VideoCapture(self.filepath)
      # self.cap = cv2.VideoCapture(self._gst_str, cv2.CAP_GSTREAMER)
      assert self.cap.isOpened(), f'Failed to open {self.filepath}'
      self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
      self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
      re, self.value = self.cap.read()
      # print(self.frames, self.cap.get(cv2.CAP_PROP_POS_FRAMES))
      atexit.register(self.cap.release)
    else:
      raise RuntimeError('No video specified')

  @property
  def _gst_str(self):
    return f'filesrc location={self.filepath} ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx !  appsink'

  def release(self):
    if self._running:
      self._running = False
    self.unobserve_all()
    self.cap.release()

  def rewind(self):
    if self._running:
      self._running = False
      # raise RuntimeError('Cannot rewind the video while it is playing')
    self.cap.open(self.filepath)

  def _read(self):
    self.cap.grab()
    re, image = self.cap.retrieve()
    if re:
      return image

    nframe = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    if nframe >= self.frames: # last frame
      self._running = False
      return self.value * 0

    self.cap.open(self.filepath)
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, nframe)
    return self.value
      
    # raise RuntimeError('Cannot read frame')
    # self.unobserve(self._trait_notifiers['value']['change'][0], names='value')
    # # self.running = False


class StreamHandler:
  handlers = [CameraStream, FileStream, URLStream]
  def __init__(self, stream_handler=None, **kwargs):
    self.set_handler(stream_handler, **kwargs)

  def set_handler(self, stream_handler=None, **kwargs):
    if stream_handler is None:
      self.stream = None
      return
    if stream_handler >= len(self.handlers):
      raise RuntimeError('Unknown stream handler')
    self.stream = self.handlers[stream_handler](**kwargs)

  @property
  def is_camera(self):
    return isinstance(self.stream, CameraStream)

  def release(self):
    if self.stream is None:
      return
    self.stream.release()
    self.stream = None

