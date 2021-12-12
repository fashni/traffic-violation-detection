import jetson.utils
from .stream import Camera

class JetsonCamera(Camera):
  capture_device = traitlets.Unicode(default_value="csi://0")
  flip = traitlets.Unicode(default_value="none")
  codec = traitlets.Unicode(default_value="h264")
  loop = traitlets.Integer(default_value=0)

  def __init__(self, *args, **kwargs):
    super(JetsonCamera, self).__init__(*args, **kwargs)
    arg = [
      f'--input-width={self.width}',
      f'--input-height={self.height}',
      f'--input-rate={self.fps}',
      f'--input-flip={self.flip}',
    ]
    if self.capture_device.startswith('rtp'):
      arg.append(f'--input-codec={self.codec}')
    if self.capture_device.startswith('file'):
      arg.append(f'--input-loop={self.loop}')
    self.cap = jetson.utils.videoSource(self.capture_device, argv=arg)

  def _read(self):
    return self.cap.Capture()
    # return jetson.utils.cudaToNumpy(img)[:, :, ::-1] #bgr for consistency
  
  def release(self):
    del self.cap
    self.cap = None
