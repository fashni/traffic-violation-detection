import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from .utils import is_crossing_stop_line, increment_path

BASE_DIR = Path().resolve()
RUNS_DIR = BASE_DIR / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

class EuclideanDistTracker:
  def __init__(self, thres=25):
    # Distance threshold
    self.thres = thres
    # Store the center positions of the objects
    self.center_points = {}
    # Keep the count of the IDs
    # each time a new object id detected, the count will increase by one
    self.id_count = 0


  def update(self, objects_rect):
    # Objects boxes and ids
    objects_bbs_ids = []

    # Get center point of new object
    for rect in objects_rect:
      x1, y1, x2, y2 = rect
      cx = (x1 + x2)/2
      cy = (y1 + y2)/2

      # Find out if that object was detected already
      same_object_detected = False
      for id, pt in self.center_points.items():
        dist = np.hypot(cx - pt[0], cy - pt[1])

        if dist < self.thres:
          self.center_points[id] = (cx, cy)
          # print(self.center_points)
          objects_bbs_ids.append([x1, y1, x2, y2, id])
          same_object_detected = True
          break

      # New object is detected we assign the ID to that object
      if same_object_detected is False:
        self.center_points[self.id_count] = (cx, cy)
        objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
        self.id_count += 1

    # Clean the dictionary by center points to remove IDS not used anymore
    new_center_points = {}
    for obj_bb_id in objects_bbs_ids:
      _, _, _, _, object_id = obj_bb_id
      center = self.center_points[object_id]
      new_center_points[object_id] = center

    # Update dictionary with IDs not used removed
    self.center_points = new_center_points.copy()
    return np.array(objects_bbs_ids)

  def destroy(self):
    self.center_points = {}
    self.id_count = 0


class CarRecord:
  JPEG_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 100]

  def __init__(self, nosave=False):
    # store objects in dictionary
    self.records = {}
    self.nosave = nosave
    # self.create_run()

  def create_run(self):
    exec_time = datetime.now()
    RUN_DIR = RUNS_DIR / exec_time.isoformat()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR = RUN_DIR / 'frames'
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    CAR_DIR = RUN_DIR / 'cars'
    CAR_DIR.mkdir(parents=True, exist_ok=True)
    WS_DIR = RUN_DIR / 'windshields'
    WS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR = RUN_DIR / 'results.json'
    self.__dict__.update(locals())

  def update(self, frame_id, cboxes_id, ws_imgs, cw_res): # bbox car [x1, y1, x2, y2, id, stop_line, seat_belt_n]
    self.update_box(cboxes_id, frame_id)
    self.save_json()
    if not self.nosave:
      self.update_ws_imgs(ws_imgs)
      self.save_frame(cw_res)
      self.update_car_imgs(cboxes_id, cw_res.img)
    return self.records

  def update_box(self, cboxes_id, frame_id):
    for cbox in cboxes_id:
      object_id = int(cbox[4])
      box = cbox[:4]
      stop_line = not not cbox[5]
      n_passenger = int(cbox[6])
      n_seat_belt = int(cbox[7])

      # create first instance for newly encountered object
      if object_id not in self.records.keys():
        self.records[object_id] = {
          'frame_id': [frame_id],
          'positions': [box.tolist()],
          'stop_line': [stop_line],
          'n_passenger': [n_passenger],
          'n_seat_belt': [n_seat_belt]
        }
        continue

      # for new instance of saved object
      # check if object has each key, then append
      if 'frame_id' in self.records[object_id].keys():
        self.records[object_id]['frame_id'] += [frame_id]
      else: # else add new key
        self.records[object_id]['frame_id'] = [frame_id]
      if 'positions' in self.records[object_id].keys():
        self.records[object_id]['positions'] += [box.tolist()]
      else: # else add new key
        self.records[object_id]['positions'] = [box.tolist()]
      if 'stop_line' in self.records[object_id].keys():
        self.records[object_id]['stop_line'] += [stop_line]
      else:
        self.records[object_id]['stop_line'] = stop_line
      if 'n_passenger' in self.records[object_id].keys():
        self.records[object_id]['n_passenger'] += [n_passenger]
      else:
        self.records[object_id]['n_passenger'] = [n_passenger]
      if 'n_seat_belt' in self.records[object_id].keys():
        self.records[object_id]['n_seat_belt'] += [n_seat_belt]
      else:
        self.records[object_id]['n_seat_belt'] = [n_seat_belt]

  def update_car_imgs(self, cboxes_id, frame):
    for cbox in cboxes_id:
      idx = int(cbox[4])
      x1,y1,x2,y2 = cbox[:4].astype(int)
      c_img = frame[y1:y2, x1:x2, :]
      car_img_dir = self.CAR_DIR / str(idx)
      car_img_dir.mkdir(parents=True, exist_ok=True)
      file_path = increment_path(car_img_dir / f'{idx}_car.jpg')
      cv2.imwrite(str(file_path), c_img, self.JPEG_QUALITY)

  def update_ws_imgs(self, ws_imgs):
    for idx, ws_img in ws_imgs:
      ws_img_dir = self.WS_DIR / str(idx)
      ws_img_dir.mkdir(parents=True, exist_ok=True)
      file_path = increment_path(ws_img_dir / f'{idx}_windshield.jpg')
      cv2.imwrite(str(file_path), ws_img.img, self.JPEG_QUALITY)
      with file_path.with_suffix('.txt').open(mode='wt') as f:
        if ws_img.yolo.size > 0:
          boxes, cls_ids = ws_img.yolo[:, :4], ws_img.yolo[:, 5].astype(int)
          [f.write(f'{cls_id} ' + ' '.join(box) + '\n') for box, cls_id in zip(boxes.astype(str), cls_ids.astype(str))]

  def save_frame(self, res):
    file_path = increment_path(self.FRAME_DIR / f'frame.jpg')
    cv2.imwrite(str(file_path), res.img)
    with file_path.with_suffix('.txt').open(mode='wt') as f:
      if res.yolo.size > 0:
        boxes, cls_ids = res.yolo[:, :4], res.yolo[:, 5].astype(int)
        [f.write(f'{cls_id} ' + ' '.join(box) + '\n') for box, cls_id in zip(boxes.astype(str), cls_ids.astype(str))]

  def save_json(self):
    with open(self.JSON_DIR, 'w') as f:
      json.dump(self.records, f, indent=4)

  def destroy(self):
    del self.records
    self.records = {}
    self.create_run()
