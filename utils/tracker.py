import numpy as np
from .utils import is_crossing_stop_line


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


class CarRecord:
  def __init__(self):
    # store objects in dictionary
    self.records = {}

  def update(self, cboxes_id, c_imgs, ws_imgs): # bbox car [x1, y1, x2, y2, id, stop_line, seat_belt_n]
    self.update_box(cboxes_id)
    self.update_car_imgs(c_imgs)
    self.update_ws_imgs(c_imgs)
    return self.records

  def update_box(self, cboxes_id):
    for cbox in cboxes_id:
      object_id = int(cbox[4])
      box = cbox[:4]
      stop_line = not not cbox[5]
      n_seat_belt = int(cbox[6])

      # create first instance for newly encountered object
      if object_id not in self.records.keys():
        self.records[object_id] = {
          'positions': [box],
          'stop_line': stop_line,
          'n_seat_belt': n_seat_belt
        }
        continue

      # for new instance of saved object
      # check if object has each key, then append
      if 'positions' in self.records[object_id].keys():
        self.records[object_id]['positions'] += [box]
      else: # else add new key
        self.records[object_id]['positions'] = [box]
      if 'stop_line' in self.records[object_id].keys():
        self.records[object_id]['stop_line'] |= stop_line
      else:
        self.records[object_id]['stop_line'] = stop_line
      if 'n_seat_belt' in self.records[object_id].keys():
        self.records[object_id]['n_seat_belt'] = max(self.records[object_id]['n_seat_belt'], n_seat_belt)
      else:
        self.records[object_id]['n_seat_belt'] = n_seat_belt

  def update_car_imgs(self, c_imgs):
    for idx, c_img in c_imgs:
      if idx not in self.records.keys():
        self.records[idx] = {'car_imgs': [c_img]}
        continue

      if 'car_imgs' not in self.records[idx].keys():
        self.records[idx]['car_imgs'] = [c_img]
      else:
        self.records[idx]['car_imgs'] += [c_img]
      
  def update_ws_imgs(self, ws_imgs):
    for idx, ws_img in ws_imgs:
      if idx not in self.records.keys():
        self.records[idx] = {'windshield_imgs': [ws_img]}
        continue

      if 'windshield_imgs' not in self.records[idx].keys():
        self.records[idx]['windshield_imgs'] = [ws_img]
      else:
        self.records[idx]['windshield_imgs'] += [ws_img]

  def save(self):
    pass

  def destroy(self):
    del self.records
    self.records = {}
