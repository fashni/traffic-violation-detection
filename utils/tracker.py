import numpy as np


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
