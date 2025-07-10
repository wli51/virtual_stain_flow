from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes

class PatchCollection:
    """
    Helper class for maintaining an ordered collection square patches on a 2D plane,
        plus optional object coordianates. 
    Each patch is as a 1D array: [x_min, y_min, x_max, y_max].
    The collection supports: 
        1. adding new patches
        2. checking for patch vs all existing patches overlap
        3. checking if a single coordiante is contained within a specific patch
        4. maintaining a boolean mask indicating if each of the object coordinates
            is contained within any patches across patch adding process.
    """
    def __init__(
            self,
            obj_coordinates: Optional[pd.DataFrame] = None,
            plane_x_max: Optional[float] = None,
            plane_y_max: Optional[float] = None,
        ):
        """
        Initializes an empty collection of bounding boxes.

        :param obj_coordinates: Optional pandas DataFrame with object coordinates.
            Each row should contain at least two columns for x and y coordinates.
        :param plane_x_max: Optional float, maximum x-coordinate of the plane.
        :param plane_y_max: Optional float, maximum y-coordinate of the plane.
        """
        # Initialize an empty collection of bounding boxes
        # that is a numpy stack of 4-dimensional arrays.
        self._boxes = np.empty((0, 4), dtype=float)

        if not isinstance(obj_coordinates, (pd.DataFrame, type(None))):
            raise TypeError("obj_coordinates must be a pandas DataFrame or None")
        if obj_coordinates is not None and obj_coordinates.shape[1] < 2:
            raise ValueError("obj_coordinates must have at least 2 columns")
        self._obj_coordinates = obj_coordinates
        if obj_coordinates is not None:
            self._obj_captured_flag = np.zeros(len(self._obj_coordinates), dtype=bool)

        if not isinstance(plane_x_max, (float, type(None))):
            raise TypeError("plane_x_max must be a float or None")
        self._plane_x_max = plane_x_max
        if not isinstance(plane_y_max, (float, type(None))):
            raise TypeError("plane_y_max must be a float or None")
        self._plane_y_max = plane_y_max

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if not (0 <= idx < len(self._boxes)):
            raise IndexError("Bounding box index out of range")
        return self._boxes[idx]

    def add(self, box):
        """
        Adds a new box to the end of the collection.

        :param box: list or array-like of shape (4,), [x_min, y_min, x_max, y_max]
        """
        box = np.asarray(box, dtype=float).reshape(1, 4)
        if box.shape != (1, 4):
            raise ValueError("Box must be a 1D array of shape (4,)")
        self._boxes = np.vstack([self._boxes, box])

    def check_overlap(self, box):
        """
        Checks whether the input box overlaps with any box in the collection.

        :param box: list or array-like of shape (4,)
        :return: bool, True if overlap with any box
        """
        if self._boxes.shape[0] == 0:
            return False

        box = np.asarray(box, dtype=float)
        x_min, y_min, x_max, y_max = box

        overlaps = (
            (self._boxes[:, 0] < x_max) &
            (self._boxes[:, 2] > x_min) &
            (self._boxes[:, 1] < y_max) &
            (self._boxes[:, 3] > y_min)
        )

        return np.any(overlaps)
    
    def check_overlap_soft(self, box, threshold=0.1):
        """
        Checks whether the input box overlaps with any existing box using IoU threshold.
        This is a softer criterion allowing for partial overlaps.

        :param box: array-like of shape (4,), [x_min, y_min, x_max, y_max]
        :param threshold: float, minimum IoU to count as overlapping
        :return: bool, True if IoU with any existing box >= threshold
        """
        if self._boxes.shape[0] == 0:
            return False

        box = np.asarray(box, dtype=float).reshape(1, 4)
        x_min, y_min, x_max, y_max = box[0]

        # Sanity check: box must be valid
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Invalid box coordinates: {box[0]}")

        # Extract existing boxes
        boxes = self._boxes
        boxes_x_min = boxes[:, 0]
        boxes_y_min = boxes[:, 1]
        boxes_x_max = boxes[:, 2]
        boxes_y_max = boxes[:, 3]

        # Calculate intersection
        inter_xmin = np.maximum(boxes_x_min, x_min)
        inter_ymin = np.maximum(boxes_y_min, y_min)
        inter_xmax = np.minimum(boxes_x_max, x_max)
        inter_ymax = np.minimum(boxes_y_max, y_max)

        inter_w = np.clip(inter_xmax - inter_xmin, a_min=0, a_max=None)
        inter_h = np.clip(inter_ymax - inter_ymin, a_min=0, a_max=None)
        inter_area = inter_w * inter_h

        # Area of union = area1 + area2 - intersection
        area1 = (x_max - x_min) * (y_max - y_min)
        area2 = (boxes_x_max - boxes_x_min) * (boxes_y_max - boxes_y_min)
        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-8)  # add epsilon for stability

        return np.any(iou >= threshold)

    def contains_point(self, index, x, y):
        """
        Checks whether a point (x, y) is within the bounding box at the given index.

        :param index: int
        :param x: float
        :param y: float
        :return: bool
        """
        if not (0 <= index < self._boxes.shape[0]):
            raise IndexError("Bounding box index out of range")

        x_min, y_min, x_max, y_max = self._boxes[index]
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
    
    @property
    def obj_captured_flag(self):
        """
        Returns a boolean array indicating which objects are captured by any patch.

        :return: numpy array of bools
        """
        if self._obj_coordinates is None:
            return np.array([], dtype=bool)
        return self._obj_captured_flag
    
    def update_captured_objects(self):
        """
        Update which objects in self._obj_coordinates are captured by any patch.

        :return: Updated captured_flags
        """
        if self._obj_coordinates is None or len(self._boxes) == 0:
            return

        obj_coords = self._obj_coordinates.iloc[:, :2].to_numpy()
        x = obj_coords[:, 0]
        y = obj_coords[:, 1]

        for box in self._boxes[len(self) - 1:]:  # only check the latest added box
            x_min, y_min, x_max, y_max = box
            inside = (
                (x >= x_min) & (x <= x_max) &
                (y >= y_min) & (y <= y_max)
            )
            self._obj_captured_flag = self._obj_captured_flag | inside  # vectorized OR
    
    def plot(self, 
             ax: Optional[Axes]=None, 
             show_indices: bool=False, 
             **kwargs
        ) -> Axes:
        """
        Helper plot functions for visualizing the PatchCollection.
        Plots all bounding boxes using matplotlib plus object coordinates if exists.
        Optionally labels each box and object with its index.
        The user may pass in a matplotlib axis to plot on, or a new one will be created.

        :param ax: optional matplotlib axis. If None, creates one.
        :param show_indices: bool, whether to label boxes with their index
        :param kwargs: additional keyword args passed to Rectangle (e.g., edgecolor, linewidth)
        :return: matplotlib axis with plotted boxes
        """
        if ax is None:
            fig, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError("ax must be a matplotlib Axes instance or None")

        for i, (x_min, y_min, x_max, y_max) in enumerate(self._boxes):
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                fill=False,
                **kwargs
            )
            ax.add_patch(rect)
            if show_indices:
                ax.text(x_min, y_min, str(i), fontsize=8, color='blue')
        
        # Plot all object coordinates
        if self._obj_coordinates is None:
            pass
        else:
            for i, row in self._obj_coordinates.iterrows():
                x, y = row[0], row[1]
                ax.plot(x, y, 'r*', markersize=6)
                if show_indices:
                    ax.text(x, y, 
                            str(i), 
                            fontsize=8, 
                            color='red', 
                            ha='left', 
                            va='bottom'
                            )

        ax.set_aspect('equal')
        if self._plane_x_max is not None:
            ax.set_xlim(*(0, self._plane_x_max))
        if self._plane_y_max is not None:
            ax.set_ylim(*(0, self._plane_y_max))
        else:
            ax.autoscale_view()
        return ax
