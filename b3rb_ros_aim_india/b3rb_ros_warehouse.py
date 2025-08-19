# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from pyzbar import pyzbar

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()

		self.state = "WAITING_FOR_ROBOT"
		self.confirmed_shelves = []
		self.mission_queue = []
		self.mission_timer = self.create_timer(1.0, self.mission_control_loop)
		self.current_task = None
		self.next_heuristic_angle = math.radians(self.initial_angle)
		self.has_initial_pose = False
		self.search_step_distance = 1.5
		self.search_step_taken = 0
		self.max_search_steps = 5
		

	def mission_control_loop(self):
		"""The main brain of the robot. Manages the state machine."""

		# We only make decisions if the robot has no active goal.
		if not self.goal_completed:
			return

		self.get_logger().info(f"--- Brain Tick --- State: {self.state}")

		if self.state == "WAITING_FOR_ROBOT":
			if self.armed and self.has_initial_pose:
				self.get_logger().info("ROBOT IS READY. Starting Directed Search.")
				self.state = "SEARCHING"
				# Immediately call the loop again to process the new SEARCHING state
				self.mission_control_loop()
			else:
				self.get_logger().info(f"Waiting for robot... Armed={self.armed}, Localized={self.has_initial_pose}")

		elif self.state == "SEARCHING":
			if len(self.confirmed_shelves) > 0:
				self.get_logger().info(f"!!! SUCCESS! Found a shelf. Mission accomplished for now. !!!")
				self.state = "IDLE"
				return

			if self.search_step_taken >= self.max_search_steps:
				self.get_logger().error("Search failed: Reached max steps without finding a shelf.")
				self.state = "IDLE"
				return

			# --- Calculate and send the NEXT incremental goal ---
			angle_rad = self.next_heuristic_angle
			
			goal_x = self.buggy_pose_x + self.search_step_distance * math.cos(angle_rad)
			goal_y = self.buggy_pose_y + self.search_step_distance * math.sin(angle_rad)
			
			self.get_logger().info(f"Search step {self.search_step_taken + 1}/{self.max_search_steps}: "
								f"Moving towards {self.next_heuristic_angle:.1f} deg.")
			
			goal_pose = self.create_goal_from_world_coord(goal_x, goal_y)
			success = self.send_goal_from_world_pose(goal_pose)
			
			if success:
				self.search_step_taken += 1
			else:
				self.get_logger().warn("Failed to send incremental search goal. Will retry on next tick.")

		elif self.state == "IDLE":
			# The robot is idle, do nothing.
			pass

	def calculate_shelf_poses(self, shelf_info):
		"""
		Calculates all 4 navigation poses for a shelf (front/back for objects, left/right for QR).
		"""
		center_x, center_y = shelf_info['center_world']
		orientation = shelf_info['orientation_rad']
		perp_orientation = orientation + math.pi / 2
		
		offset_dist = 1.0 # How far away from the shelf to stop

		# --- Poses for Object Viewing (Front and Back) ---
		# Pose 1: Front
		obj1_x = center_x + offset_dist * math.cos(perp_orientation)
		obj1_y = center_y + offset_dist * math.sin(perp_orientation)
		obj1_yaw = perp_orientation + math.pi # Look at shelf
		object_pose1 = self.create_goal_from_world_coord(obj1_x, obj1_y, obj1_yaw)
		
		# Pose 2: Back (offset in the opposite direction)
		obj2_x = center_x - offset_dist * math.cos(perp_orientation)
		obj2_y = center_y - offset_dist * math.sin(perp_orientation)
		obj2_yaw = perp_orientation # Look at shelf
		object_pose2 = self.create_goal_from_world_coord(obj2_x, obj2_y, obj2_yaw)

		# --- Poses for QR Code Viewing (Left and Right Sides) ---
		# Pose 1: Right Side
		qr1_x = center_x + offset_dist * math.cos(orientation)
		qr1_y = center_y + offset_dist * math.sin(orientation)
		qr1_yaw = orientation + math.pi # Look at shelf side
		qr_pose1 = self.create_goal_from_world_coord(qr1_x, qr1_y, qr1_yaw)

		# Pose 2: Left Side
		qr2_x = center_x - offset_dist * math.cos(orientation)
		qr2_y = center_y - offset_dist * math.sin(orientation)
		qr2_yaw = orientation # Look at shelf side
		qr_pose2 = self.create_goal_from_world_coord(qr2_x, qr2_y, qr2_yaw)
		
		return {
			'object_poses': [object_pose1, object_pose2],
			'qr_poses': [qr_pose1, qr_pose2]
		}

	def execute_next_task(self):
		"""Pops the next task from the queue and sends the goal."""
		if not self.mission_queue:
			self.get_logger().info("--- MISSION COMPLETE ---")
			self.state = "IDLE"
			return
			
		if not self.goal_completed:
			self.get_logger().warn("Waiting for previous goal to complete.")
			return

		self.current_task = self.mission_queue.pop(0)
		self.get_logger().info(f"Executing task: {self.current_task['task_type']} for Shelf {self.current_task['id']}")
		self.send_goal_from_world_pose(self.current_task['goal_pose'])


	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

		if not self.has_initial_pose:
			self.get_logger().info("<<<<< INITIAL POSE RECEIVED from SLAM. Robot is localized. >>>>>")
			self.has_initial_pose = True

	def simple_map_callback(self, message):
		"""
		Callback to process the raw map, find shelf-like objects,
		and merge them to get a stable list of confirmed shelves.
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		
		# Early exit if map is not ready
		if map_info.width < 10 or map_info.height < 10:
			return

		# --- Step 1: Convert map data to a binary image ---
		map_data = np.array(message.data).reshape((map_info.height, map_info.width))
		binary_image = np.zeros_like(map_data, dtype=np.uint8)
		binary_image[map_data == 100] = 255  # Obstacles are white for OpenCV

		# --- Step 2: Find all distinct shapes (contours) ---
		contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# --- Step 3: Analyze each contour to see if it's a potential shelf ---
		shelf_dimensions_m = (1.35, 0.5)  # (length, width) in meters
		resolution = map_info.resolution
		
		potential_shelves = []
		for contour in contours:
			# --- A) Filter by Area ---
			area_pixels = cv2.contourArea(contour)
			if area_pixels < 20:  # Filter out small noise
				continue
			
			area_m2 = area_pixels * (resolution**2)
			shelf_area_m2 = shelf_dimensions_m[0] * shelf_dimensions_m[1]
			
			if not np.isclose(area_m2, shelf_area_m2, atol=0.3):  # Looser tolerance for area
				continue

			# --- B) Filter by Shape using minAreaRect ---
			rect = cv2.minAreaRect(contour)
			(w_pixels, h_pixels) = rect[1]
			w_m, h_m = w_pixels * resolution, h_pixels * resolution
			
			dim1, dim2 = sorted((w_m, h_m))
			shelf_dim1, shelf_dim2 = sorted(shelf_dimensions_m)
			
			if not (np.isclose(dim1, shelf_dim1, atol=0.3) and np.isclose(dim2, shelf_dim2, atol=0.3)):
				continue

			# --- C) If it passes, it's a potential shelf. Calculate its properties. ---
			center_pixels = rect[0]
			world_center = self.get_world_coord_from_map_coord(center_pixels[0], center_pixels[1], map_info)
			
			# --- D) Use PCA for robust orientation (as hinted) ---
			points = contour.reshape(-1, 2).astype(np.float32)
			pca = PCA(n_components=2)
			pca.fit(points)
			angle_rad = np.arctan2(pca.components_[0][1], pca.components_[0][0])
			
			# Add this candidate to our list for this callback run
			potential_shelves.append({
				'center_world': world_center,
				'orientation_rad': angle_rad
			})

		# --- Step 4: Merge potential shelves with our master list of confirmed shelves ---
		merge_distance_threshold = 0.75  # If centers are within 0.75m, they are the same shelf

		for potential_shelf in potential_shelves:
			is_new_shelf = True
			for i, confirmed_shelf in enumerate(self.confirmed_shelves):
				distance = euclidean(potential_shelf['center_world'], confirmed_shelf['center_world'])
				
				if distance < merge_distance_threshold:
					# This is an update to an existing shelf, not a new one.
					is_new_shelf = False
					
					# Update the existing entry with the new, more current data.
					# This helps refine the position as the map improves.
					self.confirmed_shelves[i]['center_world'] = potential_shelf['center_world']
					self.confirmed_shelves[i]['orientation_rad'] = potential_shelf['orientation_rad']
					break # Move to the next potential shelf

			if is_new_shelf:
				# This potential shelf is far from all our confirmed shelves. It's a new discovery!
				self.get_logger().info(f"!!! NEW SHELF DISCOVERED at {potential_shelf['center_world']} !!!")
				new_shelf_data = {
					'id': len(self.confirmed_shelves) + 1,
					'center_world': potential_shelf['center_world'],
					'orientation_rad': potential_shelf['orientation_rad']
				}
				self.confirmed_shelves.append(new_shelf_data)

		# --- Final Logging ---
		# Periodically log the state of our confirmed shelves list
		if len(self.confirmed_shelves) > 0:
			shelf_positions = [f"ID {s['id']}: ({s['center_world'][0]:.2f}, {s['center_world'][1]:.2f})" for s in self.confirmed_shelves]
			self.get_logger().info(f"Confirmed Shelves ({len(self.confirmed_shelves)}): {shelf_positions}")

	def global_map_callback(self, message):
		"""Callback function to handle global map updates.

		Args:
			message: ROS2 message containing the global map data.

		Returns:
			None
		"""
		return

		if self.state != "EXPLORING":
			return


		self.global_map_curr = message

		if not self.goal_completed:
			return

		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontiers = self.get_frontiers_for_space_exploration(map_array)

		map_info = self.global_map_curr.info
		if frontiers:
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
											 map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)
				if (distance < min_distance_curr and
				    distance <= self.max_step_dist_world_meters and
				    distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				print("Sending goal for space exploration.")
				return
			else:
				self.max_step_dist_world_meters += 2.0
				new_min_step_dist = self.min_step_dist_world_meters - 1.0
				self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

			self.full_map_explored_count = 0
		else:
			self.full_map_explored_count += 1
			print(f"Nothing found in frontiers; count = {self.full_map_explored_count}")

	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""
		Callback function to handle incoming camera images.
		This function now also decodes QR codes using pyzbar.
		"""
		# --- Step 1: Decode the compressed ROS message into an OpenCV image ---
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		# --- Step 2: Decode QR code using pyzbar ---
		# pyzbar is more efficient and accurate with grayscale images.
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		qrcodes = pyzbar.decode(gray_image)

		# --- Step 3: Process the results ---
		if qrcodes:
			# Loop through all detected QR codes (though usually there is only one).
			for qrcode in qrcodes:
				# Decode the QR data from bytes into a human-readable string.
				qr_data_string = qrcode.data.decode('utf-8')
				
				# Check if this is a NEW QR code we haven't seen before.
				# This prevents spamming the log with the same message every frame.
				if qr_data_string != self.qr_code_str:
					self.qr_code_str = qr_data_string
					self.get_logger().info(f"!!! NEW QR CODE DETECTED: {self.qr_code_str} !!!")

				# --- (Optional) Draw a box around the QR code for debugging ---
				(x, y, w, h) = qrcode.rect
				# Draw a green rectangle around the QR code.
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
				# Add a label above the box.
				cv2.putText(image, "QR_DETECTED", (x, y - 10), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# --- Step 4: (Optional but recommended) Publish the debug image ---
		# This sends the image (with the green box) to Foxglove for visualization.
		self.publish_debug_image(self.publisher_qr_decode, image)

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates."""
		
		# Check if the robot is armed and ready
		if message.mode == 3 and message.arming == 2:
			self.armed = True
			# If the robot is armed AND we have NOT created our trigger timer yet...
			# if self.trigger_timer is None:
			# 	self.get_logger().info("Robot is ARMED. Starting autonomous sequence in 5 seconds...")
				
			# 	# Create the one-shot timer and STORE it
			# 	self.trigger_timer = self.create_timer(5.0, self.execute_hardcoded_goal_sequence)
		else:
			# If the robot is NOT armed, command it to arm.
			# self.get_logger().info("Robot is not armed. Requesting arming...") # This can be spammy, maybe comment out
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

			# If the timer exists, cancel it because we've been disarmed.
			# if self.trigger_timer is not None:
			# 	self.get_logger().warn("Robot disarmed, cancelling sequence trigger.")
			# 	self.trigger_timer.cancel()
			# 	self.trigger_timer = None

	def trigger_hardcoded_goal(self):
		"""This function is called by the timer to safely start the sequence."""
		self.execute_hardcoded_goal_sequence()

	def execute_hardcoded_goal_sequence(self):
		"""A one-shot function to send our first hardcoded goal."""

		if self.trigger_timer is not None:
			self.trigger_timer.cancel()
			self.trigger_timer = None

		self.get_logger().info("--- EXECUTING HARDCODED GOAL SEQUENCE ---")

		# The coordinates you found by driving manually
		target_x = -4.381
		target_y = 1.710
		target_yaw = 0.77 

		# Use the helper function already in the code to create a proper goal message
		goal_pose = self.create_goal_from_world_coord(target_x, target_y, target_yaw)

		# Now, send this goal to the navigation system
		self.get_logger().info(f"Sending hardcoded goal: x={target_x}, y={target_y}, yaw={target_yaw}")
		success = self.send_goal_from_world_pose(goal_pose)

		if not success:
			self.get_logger().error("Failed to send goal. Is the action server available?")

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates.

		Args:
			message: ROS2 message containing shelf objects data.

		Returns:
			None
		"""
		self.shelf_objects_curr = message
		# Process the shelf objects as needed.

		# How to send WarehouseShelf messages for evaluation.
		"""
		* Example for sending WarehouseShelf messages for evaluation.
			shelf_data_message = WarehouseShelf()

			shelf_data_message.object_name = ["car", "clock"]
			shelf_data_message.object_count = [1, 2]
			shelf_data_message.qr_decoded = "test qr string"

			self.publisher_shelf_data.publish(shelf_data_message)

		* Alternatively, you may store the QR for current shelf as self.qr_code_str.
			Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
			Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
			This, will publish the current detected objects with the last QR decoded.
		"""

		# Optional code for populating TABLE GUI with detected objects and QR data.
		"""
		if PROGRESS_TABLE_GUI:
			shelf = self.shelf_objects_curr
			obj_str = ""
			for name, count in zip(shelf.object_name, shelf.object_count):
				obj_str += f"{name}: {count}\n"

			box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
			self.table_row_count += 1

			box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
			self.table_row_count = 0
			self.table_col_count += 1
		"""

	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)
		
	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
		"""Callback executed when a goal is done. All it does is update the status."""
		status = future.result().status
		self.goal_completed = True
		self.goal_handle_curr = None

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.get_logger().info("Navigation step completed successfully.")
		else:
			self.get_logger().warn(f"Navigation step failed with status: {status}.")
	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
