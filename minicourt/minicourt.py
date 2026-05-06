import cv2
from utils import (
    convert_pixel_distance_to_meters, 
    convert_meters_to_pixel_distance,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_bbox,
    measure_distance
)
import court_dimension_constants
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.rectangle_width = 300
        self.rectangle_height = 600
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters): # Mini
        return convert_meters_to_pixel_distance(
            meters,
            court_dimension_constants.DOUBLE_LINE_WIDTH,
            self.court_width
        )

    def set_court_keypoints(self):
        drawing_key_points = [0] * 28
        # point 0: Doubles top left
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1: Doubles top right
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2: Doubles bottom left
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = int(self.court_start_y) + self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_NET * 2)
        # point 3: Doubles bottom right
        drawing_key_points[6] = drawing_key_points[0] + self.court_width
        drawing_key_points[7] = drawing_key_points[5]
        # point 4: Singles top left
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[9] = drawing_key_points[1]
        # point 5: Singles bottom left
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[11] = drawing_key_points[5]
        # point 6: Singles top right
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[13] = drawing_key_points[3]
        # point 7: Singles bottom right
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[15] = drawing_key_points[7]
        # point 8: Mini-Court top left
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_SERVICE)
        # point 9: Mini-Court top right
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(court_dimension_constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # point 10: Mini-Court bottom left
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_SERVICE)
        # point 11: Mini-Court bottom right
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(court_dimension_constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # point 12: Mini-Court top center
        drawing_key_points[24] = (drawing_key_points[16] + drawing_key_points[18]) // 2
        drawing_key_points[25] = drawing_key_points[17]
        # point 13: Mini-Court bottom center
        drawing_key_points[26] = (drawing_key_points[20] + drawing_key_points[22]) // 2
        drawing_key_points[27] = drawing_key_points[21]
        self.drawing_key_points = drawing_key_points
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3)
        ]

    def set_mini_court_position(self):
        # start is top left, end is bottom right
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        # start is top left, end is bottom right
        self.end_x = frame.shape[1] - self.buffer # frame.shape[1] is width
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.buffer
        self.end_y = self.start_y + self.rectangle_height

    def draw_court(self, frame):
        # Draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 3)

        # Draw Net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 10, 0), 3)

        # Draw Keypoints on Mini-Court
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 7, (215, 150, 0), -1)
        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the background rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.2 # transparency
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_width
    
    def get_court_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coords(
            self, 
            object_position, 
            closest_keypoint, 
            closest_keypoint_index, 
            player_height_in_pixels,
            player_height_in_meters
    ):
        dist_from_keypoint_x_pixels, dist_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_keypoint)
        
        # Convert pixel distance to meters
        dist_from_keypoint_x_meters = convert_pixel_distance_to_meters(
            dist_from_keypoint_x_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )
        dist_from_keypoint_y_meters = convert_pixel_distance_to_meters(
            dist_from_keypoint_y_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )

        # Convert to mini-court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(dist_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(dist_from_keypoint_y_meters)

        closest_mini_court_keypoint = (self.drawing_key_points[closest_keypoint_index * 2], self.self.drawing_key_points[closest_keypoint_index * 2 + 1])

        mini_court_player_position = (
            closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_court_keypoint[1] + mini_court_y_distance_pixels
        )
        return mini_court_player_position
    
    def convert_bbox_to_mini_court_coords(
            self, 
            player_boxes, 
            ball_boxes, 
            original_court_keypoints
    ):
        player_heights = {
            1: court_dimension_constants.PLAYER_1_HEIGHT_METERS,
            2: court_dimension_constants.PLAYER_2_HEIGHT_METERS
        }
        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num]
            ball_position = get_center_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key = lambda x: measure_distance(ball_position, get_center_bbox(player_bbox[x])))

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels
                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_court_keypoints, [4, 5, 6, 7, 12, 13])
                closest_keypoint = (original_court_keypoints[closest_keypoint_index * 2], original_court_keypoints[closest_keypoint_index * 2 + 1])

                # Get player height in pixels (Take a range of frames and take the max to account for lunging or knee bending)
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_bbox[i]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coords(
                    foot_position,
                    closest_keypoint,
                    closest_keypoint_index,
                    max_player_height_pixels,
                    player_heights[player_id]
                )
                output_player_bbox_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoint in pixels
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, original_court_keypoints, [4, 5, 6, 7, 12, 13])
                    closest_keypoint = (original_court_keypoints[closest_keypoint_index * 2], original_court_keypoints[closest_keypoint_index * 2 + 1])
                    mini_court_ball_position = self.get_mini_court_coords(
                        ball_position,
                        closest_keypoint,
                        closest_keypoint_index,
                        max_player_height_pixels,
                        player_heights[player_id]
                    )
                    output_ball_boxes.append(mini_court_ball_position)
            output_player_boxes.append(output_player_bbox_dict)
        return output_player_boxes, output_ball_boxes
