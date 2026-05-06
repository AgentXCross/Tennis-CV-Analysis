from utils import (save_video, 
                   read_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from minicourt import MiniCourt
import cv2

def main():
    # Read Video
    input_video_path = "input-videos/input_video.mp4"
    video_frames, fps = read_video(input_video_path)

    # Detect and Track: Players and Ball
    player_tracker = PlayerTracker(model_path = "yolov8x.pt")
    ball_tracker = BallTracker(model_path = "models/yolo5_last.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub = True,
        stub_path = "tracker_stubs/player_detections.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub = True,
        stub_path = "tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect Court Keypoints
    court_model_path = "models/keypoints_model.pt"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose the 2 Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini-Court
    mini_court = MiniCourt(video_frames[0])

    # Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, player_detections)

    # Convert Positions to Mini-Court Positions
    player_mini_court_detections, ball_mini_court_detection = mini_court.convert_bbox_to_mini_court_coords(player_detections, ball_detections, court_keypoints)

    # Draw Outputs
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw Frame Number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame #{i + 1}", (50, 170), cv2.FONT_HERSHEY_TRIPLEX, 2, (1, 255, 214), 5)

    # Save Outputs
    save_video(output_video_frames, "output-videos/output_video.avi", fps)

if __name__ == "__main__":
    main()