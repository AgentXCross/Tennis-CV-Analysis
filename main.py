from utils import (save_video, 
                   read_video)

def main():
    input_video_path = "input-videos/input_video.mp4"
    video_frames, fps = read_video(input_video_path)

    save_video(video_frames, "output-videos/output_video.avi", fps)

if __name__ == "__main__":
    main()