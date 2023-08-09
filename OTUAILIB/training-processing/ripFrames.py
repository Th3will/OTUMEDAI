import cv2
import os

def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    interval = 5  # Interval of 0.5 seconds (assuming 30 fps)

    while video.isOpened():
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Extract frames at the desired interval
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"vid1_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count}")

        frame_count += 1

    video.release()

# Path to the input video file
video_path = "/home/wni1717/dev/OTUMEDAI/VNN/uploaded_videos/vid_1.mp4"

# Directory to save the extracted frames
output_dir = "/home/wni1717/dev/OTUMEDAI/VNN/output"

extract_frames(video_path, output_dir)
