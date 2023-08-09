import cv2
import os
def crop_video_to_size(input_path, output_path, new_height, new_width):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the original video's frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the center coordinates of the video frame
    center_x = frame_width // 2
    center_y = frame_height // 2

    # Calculate the starting and ending points for cropping
    start_x = center_x - (new_width // 2)
    end_x = center_x + (new_width // 2)
    start_y = center_y - (new_height // 2)
    end_y = center_y + (new_height // 2)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[start_y:end_y, start_x:end_x]

        # Resize the cropped frame to the new size
        resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

        # Write the resized frame to the output video
        out.write(resized_frame)

        cv2.imshow('Cropped and Resized Video', resized_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to stop the video
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.system("ffmpeg -i " + output_video_path + " -vcodec libx264 /home/wni1717/dev/OTUMEDAI/test/test1_c.mp4")

# Usage example:
input_video_path = "/home/wni1717/dev/OTUMEDAI/test/mmexport1690326202125.mp4"
output_video_path = "/home/wni1717/dev/OTUMEDAI/test/test1.mp4"
new_height = 200
new_width = 200
crop_video_to_size(input_video_path, output_video_path, new_height, new_width)