import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize Mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper function to compute Euclidean distance
def compute_distance(a, b):
    return np.linalg.norm(a - b)

# Helper function to compute angles between three points
def compute_angle(a, b, c):
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # Cosine of angle between ba and bc
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Angle in radians, clipped to avoid floating-point errors
    return np.degrees(angle)  # Convert angle to degrees

# Analyze video and extract parameters
def analyze_video(video_path, output_csv, output_video, output_frames_folder):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    if not cap.isOpened():
        print("Error: Cannot open video.")  # If video cannot be opened, return
        return

    # Create output directory for frames if not exists
    os.makedirs(output_frames_folder, exist_ok=True)

    # Video writer for annotated video
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Define codec for video writing
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width of frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height of frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second of video
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))  # Set up the output video file

    frames_data = []  # List to store frame-wise data for CSV export
    serve_count = 0  # Counter for serves detected in the video
    serve_start_frame = None  # Frame when the serve starts
    is_serve_active = False  # Boolean flag to indicate if a serve is currently active
    cooldown_counter = 0  # Counter for cooldown between consecutive serves to avoid multiple serves in quick succession
    cooldown_frames = int(fps * 0.5)  # Set cooldown to 0.5 seconds (in terms of frames)

    # Iterate through all frames in the video
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()  # Read the current frame
        if not ret:
            break  # If no frame is returned, break the loop

        # Convert frame to RGB as Mediapipe works in RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)  # Process the frame with Mediapipe Pose

        if results.pose_landmarks:  # If pose landmarks are detected
            landmarks = results.pose_landmarks.landmark  # List of pose landmarks
            lm_array = np.array([(lm.x * frame_width, lm.y * frame_height) for lm in landmarks])  # Convert landmarks to 2D points on the frame

            # Draw pose landmarks and connections on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract relevant body positions (left wrist, elbow, shoulder, hip, and foot)
            wrist = lm_array[mp_pose.PoseLandmark.LEFT_WRIST.value]  # Left wrist coordinates
            elbow = lm_array[mp_pose.PoseLandmark.LEFT_ELBOW.value]  # Left elbow coordinates
            shoulder = lm_array[mp_pose.PoseLandmark.LEFT_SHOULDER.value]  # Left shoulder coordinates
            hip = lm_array[mp_pose.PoseLandmark.LEFT_HIP.value]  # Left hip coordinates
            foot = lm_array[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]  # Left foot coordinates

            # Calculate additional biomechanical parameters
            wrist_to_hip_distance = compute_distance(wrist, hip)  # Calculate distance from wrist to hip
            shoulder_angle = compute_angle(elbow, shoulder, hip)  # Calculate angle at shoulder
            trunk_rotation = compute_angle(shoulder, hip, foot)  # Calculate trunk rotation angle
            wrist_flexion = compute_angle(elbow, wrist, shoulder)  # Calculate wrist flexion angle

            # Detect circular motion (serve start condition)
            if cooldown_counter <= 0:  # If cooldown has passed
                if not is_serve_active and wrist_to_hip_distance > 50 and shoulder_angle > 5:
                    serve_start_frame = frame_idx  # Mark the current frame as the serve start
                    is_serve_active = True  # Set serve to active

                if is_serve_active and wrist_to_hip_distance < 50 and frame_idx - serve_start_frame > 10:
                    serve_count += 1  # Increment the serve count
                    is_serve_active = False  # End the current serve
                    cooldown_counter = cooldown_frames  # Reset cooldown

            if cooldown_counter > 0:  # If cooldown is active, decrement the counter
                cooldown_counter -= 1

            # Annotate the frame with the serve count and status
            cv2.putText(frame, f"Serve Count: {serve_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_serve_active:
                cv2.putText(frame, "Serve in Progress", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save frame data for CSV export
            frames_data.append({
                "Frame": frame_idx,
                "Serve Count": serve_count,
                "Wrist to Hip Distance": wrist_to_hip_distance,
                "Shoulder Angle": shoulder_angle,
                "Trunk Rotation": trunk_rotation,
                "Wrist Flexion": wrist_flexion,
            })

            # Save every 30th frame for manual review
            if frame_idx % 30 == 0:
                cv2.imwrite(f"{output_frames_folder}/frame_{frame_idx}.jpg", frame)

        # Write the annotated frame to the output video
        out.write(frame)

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    pose.close()  # Close Mediapipe pose solution

    # Save results to CSV
    if frames_data:
        df = pd.DataFrame(frames_data)  # Convert the frame data to a DataFrame
        df.to_csv(output_csv, index=False)  # Save to CSV
        print(f"Analysis complete. Data exported to {output_csv}.")
    else:
        print("No data to export.")

# File paths
video_path = "Abdelrahman.mp4"  # Path to the input video
output_csv = "biomechanical_data.csv"  # Path to save the biomechanical data as CSV
output_video = "annotated_video.mp4"  # Path to save the annotated video
output_frames_folder = "frames"  # Folder to save frames for review

# Run the video analysis
analyze_video(video_path, output_csv, output_video, output_frames_folder)
