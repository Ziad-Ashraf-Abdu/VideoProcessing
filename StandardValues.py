import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from PlayerEaluation import ServeDataProcessor

# Initialize Mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Helper function to compute Euclidean distance
def compute_distance(a, b):
    return np.linalg.norm(a - b)


# Helper function to compute angle between three points
def compute_angle(a, b, c):
    ba = a - b  # vector from b to a
    bc = c - b  # vector from b to c
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
def compute_temporal_peak(segment_data):
    # Check if segment_data is empty
    if segment_data is None or len(segment_data) == 0:  # Works for both lists and arrays
        print("Warning: segment_data is empty. Cannot compute peak.")
        return None  # Return a default value or handle the case appropriately

    # Safely compute the peak
    peak_frame = max(range(len(segment_data)), key=lambda i: segment_data[i])
    return peak_frame
# Analyze video and extract biomechanical parameters (updated to include Kinetic Chain, Trunk Rotation, etc.)
def analyze_video(video_path, output_csv, output_video, output_frames_folder, rotate=False,
                  rotation_code=cv2.ROTATE_180):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    os.makedirs(output_frames_folder, exist_ok=True)  # Create output folder if it doesn't exist

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frames_data = []  # List to store frame-wise data for CSV export
    serve_count = 0  # Counter for detected serves
    serve_start_frame = None  # Frame index marking the start of a serve
    is_serve_active = False  # Flag indicating whether a serve is in progress
    cooldown_counter = 0  # Cooldown counter to prevent multiple detections in quick succession
    cooldown_frames = int(fps * 0.5)  # e.g., 0.5 seconds cooldown

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally rotate frame if the video orientation is off.
        if rotate:
            frame = cv2.rotate(frame, rotation_code)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Compute parameters for both left and right sides
            sides = {"left": {}, "right": {}}  # Dictionary to store values for both sides
            for side, prefix in [("left", "LEFT"), ("right", "RIGHT")]:
                wrist = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_WRIST").value].x * frame_width,
                                  landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_WRIST").value].y * frame_height])
                elbow = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_ELBOW").value].x * frame_width,
                                  landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_ELBOW").value].y * frame_height])
                shoulder = np.array(
                    [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_SHOULDER").value].x * frame_width,
                     landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_SHOULDER").value].y * frame_height])
                hip = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_HIP").value].x * frame_width,
                                landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_HIP").value].y * frame_height])
                knee = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_KNEE").value].x * frame_width,
                                 landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_KNEE").value].y * frame_height])
                foot = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_FOOT_INDEX").value].x * frame_width,
                                 landmarks[
                                     getattr(mp_pose.PoseLandmark, f"{prefix}_FOOT_INDEX").value].y * frame_height])
                hand = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_INDEX").value].x * frame_width,
                                 landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_INDEX").value].y * frame_height])

                # Compute biomechanical parameters
                sides[side]["wrist_to_hip"] = compute_distance(wrist, hip)
                sides[side]["shoulder_angle"] = compute_angle(elbow, shoulder, hip)
                sides[side]["trunk_rotation"] = compute_angle(shoulder, hip, foot)
                sides[side]["knee_flexion"] = compute_angle(hip, knee, foot)
                sides[side]["wrist_flexion"] = compute_angle(elbow, wrist, hand)  # New
                sides[side]["leg_drive"] = compute_distance(hip, foot)  # Representing leg contribution
                sides[side]["impact_height"] = shoulder[1]
                sides[side]["step_distance"] = compute_distance(hip, foot)

                # Analyzing temporal data (e.g., peak angles/velocities)
                sides[side]["shoulder_peak_time"] = compute_temporal_peak(shoulder)  # Add helper
                sides[side]["hip_peak_time"] = compute_temporal_peak(hip)  # Add helper
                sides[side]["knee_peak_time"] = compute_temporal_peak(knee)  # Add helper
                sides[side]["wrist_peak_time"] = compute_temporal_peak(
                    [wrist[1] for _ in range(frame_idx)])

                # Analyze sequential contribution timings
                sides[side]["coordination_timing"] = (sides[side]["hip_peak_time"] <=
                                                      sides[side]["shoulder_peak_time"] <=
                                                      sides[side]["wrist_peak_time"])

                # Compute expanded kinetic chain using segmental contributions
                sides[side]["kinetic_chain"] = (sides[side]["trunk_rotation"] +
                                                sides[side]["shoulder_angle"] +
                                                sides[side]["knee_flexion"] +
                                                sides[side]["wrist_flexion"] +
                                                sides[side]["leg_drive"])

                # If coordination fails or is out-of-sequence, flag potential issues
                if not sides[side]["coordination_timing"]:
                    sides[side]["kinetic_chain_flagged"] = True  # Indicates timing breakdown
                else:
                    sides[side]["kinetic_chain_flagged"] = False

            # Decide active side based on Kinetic Chain index
            active_side = "left" if sides["left"]["kinetic_chain"] >= sides["right"]["kinetic_chain"] else "right"
            chosen = sides[active_side]

            # Serve detection logic
            if cooldown_counter <= 0:
                if not is_serve_active and chosen["wrist_to_hip"] > 50 and chosen["shoulder_angle"] > 10:
                    serve_start_frame = frame_idx
                    is_serve_active = True
                if is_serve_active and chosen["wrist_to_hip"] < 50 and frame_idx - serve_start_frame > 10:
                    serve_count += 1
                    is_serve_active = False
                    cooldown_counter = cooldown_frames

            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Annotate frame with serve count and key biomechanical parameters
            cv2.putText(frame, f"Serve Count: {serve_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_serve_active:
                cv2.putText(frame, "Serve in Progress", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Append chosen parameters to frame data
            frames_data.append({
                "Frame": frame_idx,
                "Serve Count": serve_count,
                # "Active Side": active_side,
                **{f"{key} (pixels)" if "distance" in key else f"{key} (degrees)": value for key, value in
                   chosen.items()},
            })

            # Draw landmarks and computations for visualization
            # Annotate the frame with calculated biomechanical parameters
            for side, prefix in [("left", "LEFT"), ("right", "RIGHT")]:
                # Visualize the relevant landmarks
                wrist = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_WRIST").value].x * frame_width,
                                  landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_WRIST").value].y * frame_height])
                elbow = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_ELBOW").value].x * frame_width,
                                  landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_ELBOW").value].y * frame_height])
                shoulder = np.array(
                    [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_SHOULDER").value].x * frame_width,
                     landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_SHOULDER").value].y * frame_height])
                hip = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_HIP").value].x * frame_width,
                                landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_HIP").value].y * frame_height])
                foot = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_FOOT_INDEX").value].x * frame_width,
                                 landmarks[
                                     getattr(mp_pose.PoseLandmark, f"{prefix}_FOOT_INDEX").value].y * frame_height])

                # Draw biomechanical angles
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(shoulder[0]), int(shoulder[1])), (0, 255, 0), 2)
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 2)

                cv2.putText(frame, f"{side.capitalize()} Shoulder Angle: {int(sides[side]['shoulder_angle'])} deg",
                            (int(shoulder[0]), int(shoulder[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            2)
                # cv2.putText(frame, f"{side.capitalize()} Trunk Rotation: {int(sides[side]['trunk_rotation'])} deg",
                #             (int(hip[0]), int(hip[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw distance for wrist to hip
                # cv2.line(frame, (int(wrist[0]), int(wrist[1])), (int(hip[0]), int(hip[1])), (255, 0, 0), 2)
                # cv2.putText(frame, f"{side.capitalize()} Wrist-Hip Distance: {int(sides[side]['wrist_to_hip'])} px",
                #             (int(wrist[0]), int(wrist[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw landmarks and connections, preserving the existing visualization
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

            # Annotate frame with serve count and serve status
            cv2.putText(frame, f"Serve Count: {serve_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_serve_active:
                cv2.putText(frame, "Serve in Progress", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        else:
            print(f"No landmarks detected in frame {frame_idx}")

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    pose.close()

    # Export collected data to CSV
    if frames_data:
        df = pd.DataFrame(frames_data)

        # Ensure only numeric data is exported (drops non-numeric columns)
        numeric_df = df.select_dtypes(include=[np.number])

        # Optional: If boolean data is included, convert it to integers (True -> 1, False -> 0)
        for column in df.columns:
            if df[column].dtype == 'bool':
                numeric_df[column] = df[column].astype(int)

        numeric_df.to_csv(output_csv, index=False)
        print(f"Analysis complete. Data exported to {output_csv}.")
    else:
        print("No data to export.")


# File paths
video_path = "Inter.mp4"
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_csv = f"output/biomechanical_data_{base_name}.csv"
output_video = f"annotated_{base_name}.mp4"
output_frames_folder = f"frames/{base_name}"
standard_file = f"standard/parameter_ranges_power.csv"
output_dir = f"player_analysis"

# Set rotation flag if needed (False if the video is correctly oriented)
rotate_flag = False
rotation_code = cv2.ROTATE_180  # Adjust if a different rotation is required

# Run analysis and then process the extracted data
analyze_video(video_path, output_csv, output_video, output_frames_folder,
              rotate=rotate_flag, rotation_code=rotation_code)
# processor = ServeDataProcessor(output_csv, output_dir, standard_file)
# processor.process_file()
