import cv2
import numpy as np
import pandas as pd
import os
import sys

# ----- Setup OpenPose -----
# Adjust these paths to your OpenPose installation
try:
    sys.path.append('/path/to/openpose/python')  # Update with your OpenPose python folder
    from openpose import pyopenpose as op
except ImportError as e:
    print("Error: OpenPose library not found. Did you enable the `BUILD_PYTHON` flag?")
    raise e

params = dict()
params["model_folder"] = "/path/to/openpose/models/"  # Update with your OpenPose models folder
params["model_pose"] = "BODY_25"  # Using the BODY_25 model
# You can adjust additional parameters (e.g., resolution, keypoint thresholds) as needed

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# ----- Helper functions -----
def compute_distance(a, b):
    return np.linalg.norm(a - b)


def compute_angle(a, b, c):
    ba = a - b  # vector from b to a
    bc = c - b  # vector from b to c
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def compute_temporal_peak(segment_data):
    if segment_data is None or len(segment_data) == 0:
        print("Warning: segment_data is empty. Cannot compute peak.")
        return None
    peak_frame = max(range(len(segment_data)), key=lambda i: segment_data[i])
    return peak_frame


# ----- Video Analysis Function using OpenPose -----
def analyze_video(video_path, output_csv, output_video, output_frames_folder, rotate=False,
                  rotation_code=cv2.ROTATE_180):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    os.makedirs(output_frames_folder, exist_ok=True)

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frames_data = []  # Store data for CSV export
    serve_count = 0
    serve_start_frame = None
    is_serve_active = False
    cooldown_counter = 0
    cooldown_frames = int(fps * 0.5)

    frame_idx = 0
    conf_threshold = 0.1  # Confidence threshold for accepting a keypoint

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, rotation_code)

        # Process the frame with OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # datum.poseKeypoints is a numpy array of shape (num_people, num_keypoints, 3)
        keypoints = datum.poseKeypoints

        if keypoints is not None and len(keypoints) > 0:
            person = keypoints[0]  # Assume the first person (your tennis player)

            # Define a helper to extract and scale a point if confidence is high enough
            def get_point(idx):
                pt = person[idx]
                if pt[2] < conf_threshold:
                    return None
                return np.array([pt[0] * frame_width, pt[1] * frame_height])

            # For BODY_25 the keypoints indices are:
            # Right side: RShoulder=2, RElbow=3, RWrist=4, RHip=9, RKnee=10, RAnkle=11
            # Left side: LShoulder=5, LElbow=6, LWrist=7, LHip=12, LKnee=13, LAnkle=14
            right_wrist = get_point(4)
            right_elbow = get_point(3)
            right_shoulder = get_point(2)
            right_hip = get_point(9)
            right_knee = get_point(10)
            right_ankle = get_point(11)

            left_wrist = get_point(7)
            left_elbow = get_point(6)
            left_shoulder = get_point(5)
            left_hip = get_point(12)
            left_knee = get_point(13)
            left_ankle = get_point(14)

            # If critical keypoints are missing on both sides, skip the frame
            if (
                    right_wrist is None or right_elbow is None or right_shoulder is None or right_hip is None or right_knee is None or right_ankle is None) and \
                    (
                            left_wrist is None or left_elbow is None or left_shoulder is None or left_hip is None or left_knee is None or left_ankle is None):
                print(f"Missing keypoints in frame {frame_idx}")
                frame_idx += 1
                continue

            sides = {"left": {}, "right": {}}

            # Process left side if available
            if left_wrist is not None and left_elbow is not None and left_shoulder is not None and \
                    left_hip is not None and left_knee is not None and left_ankle is not None:
                wrist = left_wrist
                elbow = left_elbow
                shoulder = left_shoulder
                hip = left_hip
                knee = left_knee
                foot = left_ankle  # using ankle as foot point
                hand = left_wrist  # fallback (or add hand model if available)

                sides["left"]["wrist_to_hip"] = compute_distance(wrist, hip)
                sides["left"]["shoulder_angle"] = compute_angle(elbow, shoulder, hip)
                sides["left"]["trunk_rotation"] = compute_angle(shoulder, hip, foot)
                sides["left"]["knee_flexion"] = compute_angle(hip, knee, foot)
                sides["left"]["wrist_flexion"] = compute_angle(elbow, wrist, hand)
                sides["left"]["leg_drive"] = compute_distance(hip, foot)
                sides["left"]["impact_height"] = shoulder[1]
                sides["left"]["step_distance"] = compute_distance(hip, foot)
                # Temporal placeholders (for demonstration)
                sides["left"]["shoulder_peak_time"] = compute_temporal_peak([sides["left"]["shoulder_angle"]])
                sides["left"]["hip_peak_time"] = compute_temporal_peak([sides["left"]["impact_height"]])
                sides["left"]["knee_peak_time"] = compute_temporal_peak([sides["left"]["knee_flexion"]])
                sides["left"]["wrist_peak_time"] = compute_temporal_peak([sides["left"]["wrist_to_hip"]])
                sides["left"]["coordination_timing"] = (sides["left"]["hip_peak_time"] <=
                                                        sides["left"]["shoulder_peak_time"] <=
                                                        sides["left"]["wrist_peak_time"])
                sides["left"]["kinetic_chain"] = (sides["left"]["trunk_rotation"] +
                                                  sides["left"]["shoulder_angle"] +
                                                  sides["left"]["knee_flexion"] +
                                                  sides["left"]["wrist_flexion"] +
                                                  sides["left"]["leg_drive"])
                sides["left"]["kinetic_chain_flagged"] = not sides["left"]["coordination_timing"]

            # Process right side if available
            if right_wrist is not None and right_elbow is not None and right_shoulder is not None and \
                    right_hip is not None and right_knee is not None and right_ankle is not None:
                wrist = right_wrist
                elbow = right_elbow
                shoulder = right_shoulder
                hip = right_hip
                knee = right_knee
                foot = right_ankle
                hand = right_wrist  # fallback

                sides["right"]["wrist_to_hip"] = compute_distance(wrist, hip)
                sides["right"]["shoulder_angle"] = compute_angle(elbow, shoulder, hip)
                sides["right"]["trunk_rotation"] = compute_angle(shoulder, hip, foot)
                sides["right"]["knee_flexion"] = compute_angle(hip, knee, foot)
                sides["right"]["wrist_flexion"] = compute_angle(elbow, wrist, hand)
                sides["right"]["leg_drive"] = compute_distance(hip, foot)
                sides["right"]["impact_height"] = shoulder[1]
                sides["right"]["step_distance"] = compute_distance(hip, foot)
                sides["right"]["shoulder_peak_time"] = compute_temporal_peak([sides["right"]["shoulder_angle"]])
                sides["right"]["hip_peak_time"] = compute_temporal_peak([sides["right"]["impact_height"]])
                sides["right"]["knee_peak_time"] = compute_temporal_peak([sides["right"]["knee_flexion"]])
                sides["right"]["wrist_peak_time"] = compute_temporal_peak([sides["right"]["wrist_to_hip"]])
                sides["right"]["coordination_timing"] = (sides["right"]["hip_peak_time"] <=
                                                         sides["right"]["shoulder_peak_time"] <=
                                                         sides["right"]["wrist_peak_time"])
                sides["right"]["kinetic_chain"] = (sides["right"]["trunk_rotation"] +
                                                   sides["right"]["shoulder_angle"] +
                                                   sides["right"]["knee_flexion"] +
                                                   sides["right"]["wrist_flexion"] +
                                                   sides["right"]["leg_drive"])
                sides["right"]["kinetic_chain_flagged"] = not sides["right"]["coordination_timing"]

            # Choose the active side based on the kinetic chain index
            if "left" in sides and "right" in sides:
                active_side = "left" if sides["left"].get("kinetic_chain", 0) >= sides["right"].get("kinetic_chain",
                                                                                                    0) else "right"
            elif "left" in sides:
                active_side = "left"
            elif "right" in sides:
                active_side = "right"
            else:
                active_side = None

            if active_side is None:
                frame_idx += 1
                continue

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

            # Annotate the frame with serve count and biomechanical parameters
            cv2.putText(frame, f"Serve Count: {serve_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_serve_active:
                cv2.putText(frame, "Serve in Progress", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frames_data.append({
                "Frame": frame_idx,
                "Serve Count": serve_count,
                **{f"{key} (pixels)" if "distance" in key else f"{key} (degrees)": value
                   for key, value in chosen.items()},
            })

            # Optional: Draw keypoints for visualization
            def draw_point(point, color=(0, 255, 0)):
                if point is not None:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)

            # Left side keypoints
            draw_point(left_wrist, (255, 0, 0))
            draw_point(left_elbow, (0, 255, 0))
            draw_point(left_shoulder, (0, 0, 255))
            draw_point(left_hip, (255, 255, 0))
            draw_point(left_knee, (255, 0, 255))
            draw_point(left_ankle, (0, 255, 255))
            # Right side keypoints
            draw_point(right_wrist, (255, 0, 0))
            draw_point(right_elbow, (0, 255, 0))
            draw_point(right_shoulder, (0, 0, 255))
            draw_point(right_hip, (255, 255, 0))
            draw_point(right_knee, (255, 0, 255))
            draw_point(right_ankle, (0, 255, 255))

        else:
            print(f"No pose keypoints detected in frame {frame_idx}")

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Export collected data to CSV
    if frames_data:
        df = pd.DataFrame(frames_data)
        numeric_df = df.select_dtypes(include=[np.number])
        for column in df.columns:
            if df[column].dtype == 'bool':
                numeric_df[column] = df[column].astype(int)
        numeric_df.to_csv(output_csv, index=False)
        print(f"Analysis complete. Data exported to {output_csv}.")
    else:
        print("No data to export.")


# ----- File paths and execution -----
video_path = "Inter.mp4"
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_csv = f"output/biomechanical_data_{base_name}.csv"
output_video = f"annotated_{base_name}.mp4"
output_frames_folder = f"frames/{base_name}"
# standard_file and output_dir are not used in this snippet but can be integrated as needed

rotate_flag = False
rotation_code = cv2.ROTATE_180

analyze_video(video_path, output_csv, output_video, output_frames_folder,
              rotate=rotate_flag, rotation_code=rotation_code)
