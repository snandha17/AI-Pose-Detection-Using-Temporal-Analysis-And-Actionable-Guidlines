import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# 📱 YOUR IP CAMERA SOURCE
# ==========================================
CAMERA_SOURCE = "https://192.0.0.4:8080/video"

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA_SOURCE)

# 🛠️ CRASH DETECTOR: If the camera fails, it will tell you exactly why!
if not cap.isOpened():
    print("=====================================================")
    print(f"🚨 ERROR: Could not connect to the camera at:")
    print(f"   {CAMERA_SOURCE}")
    print("   Make sure the IP Webcam app is running on your phone!")
    print("=====================================================")
    time.sleep(5)
    exit()

print("Starting Calibration...")
start_time = time.time()
prep_duration = 10 # ⏳ 10 seconds to step back
cal_duration = 10  # ⏳ 10 seconds to record the squat
lowest_angle = 180.0
mode = "PREP"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    current_time = time.time()
    elapsed = int(current_time - start_time)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0

    # ⏳ PHASE 1: PREPARATION TIMER
    if mode == "PREP":
        time_left = prep_duration - elapsed
        cv2.putText(frame, f"GET IN POSITION!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(frame, f"Starting in: {time_left}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        if time_left <= 0:
            mode = "CALIBRATE"
            start_time = time.time() 
            
    # 🎯 PHASE 2: CALIBRATION TIMER
    elif mode == "CALIBRATE":
        time_left = cal_duration - int(current_time - start_time)
        
        if 'knee_angle' in locals() and knee_angle < lowest_angle and knee_angle > 60: 
            lowest_angle = knee_angle

        cv2.putText(frame, f"SQUAT DOWN AND HOLD!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
        cv2.putText(frame, f"Time left: {time_left}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Lowest Angle: {int(lowest_angle)}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if time_left <= 0:
            break

    cv2.imshow('Calibration Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the final target to memory
with open("user_calibration.txt", "w") as f:
    f.write(str(lowest_angle))

print(f"Calibration Complete! New target saved: {lowest_angle}")


'streamlit run app.py'