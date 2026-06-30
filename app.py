import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import subprocess 
import sys # 🛠️ CRITICAL FOR LAUNCHING CALIBRATION

# ==========================================
# 📱 SET YOUR CAMERA SOURCE HERE
# ==========================================
CAMERA_SOURCE = "https://192.0.0.4:8080/video"
# ==========================================
# 🗣️ THE BULLETPROOF AUDIO COACH
# ==========================================
def speak(text):
    try:
        script = f"""
import pyttsx3
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.say("{text}")
    engine.runAndWait()
except:
    pass
"""
        subprocess.Popen([sys.executable, "-c", script], creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        pass

# ==========================================
# 🎨 STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    div.stButton > button:first-child {
        background-color: #1e293b; color: #f8fafc; border: 2px solid #3b82f6;
        border-radius: 12px; font-size: 18px; font-weight: bold; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #3b82f6; color: #ffffff; border-color: #3b82f6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
    .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
    .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
    .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
    .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
    .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧩 YOUR CUSTOM AI FUNCTIONS
# ==========================================
def get_branch_features(x, landmark_indices):
    feature_indices = []
    for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
    return tf.gather(x, feature_indices, axis=2)

def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================
# 🚀 LOAD AI MODEL & MEDIAPIPE
# ==========================================
@st.cache_resource
def load_ai():
    try:
        model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
        return model
    except Exception:
        return None

model = load_ai()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, smooth_landmarks=True, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 🌐 STREAMLIT UI LAYOUT & MEMORY
# ==========================================
st.title("🟦 POSEGURU | Clinical Rehab AI")

if 'target_angle' not in st.session_state:
    if os.path.exists("user_calibration.txt"):
        with open("user_calibration.txt", "r") as f:
            st.session_state.target_angle = float(f.read().strip())
    else:
        st.session_state.target_angle = 120.0

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Tracking")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Clinical Metrics")
    reps_placeholder = st.empty()
    angle_placeholder = st.empty()
    
    goal_placeholder = st.empty()
    goal_placeholder.markdown(f"""
        <div class="metric-card">
            <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
            <p class="goal-value">{int(st.session_state.target_angle)}°</p>
        </div>
    """, unsafe_allow_html=True)
    
    start_btn = st.button("▶ START CAMERA", use_container_width=True)
    stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
    cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

    if start_btn:
        st.session_state.camera_active = True
        speak("Camera started. Ready when you are.")
        
    if stop_btn:
        st.session_state.camera_active = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        speak("Workout paused.")

# ==========================================
# 🎯 TRIGGER YOUR CALIBRATION SCRIPT
# ==========================================
if cal_btn:
    st.session_state.camera_active = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        
    with st.spinner("Switching cameras... Please wait 2 seconds!"):
        speak("Switching to calibration mode.")
        time.sleep(2) 
        
        # 🛠️ FIXED: Uses sys.executable to prevent silent crashes in Windows!
        subprocess.run([sys.executable, "calibrate_user.py"])
        
        if os.path.exists("user_calibration.txt"):
            with open("user_calibration.txt", "r") as f:
                new_target = float(f.read().strip())
                st.session_state.target_angle = new_target 
                speak(f"Calibration complete. New target is {int(new_target)} degrees.")
                
    st.rerun()

# ==========================================
# 🎥 THE AI CAMERA LOOP
# ==========================================
if st.session_state.camera_active:
    
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(CAMERA_SOURCE)
        st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    sequence_buffer = []
    squat_state = "up"
    reps = 0
    frame_count = 0
    st.session_state.current_knee_angle = 180.0
    
    last_spoken_status = ""
    last_spoken_time = 0

    while st.session_state.cap is not None and st.session_state.cap.isOpened() and st.session_state.camera_active:
        success, frame = st.session_state.cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        if frame_count % 2 != 0:
            continue 

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        status_text = "WAITING..."
        status_class = "status-idle"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            visibility_scores = [lm[0].visibility, lm[11].visibility, lm[12].visibility, lm[23].visibility, lm[24].visibility, lm[27].visibility, lm[28].visibility]
            
            nose_y = lm[0].y
            ankle_y = (lm[27].y + lm[28].y) / 2.0
            body_height = abs(ankle_y - nose_y) 

            if min(visibility_scores) < 0.65 or body_height < 0.35:
                status_text = "STEP INTO FRAME" 
                status_class = "status-warn"
                sequence_buffer.clear()
                squat_state = "up" 
            else:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                features = []
                for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
                current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
                st.session_state.current_knee_angle = current_knee_angle 
                
                target = st.session_state.target_angle

                features.extend([current_knee_angle, current_hip_angle])
                features.extend([0.0] * (240 - len(features)))
                sequence_buffer.append(features)
                sequence_buffer = sequence_buffer[-30:]

                # Rep Count Logic
                if current_knee_angle > 160 and current_hip_angle > 155:
                    status_text = "IDLE"
                    status_class = "status-idle"
                    if squat_state == "down": 
                        reps += 1
                        squat_state = "up"
                        speak(f"{reps}") 
                else:
                    if current_knee_angle <= (target + 15):
                        squat_state = "down"
                    
                    if current_hip_angle < 75: 
                        status_text = "STRAIGHTEN YOUR BACK"
                        status_class = "status-warn"
                    elif current_knee_angle > (target + 15):
                        status_text = "GO DOWN MORE"
                        status_class = "status-warn"
                    else:
                        status_text = "PERFECT DEPTH" 
                        status_class = "status-good"

        current_time = time.time()
        if status_text not in ["WAITING...", "STEP INTO FRAME", "IDLE"]:
            if status_text != last_spoken_status or (current_time - last_spoken_time > 3.0):
                speak(status_text)
                last_spoken_status = status_text
                last_spoken_time = current_time
        
        if status_text in ["IDLE", "WAITING...", "STEP INTO FRAME"]:
            last_spoken_status = ""

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
        reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
        angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)

# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import subprocess # Allows Streamlit to run your calibration_user.py file!

# # ==========================================
# # 🎨 STREAMLIT PAGE CONFIGURATION
# # ==========================================
# st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

# # Inject Custom CSS for Premium Dark Mode Look
# st.markdown("""
#     <style>
#     /* Base Background */
#     .stApp { background-color: #0f172a; color: #f8fafc; }
    
#     /* 🎨 Premium Dark Mode Buttons */
#     div.stButton > button:first-child {
#         background-color: #1e293b;
#         color: #f8fafc;
#         border: 2px solid #3b82f6;
#         border-radius: 12px;
#         font-size: 18px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#     }
#     div.stButton > button:first-child:hover {
#         background-color: #3b82f6;
#         color: #ffffff;
#         border-color: #3b82f6;
#         box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
#     }
    
#     /* Metric Cards */
#     .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
#     .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
#     .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
    
#     /* Status Colors */
#     .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
#     .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
#     .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
#     </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 🧩 YOUR CUSTOM AI FUNCTIONS
# # ==========================================
# def get_branch_features(x, landmark_indices):
#     feature_indices = []
#     for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
#     return tf.gather(x, feature_indices, axis=2)

# def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

# def calculate_angle(a, b, c):
#     a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     return 360 - angle if angle > 180.0 else angle

# # ==========================================
# # 🚀 LOAD AI MODEL
# # ==========================================
# @st.cache_resource
# def load_ai():
#     try:
#         model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
#         return model
#     except Exception:
#         return None

# model = load_ai()
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
# mp_draw = mp.solutions.drawing_utils

# # ==========================================
# # 🌐 STREAMLIT UI LAYOUT
# # ==========================================
# st.title("🟦 POSEGURU | Clinical Rehab AI")

# # Load saved calibration if it exists on startup
# if 'target_angle' not in st.session_state:
#     if os.path.exists("user_calibration.txt"):
#         with open("user_calibration.txt", "r") as f:
#             st.session_state.target_angle = float(f.read().strip())
#     else:
#         st.session_state.target_angle = 120.0

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("### 📷 Live Tracking")
#     video_placeholder = st.empty()
#     status_placeholder = st.empty()

# with col2:
#     st.markdown("### 📊 Clinical Metrics")
#     reps_placeholder = st.empty()
#     angle_placeholder = st.empty()
    
#     goal_placeholder = st.empty()
#     goal_placeholder.markdown(f"""
#         <div class="metric-card">
#             <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
#             <p class="goal-value">{int(st.session_state.target_angle)}°</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     start_btn = st.button("▶ START CAMERA", use_container_width=True)
#     stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
#     cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

# # ==========================================
# # 🎯 TRIGGER YOUR CALIBRATION SCRIPT
# # ==========================================
# if cal_btn:
#     with st.spinner("Opening Calibration Camera... Get ready!"):
#         # Run your separate calibration script
#         subprocess.run(["python", "calibrate_user.py"])
        
#         # Once your calibration window closes, read the file it just created!
#         if os.path.exists("user_calibration.txt"):
#             with open("user_calibration.txt", "r") as f:
#                 new_target = float(f.read().strip())
#                 st.session_state.target_angle = new_target # Update the AI memory
                
#     # Force the dashboard to refresh so the big green number updates instantly
#     st.rerun()

# # ==========================================
# # 🎥 THE AI CAMERA LOOP
# # ==========================================
# if start_btn:
#     cap = cv2.VideoCapture(0)
    
#     sequence_buffer = []
#     squat_state = "up"
#     reps = 0
#     frame_count = 0
#     st.session_state.current_knee_angle = 180.0

#     while cap.isOpened() and not stop_btn:
#         success, frame = cap.read()
#         if not success: break
        
#         frame = cv2.flip(frame, 1)
#         frame_count += 1
#         results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         status_text = "WAITING..."
#         status_class = "status-idle"

#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
#                 status_text = "BACK UP (SHOW LEGS)"
#                 status_class = "status-warn"
#                 sequence_buffer.clear()
#             else:
#                 features = []
#                 for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
#                 current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
#                 current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
#                 st.session_state.current_knee_angle = current_knee_angle 
                
#                 target = st.session_state.target_angle

#                 features.extend([current_knee_angle, current_hip_angle])
#                 features.extend([0.0] * (240 - len(features)))
#                 sequence_buffer.append(features)
#                 sequence_buffer = sequence_buffer[-30:]

#                 if len(sequence_buffer) == 30 and frame_count % 4 == 0 and model is not None:
#                     class_id = np.argmax(model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0))

#                 if current_knee_angle > 160 and current_hip_angle > 155:
#                     status_text = "IDLE"
#                     status_class = "status-idle"
#                     if squat_state == "down": 
#                         reps += 1
#                         squat_state = "up"
#                 else:
#                     squat_state = "down"
#                     if current_hip_angle < 100: 
#                         status_text = "STRAIGHTEN YOUR BACK"
#                         status_class = "status-warn"
#                     elif current_knee_angle > (target + 5):
#                         status_text = "GO DOWN SLIGHTLY"
#                         status_class = "status-warn"
#                     else:
#                         status_text = "GOOD (ADAPTED REHAB)"
#                         status_class = "status-good"

#         # Update Video UI Live
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
#         # Update Text UI Live
#         status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
#         reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
#         angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)

#     cap.release()




# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import subprocess # Allows Streamlit to run your calibration_user.py file!

# # ==========================================
# # 🎨 STREAMLIT PAGE CONFIGURATION
# # ==========================================
# st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

# # Inject Custom CSS for Premium Dark Mode Look
# st.markdown("""
#     <style>
#     /* Base Background */
#     .stApp { background-color: #0f172a; color: #f8fafc; }
    
#     /* 🎨 Premium Dark Mode Buttons */
#     div.stButton > button:first-child {
#         background-color: #1e293b;
#         color: #f8fafc;
#         border: 2px solid #3b82f6;
#         border-radius: 12px;
#         font-size: 18px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#     }
#     div.stButton > button:first-child:hover {
#         background-color: #3b82f6;
#         color: #ffffff;
#         border-color: #3b82f6;
#         box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
#     }
    
#     /* Metric Cards */
#     .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
#     .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
#     .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
    
#     /* Status Colors */
#     .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
#     .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
#     .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
#     </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 🧩 YOUR CUSTOM AI FUNCTIONS
# # ==========================================
# def get_branch_features(x, landmark_indices):
#     feature_indices = []
#     for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
#     return tf.gather(x, feature_indices, axis=2)

# def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

# def calculate_angle(a, b, c):
#     a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     return 360 - angle if angle > 180.0 else angle

# # ==========================================
# # 🚀 LOAD AI MODEL
# # ==========================================
# @st.cache_resource
# def load_ai():
#     try:
#         model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
#         return model
#     except Exception:
#         return None

# model = load_ai()
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
# mp_draw = mp.solutions.drawing_utils

# # ==========================================
# # 🌐 STREAMLIT UI LAYOUT & MEMORY
# # ==========================================
# st.title("🟦 POSEGURU | Clinical Rehab AI")

# # Load saved calibration if it exists on startup
# if 'target_angle' not in st.session_state:
#     if os.path.exists("user_calibration.txt"):
#         with open("user_calibration.txt", "r") as f:
#             st.session_state.target_angle = float(f.read().strip())
#     else:
#         st.session_state.target_angle = 120.0

# # Add Camera Memory so Start/Stop buttons work correctly
# if 'camera_active' not in st.session_state:
#     st.session_state.camera_active = False

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("### 📷 Live Tracking")
#     video_placeholder = st.empty()
#     status_placeholder = st.empty()

# with col2:
#     st.markdown("### 📊 Clinical Metrics")
#     reps_placeholder = st.empty()
#     angle_placeholder = st.empty()
    
#     goal_placeholder = st.empty()
#     goal_placeholder.markdown(f"""
#         <div class="metric-card">
#             <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
#             <p class="goal-value">{int(st.session_state.target_angle)}°</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     start_btn = st.button("▶ START CAMERA", use_container_width=True)
#     stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
#     cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

#     # 🎛️ Button Logic to update camera memory
#     if start_btn:
#         st.session_state.camera_active = True
#     if stop_btn:
#         st.session_state.camera_active = False

# # ==========================================
# # 🎯 TRIGGER YOUR CALIBRATION SCRIPT
# # ==========================================
# if cal_btn:
#     with st.spinner("Opening Calibration Camera... Get ready!"):
#         # Run your separate calibration script
#         subprocess.run(["python", "calibrate_user.py"])
        
#         # Once your calibration window closes, read the file it just created!
#         if os.path.exists("user_calibration.txt"):
#             with open("user_calibration.txt", "r") as f:
#                 new_target = float(f.read().strip())
#                 st.session_state.target_angle = new_target # Update the AI memory
                
#     # Force the dashboard to refresh so the big green number updates instantly
#     st.rerun()

# # ==========================================
# # 🎥 THE AI CAMERA LOOP
# # ==========================================
# # Check our camera memory instead of just the button!
# if st.session_state.camera_active:
#     cap = cv2.VideoCapture(0)
    
#     sequence_buffer = []
#     squat_state = "up"
#     reps = 0
#     frame_count = 0
#     st.session_state.current_knee_angle = 180.0

#     # The loop will keep running as long as camera_active is True
#     while cap.isOpened() and st.session_state.camera_active:
#         success, frame = cap.read()
#         if not success: break
        
#         frame = cv2.flip(frame, 1)
#         frame_count += 1
#         results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         status_text = "WAITING..."
#         status_class = "status-idle"

#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
#                 status_text = "BACK UP (SHOW LEGS)"
#                 status_class = "status-warn"
#                 sequence_buffer.clear()
#             else:
#                 features = []
#                 for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
#                 current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
#                 current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
#                 st.session_state.current_knee_angle = current_knee_angle 
                
#                 target = st.session_state.target_angle

#                 features.extend([current_knee_angle, current_hip_angle])
#                 features.extend([0.0] * (240 - len(features)))
#                 sequence_buffer.append(features)
#                 sequence_buffer = sequence_buffer[-30:]

#                 if len(sequence_buffer) == 30 and frame_count % 4 == 0 and model is not None:
#                     class_id = np.argmax(model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0))

#                 if current_knee_angle > 160 and current_hip_angle > 155:
#                     status_text = "IDLE"
#                     status_class = "status-idle"
#                     if squat_state == "down": 
#                         reps += 1
#                         squat_state = "up"
#                 else:
#                     squat_state = "down"
#                     if current_hip_angle < 100: 
#                         status_text = "STRAIGHTEN YOUR BACK"
#                         status_class = "status-warn"
#                     elif current_knee_angle > (target + 5):
#                         status_text = "GO DOWN SLIGHTLY"
#                         status_class = "status-warn"
#                     else:
#                         status_text = "GOOD (ADAPTED REHAB)"
#                         status_class = "status-good"

#         # Update Video UI Live
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
#         # Update Text UI Live
#         status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
#         reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
#         angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)

#     cap.release()


# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import subprocess 

# # ==========================================
# # 🎨 STREAMLIT PAGE CONFIGURATION
# # ==========================================
# st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

# # Inject Custom CSS for Premium Dark Mode Look
# st.markdown("""
#     <style>
#     .stApp { background-color: #0f172a; color: #f8fafc; }
#     div.stButton > button:first-child {
#         background-color: #1e293b; color: #f8fafc; border: 2px solid #3b82f6;
#         border-radius: 12px; font-size: 18px; font-weight: bold; transition: all 0.3s ease;
#     }
#     div.stButton > button:first-child:hover {
#         background-color: #3b82f6; color: #ffffff; border-color: #3b82f6;
#         box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
#     }
#     .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
#     .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
#     .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
#     .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
#     .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
#     .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
#     </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 🧩 YOUR CUSTOM AI FUNCTIONS
# # ==========================================
# def get_branch_features(x, landmark_indices):
#     feature_indices = []
#     for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
#     return tf.gather(x, feature_indices, axis=2)

# def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

# def calculate_angle(a, b, c):
#     a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     return 360 - angle if angle > 180.0 else angle

# # ==========================================
# # 🚀 LOAD AI MODEL
# # ==========================================
# @st.cache_resource
# def load_ai():
#     try:
#         model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
#         return model
#     except Exception:
#         return None

# model = load_ai()
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
# mp_draw = mp.solutions.drawing_utils

# # ==========================================
# # 🌐 STREAMLIT UI LAYOUT & MEMORY
# # ==========================================
# st.title("🟦 POSEGURU | Clinical Rehab AI")

# if 'target_angle' not in st.session_state:
#     if os.path.exists("user_calibration.txt"):
#         with open("user_calibration.txt", "r") as f:
#             st.session_state.target_angle = float(f.read().strip())
#     else:
#         st.session_state.target_angle = 120.0

# # 🧠 Save the actual camera hardware to memory
# if 'camera_active' not in st.session_state:
#     st.session_state.camera_active = False
# if 'cap' not in st.session_state:
#     st.session_state.cap = None

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("### 📷 Live Tracking")
#     video_placeholder = st.empty()
#     status_placeholder = st.empty()

# with col2:
#     st.markdown("### 📊 Clinical Metrics")
#     reps_placeholder = st.empty()
#     angle_placeholder = st.empty()
    
#     goal_placeholder = st.empty()
#     goal_placeholder.markdown(f"""
#         <div class="metric-card">
#             <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
#             <p class="goal-value">{int(st.session_state.target_angle)}°</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     start_btn = st.button("▶ START CAMERA", use_container_width=True)
#     stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
#     cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

#     # 🎛️ SMART BUTTON LOGIC
#     if start_btn:
#         st.session_state.camera_active = True
        
#     if stop_btn:
#         st.session_state.camera_active = False
#         # Force the hardware to turn off the light!
#         if st.session_state.cap is not None:
#             st.session_state.cap.release()
#             st.session_state.cap = None

# # ==========================================
# # 🎯 TRIGGER YOUR CALIBRATION SCRIPT
# # ==========================================
# if cal_btn:
#     # Turn off main camera completely so calibration can use it
#     st.session_state.camera_active = False
#     if st.session_state.cap is not None:
#         st.session_state.cap.release()
#         st.session_state.cap = None
        
#     with st.spinner("Opening Calibration Camera... Get ready!"):
#         subprocess.run(["python", "calibrate_user.py"])
#         if os.path.exists("user_calibration.txt"):
#             with open("user_calibration.txt", "r") as f:
#                 new_target = float(f.read().strip())
#                 st.session_state.target_angle = new_target 
#     st.rerun()

# # ==========================================
# # 🎥 THE AI CAMERA LOOP
# # ==========================================
# if st.session_state.camera_active:
#     # Open camera ONLY if it isn't already open in memory
#     if st.session_state.cap is None or not st.session_state.cap.isOpened():
#         st.session_state.cap = cv2.VideoCapture(0)
    
#     sequence_buffer = []
#     squat_state = "up"
#     reps = 0
#     frame_count = 0
#     st.session_state.current_knee_angle = 180.0

#     while st.session_state.cap is not None and st.session_state.cap.isOpened() and st.session_state.camera_active:
#         success, frame = st.session_state.cap.read()
#         if not success: break
        
#         frame = cv2.flip(frame, 1)
#         frame_count += 1
#         results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         status_text = "WAITING..."
#         status_class = "status-idle"

#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
#                 status_text = "BACK UP (SHOW LEGS)"
#                 status_class = "status-warn"
#                 sequence_buffer.clear()
#             else:
#                 features = []
#                 for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
#                 current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
#                 current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
#                 st.session_state.current_knee_angle = current_knee_angle 
                
#                 target = st.session_state.target_angle

#                 features.extend([current_knee_angle, current_hip_angle])
#                 features.extend([0.0] * (240 - len(features)))
#                 sequence_buffer.append(features)
#                 sequence_buffer = sequence_buffer[-30:]

#                 if len(sequence_buffer) == 30 and frame_count % 4 == 0 and model is not None:
#                     class_id = np.argmax(model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0))

#                 if current_knee_angle > 160 and current_hip_angle > 155:
#                     status_text = "IDLE"
#                     status_class = "status-idle"
#                     if squat_state == "down": 
#                         reps += 1
#                         squat_state = "up"
#                 else:
#                     squat_state = "down"
#                     if current_hip_angle < 100: 
#                         status_text = "STRAIGHTEN YOUR BACK"
#                         status_class = "status-warn"
#                     elif current_knee_angle > (target + 5):
#                         status_text = "GO DOWN SLIGHTLY"
#                         status_class = "status-warn"
#                     else:
#                         status_text = "GOOD (ADAPTED REHAB)"
#                         status_class = "status-good"

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
#         status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
#         reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
#         angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)
        
# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import time
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import subprocess 

# # ==========================================
# # 🗣️ THE BULLETPROOF AUDIO COACH
# # ==========================================
# # ==========================================
# # 🗣️ THE BULLETPROOF AUDIO COACH
# # ==========================================
# def speak(text):
#     """
#     Runs a tiny, invisible Python program in the background just to speak the text.
#     This bypasses Windows security blocks and prevents video freezing!
#     """
#     try:
#         # We write a mini Python script as a string
#         script = f"""
# import pyttsx3
# try:
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 170)
#     engine.say("{text}")
#     engine.runAndWait()
# except:
#     pass
# """
#         # We launch it as a completely separate background process!
#         subprocess.Popen(["python", "-c", script], creationflags=subprocess.CREATE_NO_WINDOW)
#     except Exception as e:
#         pass

# # ==========================================
# # 🎨 STREAMLIT PAGE CONFIGURATION
# # ==========================================
# st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

# st.markdown("""
#     <style>
#     .stApp { background-color: #0f172a; color: #f8fafc; }
#     div.stButton > button:first-child {
#         background-color: #1e293b; color: #f8fafc; border: 2px solid #3b82f6;
#         border-radius: 12px; font-size: 18px; font-weight: bold; transition: all 0.3s ease;
#     }
#     div.stButton > button:first-child:hover {
#         background-color: #3b82f6; color: #ffffff; border-color: #3b82f6;
#         box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
#     }
#     .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
#     .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
#     .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
#     .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
#     .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
#     .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
#     </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 🧩 YOUR CUSTOM AI FUNCTIONS
# # ==========================================
# def get_branch_features(x, landmark_indices):
#     feature_indices = []
#     for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
#     return tf.gather(x, feature_indices, axis=2)

# def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

# def calculate_angle(a, b, c):
#     a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     return 360 - angle if angle > 180.0 else angle

# # ==========================================
# # 🚀 LOAD AI MODEL
# # ==========================================
# @st.cache_resource
# def load_ai():
#     try:
#         model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
#         return model
#     except Exception:
#         return None

# model = load_ai()
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
# mp_draw = mp.solutions.drawing_utils

# # ==========================================
# # 🌐 STREAMLIT UI LAYOUT & MEMORY
# # ==========================================
# st.title("🟦 POSEGURU | Clinical Rehab AI")

# if 'target_angle' not in st.session_state:
#     if os.path.exists("user_calibration.txt"):
#         with open("user_calibration.txt", "r") as f:
#             st.session_state.target_angle = float(f.read().strip())
#     else:
#         st.session_state.target_angle = 120.0

# if 'camera_active' not in st.session_state:
#     st.session_state.camera_active = False
# if 'cap' not in st.session_state:
#     st.session_state.cap = None

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("### 📷 Live Tracking")
#     video_placeholder = st.empty()
#     status_placeholder = st.empty()

# with col2:
#     st.markdown("### 📊 Clinical Metrics")
#     reps_placeholder = st.empty()
#     angle_placeholder = st.empty()
    
#     goal_placeholder = st.empty()
#     goal_placeholder.markdown(f"""
#         <div class="metric-card">
#             <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
#             <p class="goal-value">{int(st.session_state.target_angle)}°</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     start_btn = st.button("▶ START CAMERA", use_container_width=True)
#     stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
#     cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

#     if start_btn:
#         st.session_state.camera_active = True
#         speak("Camera started. Ready when you are.")
        
#     if stop_btn:
#         st.session_state.camera_active = False
#         if st.session_state.cap is not None:
#             st.session_state.cap.release()
#             st.session_state.cap = None
#         speak("Workout paused.")

# # ==========================================
# # 🎯 TRIGGER YOUR CALIBRATION SCRIPT
# # ==========================================
# if cal_btn:
#     st.session_state.camera_active = False
#     if st.session_state.cap is not None:
#         st.session_state.cap.release()
#         st.session_state.cap = None
        
#     with st.spinner("Switching cameras... Please wait 2 seconds!"):
#         speak("Switching to calibration mode. Get ready to squat.")
#         time.sleep(2) 
#         subprocess.run(["python", "calibrate_user.py"])
        
#         if os.path.exists("user_calibration.txt"):
#             with open("user_calibration.txt", "r") as f:
#                 new_target = float(f.read().strip())
#                 st.session_state.target_angle = new_target 
#                 speak(f"Calibration complete. New target is {int(new_target)} degrees.")
                
#     st.rerun()

# # ==========================================
# # 🎥 THE AI CAMERA LOOP
# # ==========================================
# if st.session_state.camera_active:
#     if st.session_state.cap is None or not st.session_state.cap.isOpened():
#         st.session_state.cap = cv2.VideoCapture(0)
    
#     sequence_buffer = []
#     squat_state = "up"
#     reps = 0
#     frame_count = 0
#     st.session_state.current_knee_angle = 180.0
    
#     # 🧠 AUDIO MEMORY: So the coach doesn't spam you!
#     last_spoken_status = ""
#     last_spoken_time = 0

#     while st.session_state.cap is not None and st.session_state.cap.isOpened() and st.session_state.camera_active:
#         success, frame = st.session_state.cap.read()
#         if not success: break
        
#         frame = cv2.flip(frame, 1)
#         frame_count += 1
#         results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         status_text = "WAITING..."
#         status_class = "status-idle"

#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
#                 status_text = "BACK UP" # Shortened for speech
#                 status_class = "status-warn"
#                 sequence_buffer.clear()
#             else:
#                 features = []
#                 for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
#                 current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
#                 current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
#                 st.session_state.current_knee_angle = current_knee_angle 
                
#                 target = st.session_state.target_angle

#                 features.extend([current_knee_angle, current_hip_angle])
#                 features.extend([0.0] * (240 - len(features)))
#                 sequence_buffer.append(features)
#                 sequence_buffer = sequence_buffer[-30:]

#                 if current_knee_angle > 160 and current_hip_angle > 155:
#                     status_text = "IDLE"
#                     status_class = "status-idle"
#                     if squat_state == "down": 
#                         reps += 1
#                         squat_state = "up"
#                         speak(f"{reps}") # IT COUNTS YOUR REPS OUT LOUD!
#                 else:
#                     squat_state = "down"
#                     if current_hip_angle < 100: 
#                         status_text = "STRAIGHTEN YOUR BACK"
#                         status_class = "status-warn"
#                     elif current_knee_angle > (target + 5):
#                         status_text = "GO DOWN SLIGHTLY"
#                         status_class = "status-warn"
#                     else:
#                         status_text = "GOOD" # Shortened for speech
#                         status_class = "status-good"

#         # 🗣️ SMART AUDIO TRIGGER LOGIC
#         current_time = time.time()
#         # Only speak if the status changes, OR if you've been stuck on a warning for 3 seconds
#         if status_text != "WAITING..." and status_text != "IDLE":
#             if status_text != last_spoken_status or (current_time - last_spoken_time > 3.0):
#                 speak(status_text)
#                 last_spoken_status = status_text
#                 last_spoken_time = current_time
        
#         # Reset the spoken status if you return to IDLE so it can speak immediately on the next rep
#         if status_text == "IDLE":
#             last_spoken_status = "IDLE"

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
#         status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
#         reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
#         angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)