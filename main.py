import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import threading
import subprocess
import webbrowser

# ============================================================================
# GAZELINK v3.8 - CORNER FIX + CLEAN UI
# ============================================================================

# --- ULTRA-SLOW CONFIGURATION ---
SMOOTHING = 12
CURSOR_SPEED = 0.12
MOUSE_DURATION = 0.08
BLINK_THRESHOLD = 0.20
DOUBLE_BLINK_WINDOW = 0.5
BLINK_DURATION_MAX = 0.35
MOUTH_OPEN_THRESHOLD = 0.55
MOUTH_GOOGLE_DURATION = 3.5
MOUTH_CONFIRM_FRAMES = 10
HEAD_TILT_SENSITIVITY = 0.002
GOOGLE_COOLDOWN = 5.0

# --- CURSOR FREEZE ---
cursor_frozen = False
frozen_x, frozen_y = 0, 0

# --- SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.75, min_tracking_confidence=0.75)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.005
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- STATE ---
class ControlState:
    def __init__(self):
        self.cursor_queue = deque(maxlen=SMOOTHING)
        self.slow_cursor_queue = deque(maxlen=25)
        self.calibrating = False
        self.calib_step = 0
        self.x_min = 0.42; self.x_max = 0.58
        self.y_min = 0.42; self.y_max = 0.58
        self.calib_buffer_x = []; self.calib_buffer_y = []
        self.blink_active = False
        self.blink_start = 0
        self.last_blink_end = 0
        self.blink_count = 0
        self.mouth_open_start = 0
        self.mouth_confirm_count = 0
        self.last_google_time = 0

state = ControlState()

# --- CORE DETECTION ---
def get_aspect_ratio(landmarks, points):
    coords = np.array([[landmarks[i].x, landmarks[i].y] for i in points])
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    return (A + B) / (2.0 * C) if C > 0 else 1.0

def get_mouth_open_ratio(landmarks):
    top_lip = landmarks[13]; bottom_lip = landmarks[14]
    left_corner = landmarks[61]; right_corner = landmarks[291]
    width = abs(left_corner.x - right_corner.x)
    height = abs(top_lip.y - bottom_lip.y)
    inner_top = landmarks[78]; inner_bottom = landmarks[308]
    inner_height = abs(inner_top.y - inner_bottom.y)
    primary_ratio = height / width if width > 0 else 0
    inner_ratio = inner_height / width if width > 0 else 0
    return max(primary_ratio, inner_ratio * 1.2)

# --- FIXED CURSOR (NO CORNER STICKING) ---
def get_iris_center(landmarks):
    return landmarks[473].x, landmarks[473].y

def calculate_head_tilt(landmarks):
    nose_tip = landmarks[1]; chin = landmarks[152]
    angle = np.degrees(np.arctan2(nose_tip.x - chin.x, nose_tip.y - chin.y))
    return angle * HEAD_TILT_SENSITIVITY

def map_coordinates(x, y):
    # FIXED: Anti-corner sticking with padding
    padding = 0.05
    norm_x = np.clip((x - state.x_min) / (state.x_max - state.x_min), padding, 1-padding)
    norm_y = np.clip((y - state.y_min) / (state.y_max - state.y_min), padding, 1-padding)
    return (norm_x * screen_w, norm_y * screen_h)

def move_cursor_smooth(tx, ty):
    global cursor_frozen, frozen_x, frozen_y
    
    if cursor_frozen:
        return frozen_x, frozen_y
    
    state.cursor_queue.append((tx, ty))
    state.slow_cursor_queue.append((tx, ty))
    
    if len(state.cursor_queue) == SMOOTHING and len(state.slow_cursor_queue) == 25:
        avg_x = np.mean([p[0] for p in state.cursor_queue]) * 0.3 + \
                np.mean([p[0] for p in state.slow_cursor_queue]) * 0.7
        avg_y = np.mean([p[1] for p in state.cursor_queue]) * 0.3 + \
                np.mean([p[1] for p in state.slow_cursor_queue]) * 0.7
        
        # ANTI-CORNER: Deadzone near edges
        margin = 50
        if (margin < avg_x < screen_w - margin and 
            margin < avg_y < screen_h - margin):
            
            current_x, current_y = pyautogui.position()
            distance = np.sqrt((avg_x - current_x)**2 + (avg_y - current_y)**2)
            
            if distance > 3:  # Smaller threshold
                duration = min(MOUSE_DURATION, distance * CURSOR_SPEED)
                pyautogui.moveTo(avg_x, avg_y, duration=duration)
        
        return avg_x, avg_y
    return tx, ty

def safe_double_click():
    global cursor_frozen, frozen_x, frozen_y
    print("🎯 DOUBLE BLINK → FREEZE → DOUBLE CLICK")
    
    frozen_x, frozen_y = pyautogui.position()
    cursor_frozen = True
    
    pyautogui.click(frozen_x, frozen_y, duration=0.01)
    time.sleep(0.08)
    pyautogui.click(frozen_x, frozen_y, duration=0.01)
    
    time.sleep(0.15)
    cursor_frozen = False

def open_google_once():
    current_time = time.time()
    if current_time - state.last_google_time < GOOGLE_COOLDOWN:
        return False
    
    def run():
        print("🌐 Single Google tab opened")
        state.last_google_time = time.time()
        try:
            subprocess.Popen(['chrome', '--new-tab', 'https://www.google.com'], 
                           startupinfo=subprocess.STARTUPINFO())
        except:
            webbrowser.open_new_tab('https://www.google.com')
    
    threading.Thread(target=run, daemon=True).start()
    return True

# --- CLEAN MINIMAL UI ---
def draw_ui(frame, left_ear, right_ear, mouth_ratio, tilt, cursor_x, cursor_y, h, w):
    # Minimal header (NO version title)
    cv2.rectangle(frame, (0, 0), (w, 25), (30, 30, 40), -1)
    
    # Status only
    freeze_status = "FROZEN" if cursor_frozen else "ACTIVE"
    status_color = (0, 165, 255) if cursor_frozen else (0, 255, 0)
    cv2.putText(frame, freeze_status, (10, 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    if state.calibrating:
        step = "TL" if state.calib_step == 1 else "BR"
        cv2.putText(frame, f"CAL:{step}", (120, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Compact metrics (right side)
    y_pos = h - 95
    cv2.putText(frame, f"L:{left_ear:.2f}", (w-150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0) if left_ear > BLINK_THRESHOLD else (0, 0, 255), 1)
    cv2.putText(frame, f"R:{right_ear:.2f}", (w-150, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0) if right_ear > BLINK_THRESHOLD else (0, 0, 255), 1)
    
    m_color = (0, 255, 0) if mouth_ratio > MOUTH_OPEN_THRESHOLD else (150, 150, 150)
    cv2.putText(frame, f"M:{mouth_ratio:.2f}", (w-150, y_pos+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_color, 1)
    
    # Google timer
    if state.mouth_open_start > 0 and state.mouth_confirm_count > 0:
        elapsed = time.time() - state.mouth_open_start
        remain = max(0, MOUTH_GOOGLE_DURATION - elapsed)
        cv2.putText(frame, f"G:{remain:.1f}", (w-150, y_pos+60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        prog = min(elapsed / MOUTH_GOOGLE_DURATION, 1.0)
        cv2.rectangle(frame, (w-160, y_pos+75), (w-160 + int(140*prog), y_pos+78), (255, 255, 0), -1)

# --- MAIN LOOP ---
print("="*60)
print("👁️👁️ Double blink = FREEZE + double-click")
print("😮 Mouth 3.5s = Single Google tab") 
print("↔️ Head tilt = Ultra-slow cursor")
print("C=Calibrate | Q=Quit")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    h, w = frame.shape[:2]
    current_time = time.time()
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # FIXED CURSOR MOVEMENT
        tilt = calculate_head_tilt(landmarks)
        iris_x, iris_y = get_iris_center(landmarks)
        
        if state.calibrating:
            state.calib_buffer_x.append(iris_x)
            state.calib_buffer_y.append(iris_y)
        else:
            target_x, target_y = map_coordinates(iris_x + tilt, iris_y)
            cursor_x, cursor_y = move_cursor_smooth(target_x, target_y)
        
        # Double blink
        left_ear = get_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144])
        right_ear = get_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380])
        
        if min(left_ear, right_ear) < BLINK_THRESHOLD:
            if not state.blink_active:
                state.blink_active = True
                state.blink_start = current_time
        else:
            if state.blink_active:
                duration = current_time - state.blink_start
                if duration < BLINK_DURATION_MAX:
                    if state.blink_count == 1 and (current_time - state.last_blink_end) < DOUBLE_BLINK_WINDOW:
                        safe_double_click()
                        state.blink_count = 0
                    else:
                        state.blink_count = 1
                        state.last_blink_end = current_time
                state.blink_active = False
        
        # Mouth detection
        mouth_ratio = get_mouth_open_ratio(landmarks)
        if mouth_ratio > MOUTH_OPEN_THRESHOLD:
            state.mouth_confirm_count += 1
            if state.mouth_open_start == 0 and state.mouth_confirm_count >= MOUTH_CONFIRM_FRAMES:
                state.mouth_open_start = current_time
        else:
            state.mouth_open_start = 0
            state.mouth_confirm_count = 0
        
        if (state.mouth_open_start > 0 and 
            (current_time - state.mouth_open_start) >= MOUTH_GOOGLE_DURATION and
            current_time - state.last_google_time >= GOOGLE_COOLDOWN):
            open_google_once()
        
        draw_ui(frame, left_ear, right_ear, mouth_ratio, tilt, cursor_x, cursor_y, h, w)
    
    cv2.imshow("GazeLink", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        state.calibrating = True
        state.calib_step = 1
        state.calib_buffer_x, state.calib_buffer_y = [], []
        print("Calibrate: TOP-LEFT → SPACE")
    elif key == 32:
        if state.calibrating and state.calib_buffer_x:
            if state.calib_step == 1:
                state.x_min, state.y_min = np.mean(state.calib_buffer_x), np.mean(state.calib_buffer_y)
                state.calib_step = 2
                print("BOTTOM-RIGHT → SPACE")
            else:
                state.x_max, state.y_max = np.mean(state.calib_buffer_x), np.mean(state.calib_buffer_y)
                state.calibrating = False
                print("Calibration complete!")

cap.release()
cv2.destroyAllWindows()
print("GazeLink stopped")
