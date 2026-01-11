**GazeLink - Professional Assistive Technology:**

GazeLink is a specialized Human-Computer Interaction (HCI) system engineered to bridge the digital divide for individuals living with Quadriplegia, ALS, and Spinal Cord Injuries.
For millions who have lost the use of their hands, standard computers are inaccessible, and traditional eye-trackers are often prohibitively expensive or too unstable for daily use. 
GazeLink solves this by transforming any standard laptop webcam into a high-precision input device. Unlike generic gaze trackers, GazeLink prioritizes stability over speed—utilizing 
custom "deadzone" algorithms and "freeze-click" logic to filter out involuntary jitters. This ensures that digital autonomy is not a luxury for the few, but an accessible right for 
everyone.



**Key Features:**

* **Precision Cursor Control:** Uses a custom "Grid + Deadzone" stabilization algorithm to eliminate micro-jitters, offering pixel-perfect control.
* **Smart Double-Clicking:** Features a unique "Freeze-Frame" click logic. The cursor locks in place during a double-blink to prevent accidental drags.
* **Voice Assistant Integration:** Innovative Mouth Gesture triggers Google Voice Search automatically without waking the whole system.
* **Head Tilt Scrolling:** Intuitive scrolling by simply tilting the head up or down.
* **Dynamic Calibration:** Includes a 2-point calibration system to adapt to different screen sizes and user seating positions.



 **Tech Stack:**

* **Language:** Python 3.12+
* **Computer Vision:** OpenCV (cv2) \& Google MediaPip
* **Automation:** PyAutoGUI
* **Math/Logic:** NumPy \& Deque (Collections)



  **Gesture**	          
  **1. Eye Movement:** Moves the Cursor (with smoothing)

  **2. Double Blink:** Double Click (freezes cursor for precision)

  **3. Mouth Open:** 	 Opens Google Voice Assistant

  **4.  Head Tilt:**    Scrolls the page





**Installation**

**1. Clone the Repository**

**2. Install Dependencies:**
   pip install opencv-python mediapipe pyautogui numpy

**3. Run the System:**
   python main.py





