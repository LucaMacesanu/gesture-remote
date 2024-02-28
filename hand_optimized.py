import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time



WRIST = 0
THUMB = 1
POINTER = 2
MIDDLE = 3
RING = 4
PINKY = 5


ORIGINAL_INDEXES = [0,4,8,12,16,20]

def map_range(x, a, b, c, d):
    return c + (x - a) * (d - c) / (b - a)


def dist(xs,ys,i,j):
    return np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)

def is_dr_strange(xs,ys,norm):
    if dist(xs,ys,POINTER,MIDDLE)/norm < 0.2:
        if dist(xs,ys,MIDDLE, RING)/norm > 0.88:
            if dist(xs,ys,RING, PINKY)/norm < 0.2:
                return True
    return False

def interpolate(x_last, x_target, alpha):
    return x_target - (x_target - x_last) * (1 - alpha)


def is_touching(xs,ys,i,j,norm):
    d = dist(xs,ys,i,j)/norm
    if d < 0.2:
        return True
    else:
        return False


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    y_last = 0
    x_last = 0
    reference_last = [0,0]
    cursor_last = [0,0]
    l_norm_last = 0
    flag = True
    results = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
        image = cv2.pyrDown(image)

        # Process the image to find hand landmarks
        if flag: results = hands.process(image)
        flag = not flag

        # removed to make it faster
        # Convert back to BGR for OpenCV
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_handedness:
            hand_ness = []
            for idx,handedness in enumerate(results.multi_handedness):
                hand_ness.append(handedness.classification[0].label)
            
        #to seperate between left and right hand
        R = [[],[]]
        L = [[],[]]
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine which list to use based on handedness
                xs, ys = (R[0], R[1]) if hand_ness[hand_idx] == "Right" else (L[0], L[1])
                
                # Directly access the landmarks you need
                for idx in ORIGINAL_INDEXES:
                    landmark = hand_landmarks.landmark[idx]
                    # x = int(landmark.x * image.shape[1])
                    # y = int(landmark.y * image.shape[0])
                    x = (landmark.x * image.shape[1])
                    y = (landmark.y * image.shape[0])

                    xs.append(x)
                    ys.append(y)

                    # Draw a circle on the landmark
                    # if(hand_ness[hand_idx] == "Right"):
                    #     cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                    # if(hand_ness[hand_idx] == "Left"):
                    #     cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                    
        # cv2.imshow('MediaPipe Hands', image)
        #If both hands detected its time for the magic
        if(len(L[0]) == 6 and len(R[0]) == 6):
            l_norm = dist(L[0],L[1],THUMB,WRIST)
            if(is_dr_strange(L[0],L[1],l_norm)):
                # smooth the l_norm
                l_norm = interpolate(l_norm_last,l_norm,0.2)
                l_norm_last = l_norm

                reference = [L[0][WRIST],L[1][WRIST]]
                cursor = [R[0][THUMB] - reference[0],R[1][THUMB] - reference[1]]


                # smooth the reference and cursor
                reference[0] = interpolate(reference_last[0],reference[0],0.01)
                reference[1] = interpolate(reference_last[1],reference[1],0.01)
                reference_last = reference
                
                cursor[0] = interpolate(cursor_last[0],cursor[0],0.1)
                cursor[1] = interpolate(cursor_last[1],cursor[1],0.1)
                cursor_last = cursor
                

                # Scale to fit screen
                x = map_range(cursor[0]/l_norm,1,6,0,pyautogui.size().width)
                y = map_range(cursor[1]/l_norm,0,3,0,pyautogui.size().height)

                #adjust offsets for more comfortable positioning
                y += pyautogui.size().height/2 

                # # add a dead space of 15 pix
                # if(np.abs(x-x_last) <= 15): x = x_last
                # if(np.abs(y-y_last) <= 15): y = y_last

                # apply smoothing
                x = interpolate(x_last,x,0.8)
                y = interpolate(y_last,y,0.8)

                # constrain to screen size
                x = np.clip(x,0,pyautogui.size().width - 1)
                y = np.clip(y,0,pyautogui.size().height - 1)

                # Move cursor to location
                pyautogui.moveTo(x,y)
                x_last = x
                y_last = y

                if(not flag):
                    if(is_touching(R[0],R[1],POINTER,THUMB,l_norm)):    
                        print("Left click at: ",cursor)
                        pyautogui.click(button='left')
                        time.sleep(0.5)
                    if(is_touching(R[0],R[1],MIDDLE,THUMB,l_norm)):    
                        print("Right click at: ",cursor)
                        pyautogui.click(button='right')
                        time.sleep(0.5)
                    if(is_touching(R[0],R[1],RING,THUMB,l_norm)):
                        print("Double click at: ",cursor)
                        pyautogui.click(button='left')
                        pyautogui.click(button='left')
                        time.sleep(0.5)

        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
    cap.release()
    cv2.destroyAllWindows()


