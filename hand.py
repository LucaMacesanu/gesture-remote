import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

WRIST = 0
THUMB = 4
POINTER = 8
MIDDLE = 12
RING = 16
PINKY = 20



cap = cv2.VideoCapture(0)  # 0 for the default camera

def map_range(x, a, b, c, d):
    """
    Maps a value x from range [a, b] to range [c, d].

    :param x: The value to map.
    :param a: The lower bound of the original range.
    :param b: The upper bound of the original range.
    :param c: The lower bound of the new range.
    :param d: The upper bound of the new range.
    :return: The value of x mapped to the new range [c, d].
    """
    return c + (x - a) * (d - c) / (b - a)


def dist(xs,ys,i,j):
    return np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)

def is_dr_strange(xs,ys,norm):
    if dist(xs,ys,POINTER,MIDDLE)/norm < 0.2:
        if dist(xs,ys,MIDDLE, RING)/norm > 0.88:
            if dist(xs,ys,RING, PINKY)/norm < 0.2:
                return True
    return False

def inverse_exponential_interpolation(x_last, x_target, alpha):
    # return x_target - (x_target - x_last)**2 * (1-alpha)
    return x_target - (x_target - x_last) * (1 - alpha)

def smooth_transition(x_last, x_target, alpha):
    """
    Smoothly transition a value towards a target.

    :param x_last: The last known value (current value of x).
    :param x_target: The target value x is moving towards.
    :param alpha: Smoothing factor (between 0 and 1). 
                  Smaller values result in smoother transitions.
    :return: New value of x after applying smoothing.
    """
    return x_last + alpha * (x_target - x_last)

def is_pointer_clicked(xs,ys):
        if(len(xs) > 20 and len(ys) > 20):
            d = np.sqrt((xs[4]-xs[8])**2 + (ys[4]-ys[8])**2)
            if(d > 25):
                print("Open")
                return False
            else:
                print("Closed")
                return True
def is_touching(xs,ys,i,j,norm):
    d = dist(xs,ys,i,j)/norm
    if d < 0.2:
        return True
    else:
        return False

y_last = 0
x_last = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    image = cv2.pyrDown(image)

    # Process the image to find hand landmarks
    results = hands.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_handedness:
        hand_ness = []
        for idx,handedness in enumerate(results.multi_handedness):
            hand_ness.append(handedness.classification[0].label)
        
    #to seperate between left and right hand
    R = [[],[]]
    L = [[],[]]
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # swapped because openc swapped
            if(hand_ness[hand_idx] == "Right"):
                xs = R[0]
                ys = R[1]
            elif(hand_ness[hand_idx] == "Left"):
                xs = L[0]
                ys = L[1]
            else:
                print("Error determining handedness")
            
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                
                xs.append(x)
                ys.append(y)

                # Draw a circle on the landmark
                if(hand_ness[hand_idx] == "Right"):
                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                if(hand_ness[hand_idx] == "Left"):
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                
    cv2.imshow('MediaPipe Hands', image)
    #If both hands detected its time for the magic
    if(L and R):
        if(len(L[0]) > 20 and len(R[0]) > 20):
            l_norm = dist(L[0],L[1],THUMB,WRIST)
            if(is_dr_strange(L[0],L[1],l_norm)):
                reference = [L[0][WRIST],L[1][WRIST]]
                cursor = [R[0][THUMB] - reference[0],R[1][THUMB] - reference[1]]
                # Scale to fit screen
                x = map_range(cursor[0]/l_norm,2,5,0,pyautogui.size().width)
                y = map_range(cursor[1]/l_norm,0,3,0,pyautogui.size().height)

                #adjust offsets for more comfortable positioning
                y += pyautogui.size().height/2 

                # add a dead space of 30 pix
                if(np.abs(x-x_last) <= 30): x = x_last
                if(np.abs(y-y_last) <= 30): y = y_last

                # apply smoothing
                x = inverse_exponential_interpolation(x_last,x,0.3)
                y = inverse_exponential_interpolation(y_last,y,0.3)

                # constrain to screen size
                x = np.clip(x,0,pyautogui.size().width)
                y = np.clip(y,0,pyautogui.size().height)

                # Move cursor to location
                pyautogui.moveTo(x,y)
                x_last = x
                y_last = y


                if(is_touching(R[0],R[1],POINTER,THUMB,l_norm)):    
                    print("Left click at: ",cursor)
                    pyautogui.click(button='left')
                if(is_touching(R[0],R[1],MIDDLE,THUMB,l_norm)):    
                    print("Right click at: ",cursor)
                    pyautogui.click(button='right')




    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


