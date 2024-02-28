import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

WRIST = 0
THUMB = 4
POINTER = 8
MIDDLE = 12
RING = 16
PINKY = 20



cap = cv2.VideoCapture(0)  # 0 for the default camera

def dist(a,b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def dist2(xs,ys,i,j):
    return np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)

def is_dr_strange(X):
    if dist(X[THUMB],X[WRIST]) > 50:
        return True
    return False

def is_pointer_clicked(xs,ys):
        if(len(xs) > 20 and len(ys) > 20):
            d = np.sqrt((xs[4]-xs[8])**2 + (ys[4]-ys[8])**2)
            if(d > 25):
                print("Open")
                return False
            else:
                print("Closed")
                return True

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find hand landmarks
    results = hands.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    xs = []
    ys = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # hand = hand_landmarks.hand
            # print("dir hand ", dir(hand_landmarks))
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # x = landmark.x
                # y = landmark.y
                
                xs.append(x)
                ys.append(y)

                # Do something with the landmark coordinates
                # For example, draw a circle on the landmark

                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                
    cv2.imshow('MediaPipe Hands', image)
    if(len(xs) > 20 and len(ys) > 20):
        norm = dist2(xs,ys,THUMB,WRIST)
        print("Thumb to wrist: ", dist2(xs,ys,THUMB,WRIST)/norm)
        print("Pointer to middle: ", dist2(xs,ys,POINTER,MIDDLE)/norm)
        print("Middle to ring: ", dist2(xs,ys,MIDDLE,RING)/norm)
        print("Ring to Pinky: ", dist2(xs,ys,RING,PINKY)/norm)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         continue

#     # Convert the BGR image to RGB and process it
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Convert back to BGR for displaying
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     if results.multi_hand_landmarks:
#         for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

#             # Choose a color based on the hand index
#             color = (0, 255, 0) if hand_idx == 0 else (0, 0, 255)

#             # Draw landmarks and connections
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=color, thickness=2))

#             # Optional: Add text to identify the hand
#             cv2.putText(image, f'Hand {hand_idx + 1}', 
#                         (10, 30 * (hand_idx + 1)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#     # Display the image
#     cv2.imshow('MediaPipe Hands', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

