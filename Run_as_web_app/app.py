from flask import Flask, render_template, Response, jsonify
import mediapipe as mp
import cv2
from collections import Counter
import random
import time

app = Flask(__name__)

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Global variables
camera = cv2.VideoCapture(0)
game_result = {"result": "", "player_move": "", "computer_move": ""}
game_active = False
h = 480
w = 640
tip = [8, 12, 16, 20]
mid = [6, 10, 14, 18]

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0]:
        return "rock"
    elif fingers == [1, 1, 0, 0]:
        return "scissors"
    elif fingers == [1, 1, 1, 1]:
        return "paper"
    return "unknown"

def get_game_result(player_move, computer_move):
    if player_move == computer_move:
        return "TIE"
    elif (player_move == "rock" and computer_move == "scissors") or \
         (player_move == "scissors" and computer_move == "paper") or \
         (player_move == "paper" and computer_move == "rock"):
        return "WIN!"
    else:
        return "LOSS"

def generate_frames():
    global game_active, game_result
    
    with mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    ) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame1 = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if game_active and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame1,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get finger positions
                    fingers = []
                    hand_points = []
                    
                    for id, lm in enumerate(hand_landmarks.landmark):
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        hand_points.append([id, x, y])
                    
                    # Check finger positions
                    if hand_points:
                        for id in range(4):
                            if tip[id] == 8 and mid[id] == 6:
                                if hand_points[tip[id]][2] < hand_points[mid[id]][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                            elif tip[id] == 12 and mid[id] == 10:
                                if hand_points[tip[id]][2] < hand_points[mid[id]][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                            elif tip[id] == 16 and mid[id] == 14:
                                if hand_points[tip[id]][2] < hand_points[mid[id]][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                            elif tip[id] == 20 and mid[id] == 18:
                                if hand_points[tip[id]][2] < hand_points[mid[id]][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                        
                        # Detect gesture and update game result
                        player_move = detect_gesture(fingers)
                        if player_move != "unknown":
                            game_result["player_move"] = player_move
                            game_result["result"] = get_game_result(
                                player_move, 
                                game_result["computer_move"]
                            )
            
            # Add text overlay to show current state
            cv2.putText(frame1, f"Game Active: {game_active}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if game_active:
                cv2.putText(frame1, f"Player Move: {game_result['player_move']}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_game')
def start_game():
    global game_active, game_result
    game_active = True
    game_result = {
        "result": "waiting",
        "player_move": "",
        "computer_move": random.choice(['rock', 'paper', 'scissors'])
    }
    return jsonify({"status": "game_started"})

@app.route('/get_result')
def get_result():
    global game_result, game_active
    game_active = False
    return jsonify(game_result)

if __name__ == '__main__':
    app.run(debug=True)