import numpy as np
import time
from pynput import mouse, keyboard
import tensorflow as tf
from collections import deque
import math

model = tf.keras.models.load_model("mouse_behavior_model.h5")
window = deque(maxlen=10)
features = []

stop_flag = False

def compute_features(prev, curr):
    # Time difference
    t_diff = curr[0] - prev[0]

    # Coords difference
    x1, y1 = prev[1], prev[2]
    x2, y2 = curr[1], curr[2]
    distance = math.hypot(x2 - x1, y2 - y1)

    speed = distance / t_diff if t_diff > 0 else 0

    is_click = 1 if curr[3] == "click" else 0

    return [t_diff, distance, speed, is_click]

def on_move(x, y):
    now = time.time() * 1000
    window.append((now, x, y, "move"))
    check_prediction()

def on_click(x, y, button, pressed):
    if not pressed:
        return
    now = time.time() * 1000
    window.append((now, x, y, "click"))
    check_prediction()

def on_key_press(key):
    global stop_flag
    if key == keyboard.Key.esc:
        print("\n检测到 ESC 键，准备停止监听...")
        stop_flag = True
        mouse_listener.stop()
        return False

def check_prediction():
    if len(window) >= 2:
        last = list(window)[-2:]
        feat = compute_features(last[0], last[1])
        features.append(feat)

        if len(features) == 10:
            X = np.array(features)
            avg_feat = np.mean(X, axis=0).reshape(1, -1)

            prediction = model.predict(avg_feat, verbose=0)[0]
            human_prob = prediction[0]
            attack_prob = prediction[1]

            if attack_prob > 0.9:
                print(f"\n检测到可能的异常鼠标行为！攻击概率：{attack_prob:.2f}")
            else:
                print(f"\n正常鼠标行为，攻击概率：{attack_prob:.2f}")

            features.clear()


mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_key_press)

print("开始实时检测，按 ESC 退出")
mouse_listener.start()
keyboard_listener.start()

keyboard_listener.join()
mouse_listener.join()
