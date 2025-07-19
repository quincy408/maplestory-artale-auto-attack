import pyautogui
import cv2
import time
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from ultralytics import YOLO
import threading
import random
import keyboard

class GameState:
    CURRENT = ""
    STANDBY = "standBy"
    USER_IS_USING = "userIsUsing"
    LEFT_ATTK = "leftAttack"
    RIGHT_ATTK = "rightAttack"
    NO_WINDOW = "noWindow"

    @classmethod
    def current_State(cls):
        return cls.CURRENT
    
    @classmethod
    def standby(cls):
        cls.CURRENT = cls.STANDBY

    @classmethod
    def user_is_using(cls):
        cls.CURRENT = cls.USER_IS_USING
    
    @classmethod
    def left_attack(cls):
        cls.CURRENT = cls.LEFT_ATTK

    @classmethod
    def right_attack(cls):
        cls.CURRENT = cls.RIGHT_ATTK
    
    @classmethod
    def no_window(cls):
        cls.CURRENT = cls.NO_WINDOW

    
class ObjectDetection:
    user_model = None
    monster_model = None
    last_player_pos = None
    left_monster_count = 0
    right_monster_count = 0

    @classmethod
    def load_model(cls, user_model_name=None, monster_model_name=None):
        user_model_path = f'./yolo/model/{user_model_name}/best.pt'
        monster_model_path = f'./yolo/model/{monster_model_name}/best.pt'
        cls.user_model = YOLO(user_model_path)
        cls.monster_model = YOLO(monster_model_path)

    @staticmethod
    def active_ms_window():
        try:
            maple_window = pyautogui.getWindowsWithTitle("MapleStory Worlds-Artale (繁體中文版)")[0]
            print("切換視窗成功")
            maple_window.activate()
            GameState.standby()
            time.sleep(0.05)
        except:
            print("找尋視窗失敗")
            GameState.no_window
    
    @staticmethod
    def windows_screenshot():
        # 擷取 MapleStory 視窗區域畫面
        maple_window = pyautogui.getWindowsWithTitle("MapleStory Worlds-Artale (繁體中文版)")[0]
        x, y, w, h = maple_window.left, maple_window.top, maple_window.width, maple_window.height
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        screenshot.save("screenshot.png")

    @classmethod
    def detection_result(cls, display_detection_result=True):
        img = cv2.imread('screenshot.png')
        player_results = cls.user_model(img, verbose=False)[0]
        monster_results = cls.monster_model(img, verbose=False)[0]

        if len(player_results.boxes) == 0:
            x1, y1, x2, y2 = cls.last_player_pos

        # 繪製 player 的檢測框（綠色）
        # 如果有檢測到 Player，更新最後位置
        if len(player_results.boxes) > 0:
            box = player_results.boxes[0]
            cls.last_player_pos = tuple(map(int, box.xyxy[0]))  # 更新為 (x1, y1, x2, y2)
            x1, y1, x2, y2 = cls.last_player_pos
            player_detected = True
        # 如果沒檢測到，但之前有記錄，使用最後位置
        elif cls.last_player_pos is not None:
            x1, y1, x2, y2 = cls.last_player_pos
            player_detected = False
        
        # 計算攻擊範圍
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        attack_x1_left, attack_x2_left = max(0, cx-20), min(img.shape[1], cx + 420)
        attack_x1_right, attack_x2_right = max(0, cx-420), min(img.shape[1], cx+20)
        attack_y = max(0, cy - 50)  # 攻擊範圍的頂邊 Y 座標

        # 初始化計數器
        cls.left_monster_count = 0
        cls.right_monster_count = 0

        # 計算左右兩側的怪物數量
        for box in monster_results.boxes:
            if box.conf > 0.4:
                mx1, my1, mx2, my2 = map(int, box.xyxy[0])
                if (mx1 < attack_x2_left and mx2 > attack_x1_left and 
                    my2 >= attack_y and my1 <= attack_y):  # 怪物底部穿過或接觸攻擊線
                    cls.right_monster_count += 1

                if (mx1 < attack_x2_right and mx2 > attack_x1_right and 
                    my2 >= attack_y and my1 <= attack_y):  # 同上
                    cls.left_monster_count += 1

        if display_detection_result == True:
            current_state = f'State: {GameState.current_State()}'
            text_size = cv2.getTextSize(current_state, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = img.shape[1] - text_size[0] - 10  # 右邊留10像素邊距
            # 計算文字位置（右上角 + 邊距）
            text_x = img.shape[1] - text_size[0] - 10
            text_y = 50
            text_width = text_size[0]
            text_height = text_size[1]
            cv2.rectangle(img,
              (text_x - 5, text_y - text_height - 5),  # 左上角
              (text_x + text_width + 5, text_y + 5),   # 右下角
              (255, 255, 255),  # 白色
              -1)  # 填滿
            cv2.putText(img, current_state, (text_x, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # 繪製 Player 框（只有當下檢測到時才繪製）
            if player_detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.putText(img, f'Player: {float(box.conf):.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # 可選：用不同顏色標記這是記憶的位置
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)  # 暗綠色邊框

            # 繪製頂邊（藍色，線寬 2px）
            cv2.line(img, (attack_x1_left, attack_y+5), (attack_x2_left, attack_y+5), (255, 0, 0), 2)
            cv2.line(img, (attack_x1_right, attack_y-5), (attack_x2_right, attack_y-5), (0, 0, 255), 2)

            # 在右側攻擊線上方顯示數量
            cv2.putText(img, f'{cls.right_monster_count}', 
                        ((attack_x2_left) - 10, attack_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 在左側攻擊線上方顯示數量
            cv2.putText(img, f'{cls.left_monster_count}', 
                        ((attack_x1_right), attack_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
            # 繪製 monster 的檢測框（紅色）
            for box in monster_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 獲取框的座標
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 紅色框
                cv2.putText(img, f'Monster: {float(box.conf):.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 顯示置信度
            
            # 顯示圖片
            cv2.imshow('---', img)
            cv2.waitKey(1) 

    @classmethod
    def get_left_monster_count(cls):
        return cls.left_monster_count
    
    @classmethod
    def get_right_monster_count(cls):
        return cls.right_monster_count
    
class GameScript:
    keyboard = KeyboardController()
    last_direction = None

    @classmethod
    def reset_direction(cls):
        cls.last_direction = None

    @classmethod
    def press_left_release(cls, wait_time=0.1):
        cls.keyboard.press(Key.left)
        time.sleep(wait_time)
        cls.keyboard.release(Key.left)
        cls.last_direction = True
        time.sleep(0.1)

    @classmethod
    def press_right_release(cls, wait_time=0.1):
        cls.keyboard.press(Key.right)
        time.sleep(wait_time)
        cls.keyboard.release(Key.right)
        cls.last_direction = False
        time.sleep(0.1)

    @classmethod
    def press_up_release(cls, wait_time=0.1):
        cls.keyboard.press(Key.up)
        time.sleep(wait_time)
        cls.keyboard.release(Key.up)

    @classmethod
    def press_down_release(cls, wait_time=0.1):
        cls.keyboard.press(Key.down)
        time.sleep(wait_time)
        cls.keyboard.release(Key.down)

    @classmethod
    def press_space_release(cls, wait_time=0.1):
        cls.keyboard.press(Key.space)
        time.sleep(wait_time)
        cls.keyboard.release(Key.space)

    @classmethod
    def press_a_release(cls, wait_time=0.05):
        cls.keyboard.press('a')
        time.sleep(wait_time)
        cls.keyboard.release('a')

    @classmethod
    def press_s_release(cls, wait_time=0.05):
        cls.keyboard.press('s')
        time.sleep(wait_time)
        cls.keyboard.release('s')

    @classmethod
    def press_d_release(cls, wait_time=0.05):
        cls.keyboard.press('d')
        time.sleep(wait_time)
        cls.keyboard.release('d')

class MainScript:
    def __init__(self):
        ObjectDetection.active_ms_window()
        self.display_detection_result = True
        # 啟動顯示視窗執行緒，只執行一次
        threading.Thread(target=self.detection_thread, daemon=True).start()
        threading.Thread(target=self.action_thread, daemon=True).start()

 
    def detection_thread(self):
        """處理圖像辨識和顯示的線程"""
        while True:
            ObjectDetection.windows_screenshot()
            ObjectDetection.detection_result(display_detection_result=self.display_detection_result)


    def action_thread(self):
        """處理遊戲動作的線程"""
        while True:
            if not any(keyboard.is_pressed(k) for k in ['up', 'down', 'left', 'right']):
                tmp_left_monster_count = ObjectDetection.get_left_monster_count()
                tmp_right_monster_count = ObjectDetection.get_right_monster_count()
                # print(f"發現左邊{tmp_left_monster_count}個 右邊{tmp_right_monster_count}個 ")
                # 判斷是否兩邊數量相等
                if tmp_left_monster_count == 0 and tmp_right_monster_count == 0:
                        GameState.standby()
                        time.sleep(0.2)
                        continue
                if (tmp_left_monster_count == tmp_right_monster_count):
                    # 隨機選擇左邊或右邊 (50% 機率)
                    attack_left = random.choice([True, False])
                    if GameState.current_State() == GameState.LEFT_ATTK:
                        attack_left = True
                    if attack_left:
                        GameState.left_attack()
                        # 處理左邊
                        jump = 0
                        if GameScript.last_direction != True:
                            GameScript.press_left_release(wait_time=0.04)
                        if tmp_left_monster_count == 1:
                            # 只剩1隻時，50%機率用S/D鍵攻擊
                            while ObjectDetection.get_left_monster_count() == 1:
                                jump = jump + 1
                                if jump > 3:
                                    break
                                if random.random() < 0.8:
                                    GameScript.press_s_release()
                                else:
                                    GameScript.press_d_release()
                        else:
                            # 正常A鍵攻擊
                            while ObjectDetection.get_left_monster_count() > 1:
                                if random.random() < 0.1:
                                    GameScript.press_left_release(wait_time=random.uniform(0.1, 0.2))
                                GameScript.press_a_release()
                    else:
                        GameState.right_attack()
                        # 處理右邊
                        jump = 0
                        if GameScript.last_direction != False:
                            GameScript.press_right_release(wait_time=0.04)
                        if tmp_right_monster_count == 1:
                            # 只剩1隻時，50%機率用S/D鍵攻擊
                            while ObjectDetection.get_right_monster_count() == 1:
                                jump = jump + 1
                                if jump > 3:
                                    break
                                if random.random() < 0.8:
                                    GameScript.press_s_release()
                                else:
                                    GameScript.press_d_release()
                        else:
                            # 正常A鍵攻擊
                            while ObjectDetection.get_right_monster_count() > 1:
                                if random.random() < 0.1:
                                    GameScript.press_right_release(wait_time=random.uniform(0.1, 0.2))
                                GameScript.press_a_release()    
                elif (tmp_left_monster_count > tmp_right_monster_count):
                    # 原本的左邊較多邏輯
                    # print(f"{c}扁左邊:{ObjectDetection.get_left_monster_count()}")
                    GameState.left_attack()
                    jump = 0
                    if GameScript.last_direction != True:
                        GameScript.press_left_release(wait_time=0.04)
                    if tmp_left_monster_count == 1:
                        # 只剩1隻時的特殊處理
                        while ObjectDetection.get_left_monster_count() == 1:
                            jump = jump + 1
                            if jump > 3:
                                break
                            if random.random() < 0.8:
                                GameScript.press_s_release()
                            else:
                                GameScript.press_d_release()
                    else:
                        # 正常A鍵攻擊
                        while ObjectDetection.get_left_monster_count() > 1:
                            if random.random() < 0.1:
                                GameScript.press_left_release(wait_time=random.uniform(0.1, 0.2))
                            GameScript.press_a_release()
                elif (tmp_right_monster_count > tmp_left_monster_count):
                    # 原本的右邊較多邏輯
                    # print(f"{c}扁右邊:{ObjectDetection.get_right_monster_count()}")
                    GameState.right_attack()
                    jump = 0
                    if GameScript.last_direction != False:
                        GameScript.press_right_release(wait_time=0.04)
                    if tmp_right_monster_count == 1:
                        # 只剩1隻時的特殊處理
                        while ObjectDetection.get_right_monster_count() == 1:
                            jump = jump + 1
                            if jump > 3:
                                break
                            if random.random() < 0.8:
                                 GameScript.press_s_release()
                            else:
                                GameScript.press_d_release()
                    else:
                        # 正常A鍵攻擊
                        while ObjectDetection.get_right_monster_count() > 1:
                            if random.random() < 0.1:
                                GameScript.press_right_release(wait_time=random.uniform(0.1, 0.2))
                            GameScript.press_a_release()
                time.sleep(0.7)
            else:
                GameState.user_is_using()
                GameScript.reset_direction()
                time.sleep(0.25)
 

    def main(self, display_detection_result):
        self.display_detection_result = display_detection_result
        while True:
            time.sleep(1)


        


