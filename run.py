import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

def load_images():
    # Importing all images
    img_background = cv2.imread("statics/Background.png")
    img_gameover = cv2.imread("statics/gameOver.png")
    img_ball = cv2.imread("statics/Ball.png", cv2.IMREAD_UNCHANGED)
    img_bat1 = cv2.imread("statics/bat.png", cv2.IMREAD_UNCHANGED)
    img_bat2 = cv2.imread("statics/bat.png", cv2.IMREAD_UNCHANGED)
    return img_background, img_gameover, img_ball, img_bat1, img_bat2


def draw_scoreboard(img, score):
    cv2.putText(img,
                text=str(score[0]),
                org=(300, 650),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(255, 255, 255),
                thickness=5)
    cv2.putText(img,
                text=str(score[1]),
                org=(900, 650),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(255, 255, 255),
                thickness=5)

def draw_countdown(img,seconds,losed_time,wb,hb):
    cv2.putText(img,
                text=str(3-seconds+losed_time),
                org=(wb,hb+100),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0, 0, 0),
                thickness=5)

def show_game_over(img, score):
    cv2.putText(img,
                text=str(score[1] + score[0]).zfill(2),
                org=(585, 360),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=2.5,
                color=(200, 0, 200),
                thickness=5)
    return img


def move_ball(ballPos, speedX, speedY, score,seconds,losed,losed_time):
    if (losed==1):
        if (seconds-losed_time>3):
            losed=0
        else: return ballPos,speedY,score,losed
    # Move the Ball
    if ballPos[1] >= 500 or ballPos[1] <= 10:
        speedY = -speedY

    ballPos[0] += speedX
    ballPos[1] += speedY

    return ballPos, speedY, score,losed


def detect_and_handle_hands(img, ballPos, speedX, score, img_bat1, img_bat2,
                            img_background, detector, multiplier):
    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, img_background, 0.8, 0)

    # Overlay bat images on hands and check for ball collision
    if len(hands) == 2:
        if hands[0]['bbox'][0] > hands[1]['bbox'][0]:
            hands[0], hands[1] = hands[1], hands[0]
    order = 0
    if len(hands) == 1 and hands[0]['bbox'][0] >= img_background.shape[1] / 2:
        order = 1
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = img_bat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if order == 0:
                img = cvzone.overlayPNG(img, img_bat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -round(speedX * multiplier)
                    ballPos[0] += 30

            if order == 1:
                img = cvzone.overlayPNG(img, img_bat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -round(speedX * multiplier)
                    ballPos[0] -= 30
            order += 1

    return img, ballPos, speedX, score

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    img_background, img_gameover, img_ball, img_bat1, img_bat2 = load_images()

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Variables
    ticks=0
    FPS=30
    losed=1
    losed_time=0
    hb=img_background.shape[0]//2
    wb=img_background.shape[1]//2
    ballPos = [wb,hb]
    speedX = 15
    speedY = 15
    multiplier = 1.1
    gameOver = False
    score = [0, 0]

    while True:
        ticks+=1
        seconds=ticks // FPS
        _, img = cap.read()
        img = cv2.flip(img, 1)
        imgRaw = img.copy()

        # Detect and handle hands
        img, ballPos, speedX, score = detect_and_handle_hands(
            img, ballPos, speedX, score, img_bat1, img_bat2, img_background,
            detector, multiplier)

        # Check if the game is over
        if ballPos[0] <40:
            score[1]+=1
            losed=1
            losed_time=seconds
            ballPos = [wb,hb]

            speedX=-speedX
        if ballPos[0] >= 1200:
            losed=1
            losed_time=seconds
            score[0]+=1
            ballPos = [wb,hb]
            speedX=-speedX
        if (score[0]==3 or score[1]==3):
            gameOver=1
        if gameOver:
            img = show_game_over(img_gameover, score)
        else:
            ballPos, speedY, score, losed= move_ball(ballPos, speedX, speedY, score,seconds,losed,losed_time)
            img = cvzone.overlayPNG(img, img_ball, ballPos)

            draw_scoreboard(img, score)
            
            if (losed):
                draw_countdown(img,seconds,losed_time,wb,hb)

        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('r'):
            ballPos = [100, 100]
            speedX = 15
            speedY = 15
            gameOver = False
            score = [0, 0]
            img_gameover = cv2.imread("statics/gameOver.png")
    print(losed_time)


if __name__ == "__main__":
    main()
