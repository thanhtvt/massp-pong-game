import cv2
import cvzone
import numpy as np
import random
from cvzone.HandTrackingModule import HandDetector


def load_images():
    # Importing all images
    img_background = cv2.imread("statics/Background.png")
    img_gameover = cv2.imread("statics/gameOver.png")
    img_ball = cv2.imread("statics/Ball.png", cv2.IMREAD_UNCHANGED)
    img_bat1 = cv2.imread("statics/bat2.png", cv2.IMREAD_UNCHANGED)
    img_bat2 = cv2.imread("statics/bat2.png", cv2.IMREAD_UNCHANGED)
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


def draw_countdown(img, seconds, losed_time):
    cv2.putText(img,
                text=str(3 - seconds + losed_time),
                org=(600, 650),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0, 144, 255),
                thickness=5)


def show_game_over(img):
    cv2.putText(img,
                text=str("GAME OVER"),
                org=(360, 120),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0, 144, 255),
                thickness=5)
    return img


def move_ball(ballPos, speedX, speedY, score, seconds, losed, losed_time):
    if losed == 1:
        if seconds - losed_time > 3:
            losed = 0
        else:
            return ballPos, speedY, score, losed

    # Move the Ball
    if ballPos[1] >= 500 or ballPos[1] <= 10:
        speedY = -speedY

    ballPos[0] += speedX
    ballPos[1] += speedY

    return ballPos, speedY, score, losed


def detect_and_handle_hands(img, ballPos, speedX, speedY, score, img_bat1, img_bat2,
                            img_background, detector, multiplier, init_speedX):
    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)

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
                if (59 < ballPos[0] < 59 + w1 and \
                    y1 < ballPos[1] < y1 + h1) or \
                        (59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] + 50 < y1 + h1):
                    speedX = -round(speedX * multiplier)
                    if abs(speedX) >= 50:
                        speedX = 50
                    ballPos[0] += 50
                elif 59 > ballPos[0]:
                    
                    y_new = ballPos[1] +  speedY*(59+w1-ballPos[0])/speedX
                    if y1 < y_new < y1 + h1 or y1 <y_new + 50 < y1+h1:
                        speedX = -round(speedX * multiplier)
                        if abs(speedX) >= 50:
                            speedX = 50
                        ballPos[0] += 50

            if order == 1:
                img = cvzone.overlayPNG(img, img_bat2, (1195, y1))
                if (1195 < ballPos[0]+50 < 1195 + w1 and y1 < ballPos[1] < y1 + h1) or \
                (1195 < ballPos[0] + 50< 1195 + w1 and y1 < ballPos[1] + 50 < y1 + h1):
                    speedX = -round(speedX * multiplier)
                    if abs(speedX) >= 50:
                        speedX = -50
                    ballPos[0] -= 50
                elif ballPos[0] + 50 > 1195+w1:
                    y_new = ballPos[1]-speedY*(ballPos[0]+50-1195)/speedX
                    if y1 < y_new< y1 + h1 or y1 < y_new + 50< y1 + h1:
                        speedX = -round(speedX * multiplier)
                        if abs(speedX) >= 50:
                            speedX = -50
                        ballPos[0] -= 50
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
    ticks = 0
    FPS = 30
    losed = 1
    losed_time = 0
    hb = img_background.shape[0] // 2 + 50
    wb = img_background.shape[1] // 2 - 20
    ballPos = [wb, hb]
    speedX = 15
    speedY = -10
    initial_speedX = speedX
    initial_speedY = speedY
    multiplier = 1.1
    gameOver = False
    score = [0, 0]

    while True:
        ticks += 1
        seconds = ticks // FPS
        _, img = cap.read()
        img = cv2.flip(img, 1)
        imgRaw = img.copy()

        # Detect and handle hands
        img, ballPos, speedX, score = detect_and_handle_hands(
            img, ballPos, speedX, speedY, score, img_bat1, img_bat2, img_background,
            detector, multiplier, initial_speedX)

        # Check if the game is over
        if ballPos[0] < 40:
            score[1] += 1
            losed = 1
            losed_time = seconds
            ballPos = [wb, hb]

            # speedX = -speedX
            speedX = initial_speedX
            speedY = initial_speedY
        if ballPos[0] >= 1200:
            losed = 1
            losed_time = seconds
            score[0] += 1
            ballPos = [wb, hb]
            # speedX = -speedX
            speedX = -initial_speedX
            speedY = -initial_speedY
        if score[0] == 3 or score[1] == 3:
            gameOver = 1
            show_game_over(img)
        else:
            ballPos, speedY, score, losed = move_ball(ballPos, speedX, speedY,
                                                      score, seconds, losed,
                                                      losed_time)
            try:
                img = cvzone.overlayPNG(img, img_ball, ballPos)
            except:
                if ballPos[0] < 40:
                    score[1] += 1
                    losed = 1
                    losed_time = seconds
                    ballPos = [wb, hb]

                    # speedX = -speedX
                    speedX = initial_speedX
                    speedY = initial_speedY
                if ballPos[0] >= 1200:
                    losed = 1
                    losed_time = seconds
                    score[0] += 1
                    ballPos = [wb, hb]
                    # speedX = -speedX
                    speedX = -initial_speedX
                    speedY = -initial_speedY

            draw_scoreboard(img, score)

            if losed:
                draw_countdown(img, seconds, losed_time)

        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('r'):
            ballPos = [wb, hb]
            speedX = 15
            speedY = 15
            gameOver = False
            score = [0, 0]
            img_gameover = cv2.imread("statics/gameOver.png")


if __name__ == "__main__":
    main()
