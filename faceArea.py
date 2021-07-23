import cv2
import numpy as np

# imgPathの画像をグレースケールにしてndarrayで返す(h, w)
def getGrayImage(imgPath: str) -> np.ndarray:
    img = cv2.imread(imgPath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img/img_gray.png', img_gray)
    return img_gray

# 分類機のパス
cascade_path = "./model/haarcascade_frontalface_default.xml"

# 顔領域をtupleで返す(x, y, w, h)
def faceArea(img: np.ndarray) -> tuple:
    tmpImg = img
    tmp = None
    # 分類機の特徴量を取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔領域を取得
    faceareas = cascade.detectMultiScale(tmpImg)
    # 取得した顔領域からx, y, w, hを取って黒、太さ2の四角形を書き込む
    for x, y, w, h in faceareas:
        cv2.rectangle(tmpImg, (x, y), (x + w, y + h), 0, thickness=2)
        tmp = (x, y, w, h)
    cv2.imwrite('img/output.png', tmpImg)
    if(tmp != None):
        return tmp
    else:
        return (0, 0, 0, 0)

""" これはテストコード
def main():
    path = './img/tsuru512006_TP_V4.jpg'
    t = faceArea(getGrayImage(path))
    print(t)

if __name__ == '__main__':
    main()
"""