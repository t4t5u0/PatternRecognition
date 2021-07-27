from __future__ import annotations

import cv2
import numpy as np


class FaceArea:
    "顔面"

    def __init__(
            self,
            img_path: str,
            cascade_path="./model/haarcascade_frontalface_default.xml"
    ) -> None:
        self.cascade_path = cascade_path
        self.norm_img_path = './img/norm64.png'
        self.img_path = img_path
        self.gray_image: np.ndarray = None
        self.clipped_img: np.ndarray = None
        self.norm_img: np.ndarray = None
        self.face_box: tuple[int] = (0, 0, 0, 0)

    def getGrayImage(self) -> FaceArea:
        "imgPathの画像をグレースケールにしてndarrayで返す(h, w)"
        img = cv2.imread(self.img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('img/img_gray.png', img_gray)
        self.gray_image = img_gray
        return self

    def faceArea(self) -> FaceArea:
        "顔領域をtupleで返す(x, y, w, h)"
        tmpImg = self.gray_image
        result = None
        # 分類機の特徴量を取得
        cascade: cv2.Image = cv2.CascadeClassifier(cascade_path)
        # 顔領域を取得
        faceareas = cascade.detectMultiScale(tmpImg)
        # 取得した顔領域からx, y, w, hを取って黒、太さ2の四角形を書き込む
        for x, y, w, h in faceareas:
            cv2.rectangle(tmpImg, (x, y), (x + w, y + h), 0, thickness=2)
            result = (x, y, w, h)
        cv2.imwrite('img/output.png', tmpImg)
        if result:
            self.face_box = result
        return self

    def clip(self) -> FaceArea:
        x, y, w, h = self.face_box
        self.clipped_img = self.gray_image[y:y+h, x:x+w, ]
        cv2.imwrite('./img/clip.png', self.clipped_img)
        return self

    def norm64(self) -> FaceArea:
        """64x64にする"""
        norm_img = cv2.resize(self.clipped_img, (64, 64))
        cv2.imwrite('./img/norm64.png', norm_img)
        self.norm_img = norm_img
        return self


def getGrayImage(imgPath: str) -> np.ndarray:
    "imgPathの画像をグレースケールにしてndarrayで返す(h, w)"
    img = cv2.imread(imgPath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img/img_gray.png', img_gray)
    return img_gray


# 分類機のパス
cascade_path = "./model/haarcascade_frontalface_default.xml"

# 顔領域をtupleで返す(x, y, w, h)


def faceArea(img: np.ndarray) -> tuple[tuple[int], np.ndarray]:
    tmpImg = img
    result = None
    # 分類機の特徴量を取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔領域を取得
    faceareas = cascade.detectMultiScale(tmpImg)
    # 取得した顔領域からx, y, w, hを取って黒、太さ2の四角形を書き込む
    for x, y, w, h in faceareas:
        cv2.rectangle(tmpImg, (x, y), (x + w, y + h), 0, thickness=2)
        result = (x, y, w, h)
    cv2.imwrite('img/output.png', tmpImg)
    if(result != None):
        return result, img
    else:
        return (0, 0, 0, 0)

# これはテストコード


def main():
    path = './img/tsuru512006_TP_V4.jpg'
    t = faceArea(getGrayImage(path))
    print(t)


if __name__ == '__main__':
    main()
