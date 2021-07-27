from __future__ import annotations

import matplotlib.pyplot as plt

from faceArea import *
from norm import *
from dct import *


def main():
    path = './img/tsuru512006_TP_V4.jpg'
    # norm64(clip(*faceArea(getGrayImage(path))))
    face = FaceArea(path).getGrayImage().faceArea().clip().norm64()
    norm_img_path = face.norm_img_path
    norm_img = face.norm_img

    dct = DCT(64)  # 離散コサイン変換を行うクラスを作成

    c = dct.dct2(norm_img)  # 2次元離散コサイン変換
    y = dct.idct2(c)  # 2次元離散コサイン逆変換

    # 元の画像と復元したものを表示
    plt.subplot(1, 2, 1)
    plt.imshow(norm_img, cmap="Greys")
    plt.title("original")
    plt.xticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap="Greys")
    plt.title("restored")
    plt.xticks([])
    plt.savefig("img/result.png")
    plt.show()


if __name__ == '__main__':
    main()
