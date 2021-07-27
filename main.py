from __future__ import annotations

import matplotlib.pyplot as plt

from faceArea import *
from norm import *


def main():
    path = './img/tsuru512006_TP_V4.jpg'
    # norm64(clip(*faceArea(getGrayImage(path))))
    face = FaceArea(path)
    norm_img_path = face.getGrayImage().faceArea().clip().norm64().norm_img_path
    norm_img = face.getGrayImage().faceArea().clip().norm64().norm_img


if __name__ == '__main__':
    main()
