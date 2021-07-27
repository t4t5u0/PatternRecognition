import numpy as np
import cv2


def clip(box: tuple[int], img: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    clipped_img = img[y:y+h, x:x+w,]
    cv2.imwrite('./img/clip.png', clipped_img)
    return clipped_img


def norm64(img: np.ndarray) -> np.ndarray:
    """64x64にする"""
    norm_img = cv2.resize(img, (64, 64))
    cv2.imwrite('./img/norm64.png', norm_img)
    return norm_img
