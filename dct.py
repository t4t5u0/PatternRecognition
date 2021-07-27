import numpy as np


class DCT:
    def __init__(self, N: int) -> None:
        self.N = N  # データ数．
        # 1次元，2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])

        # Nが大きいとメモリリークを起こすので注意
        # MNISTの28x28程度なら問題ない
        self.phi_2d = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i*phi_j

    def dct(self, data: np.ndarray) -> np.ndarray:
        """ 1次元離散コサイン変換を行う """
        return self.phi_1d.dot(data)

    def idct(self, c):
        """ 1次元離散コサイン逆変換を行う """
        return np.sum(self.phi_1d.T * c, axis=1)

    def dct2(self, data):
        """ 2次元離散コサイン変換を行う """
        return np.sum(self.phi_2d.reshape(self.N**2, self.N**2)*data.reshape(self.N**2), axis=1).reshape(self.N, self.N)

    def idct2(self, c):
        """ 2次元離散コサイン逆変換を行う """
        return np.sum((c.reshape(self.N, self.N, 1)*self.phi_2d.reshape(self.N, self.N, self.N**2)).reshape(self.N**2, self.N**2), axis=0).reshape(self.N, self.N)

    def phi(self, k: int) -> np.ndarray:
        """ 離散コサイン変換(DCT)の基底関数 """
        # DCT-II
        if k == 0:
            return np.ones(self.N)/np.sqrt(self.N)
        else:
            return np.sqrt(2.0/self.N)*np.cos((k*np.pi/(2*self.N))*(np.arange(self.N)*2+1))
