import math
import random
import sys
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np

class ThreeLayerNN:
    def __init__(self, input_nodes, hide_nodes, output_nodes, lr):
        # 入力層，隠れ層，出力層の層数
        self.input_nodes = input_nodes
        self.hide_nodes = hide_nodes
        self.output_nodes = output_nodes

        # 学習係数
        self.lr = lr

        # 重みの初期化
        self.weight_input_hide = np.random.uniform(-1.0, 1.0, (self.hide_nodes, self.input_nodes))
        self.weight_hide_output = np.random.uniform(-1.0, 1.0, (self.output_nodes, self.hide_nodes))

        # ニューラルネットワーク(NN)の入出力関数
        self.fx = NNfunc

        # ニューラルネットワーク(NN)の入出力関数の微分
        self.fxdaf = derivative_NNfunc

    # 順方向と逆方向の同定
    def Forward_Reverse_direction(self, input_data, ideal_data):
        # 引数のリストを縦ベクトルに変換
        vertical_input = np.array(input_data, ndmin=2, dtype=np.float).T

        # 隠れ層の入力
        input_hide = np.dot(self.weight_input_hide, vertical_input)

        # 隠れ層の出力
        hide_output = self.fx(input_hide)

        # 出力層
        output = np.dot(self.weight_hide_output, hide_output)

        # 誤差計算
        error_output = ideal_data - output

        # 重みの更新
        self.weight_input_hide += self.lr * np.dot(error_output * self.fxdaf(output) * self.weight_hide_output, self.fxdaf(input_hide) * vertical_input.T)
        self.weight_hide_output += self.lr * np.dot((error_output * self.fxdaf(output)), hide_output.T)

        return output[0][0]

# NNの入出力関数
def NNfunc(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

# NNの入力関数の微分
def derivative_NNfunc(x):
    return 2 * np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x)))

# システムの出力
@lru_cache(maxsize=1000)
def y(k):
    if k <= 0:
        return 0
    else:
        Y = 1.38 * y(k-1) - 0.47 * y(k-2) + 0.1 * y(k-1) * y(k-1) - 0.05 * y(k-2) * y(k-3)
        U = 0.25 * u(k-1) + 0.2 * u(k-2)
        V = v() - 1.38 * v() + 0.47 * v()
        return Y + U + V

# システムに加わるノイズ
def v():
    return random.gauss(0, 1 / 100)

# システム出力の目標値
def r(k):
    return 0 if k == 0 else gr(math.sin((2 * math.pi * k) / 50))

def gr(k):
    return 1 if k > 0 else 0.5

# システムの入力
@lru_cache(maxsize=1000)
def u(k):
    return 0 if k <= 0 else u(k-1) + 0.32 * (r(k) - y(k)) - 0.3 * (r(k-1) - y(k-1))

# 評価関数の計算
def evaluation(sys_list, nn_list):
    tmp_list = []
    for k in range(0, 5001):
        tmp_list.append(0.5 * (sys_list[k] - nn_list[k]) * (sys_list[k] - nn_list[k]))

    return tmp_list

# 評価関数のグラフの描画
def draw_eval_graph(title, j_list, hide_node, lr):
    height = np.array(j_list)
    left = [n for n in range(0, 5001)]
    plt.bar(left, height)
    plt.title(title)
    plt.xlabel("sampling number\nHidden layer : {0}\nLearning rate : {1}".format(hide_node, lr))
    plt.ylabel("cost function")
    plt.show()

# 同定システムのグラフの描画
def draw_identification_graph(title, sys_list, nn_list, sys_label_name, nn_label_name, start_k, end_k, hide_node, lr):
    plt.title(title)
    plt.plot(np.linspace(start_k, end_k, 100), sys_list[start_k:end_k], label=sys_label_name)
    plt.plot(np.linspace(start_k, end_k, 100), nn_list[start_k:end_k], label=nn_label_name)
    plt.legend(loc="upper right")
    plt.ylim(-0.4, 1.4)
    plt.ylabel("output")
    plt.xlim(start_k, end_k)
    plt.xlabel("sampling number\nHidden layer : {0}\nLearning rate : {1}".format(hide_node, lr))
    plt.show()

if __name__ == '__main__':
    # 再帰をする回数を指定
    sys.setrecursionlimit(10000000)
    # システムの出力のリスト
    y_list = []
    # システムの入力のリスト
    u_list = []

    # 時間区間ごとのシステムの出力
    for k in range(0, 5001):
        y_list.append(y(k))

    # 時間区間ごとのシステムの入力
    for k in range(0, 5001):
        u_list.append(u(k))

    # 入力層，隠れ層，出力層の層数
    input_node = 4
    hide_node = 10
    output_node = 1

    # 順方向の学習係数
    lr = 0.185

    # 順方向のNNの出力のリスト
    forward_ynn_list = []

    # 順方向のNNの初期化
    forward_nn = ThreeLayerNN(input_node, hide_node, output_node, lr)

    # 順方向のNNの出力
    for k in range(0, 5001):
        if k > 1:
            input_x = [y_list[k - 1], y_list[k - 2], u_list[k - 1], u_list[k - 2]]
            forward_ynn_list.append(forward_nn.Forward_Reverse_direction(input_x, y_list[k]))
        else:
            input_x = [0, 0, 0, 0]
            forward_ynn_list.append(forward_nn.Forward_Reverse_direction(input_x, y_list[k]))

    # 逆方向のNNの出力リスト
    reverse_ynn_list = []

    # 逆方向のNNの初期化
    reverse_nn = ThreeLayerNN(input_node, hide_node, output_node, lr)

    # 逆方向のNNの出力
    for k in range(0, 5001):
        if k > 1:
            input_x = [y_list[k], y_list[k - 1], y_list[k - 2], u_list[k - 2]]
            reverse_ynn_list.append(reverse_nn.Forward_Reverse_direction(input_x, u_list[k]))
        else:
            input_x = [0, 0, 0, 0]
            reverse_ynn_list.append(reverse_nn.Forward_Reverse_direction(input_x, u_list[k]))

    # 順方向の評価関数のリスト
    forward_j_list = evaluation(y_list, forward_ynn_list)

    # 順方向の評価関数のグラフ描画
    draw_eval_graph("Forward Evaluation Function", forward_j_list, hide_node=hide_node, lr=lr)

    # 逆方向の評価関数のリスト
    reverse_j_list = evaluation(u_list, reverse_ynn_list)

    # 逆方向の評価関数のグラフ描画
    draw_eval_graph("Reverse Evaluation Function", reverse_j_list, hide_node=hide_node, lr=lr)

    # グラフの横軸のはじめと終わり
    start_k = 0
    end_k = start_k + 100

    # 順方向の同定システムのグラフ描画
    draw_identification_graph("Forward Identification System", y_list, forward_ynn_list, sys_label_name="y", nn_label_name="ynn", start_k=start_k, end_k=end_k, hide_node=hide_node, lr=lr)
    draw_identification_graph("Forward Identification System", y_list, forward_ynn_list, sys_label_name="y", nn_label_name="ynn", start_k=4900, end_k=5000, hide_node=hide_node, lr=lr)

    # 逆方向の同定システムのグラフ描画
    draw_identification_graph("Reverse Identification System", u_list, reverse_ynn_list, sys_label_name="u", nn_label_name="ynn", start_k=start_k, end_k=end_k, hide_node=hide_node, lr=lr)
    draw_identification_graph("Reverse Identification System", u_list, reverse_ynn_list,sys_label_name="u", nn_label_name="ynn", start_k=4900, end_k=5000, hide_node=hide_node, lr=lr)
