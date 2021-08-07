# coding: utf-8
import os

"""環境定数"""

#----------debug----------
# When using tensorflow or keras
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#----------game----------
# SimpleReversi
BOARD_SIZE = 8  #パネル数の平方根
PIECE_XY = BOARD_SIZE//2-1  #駒の配置(左上)

# Reversi
REPLAY_DATA_PATH = "data/replay.csv"  #リプレイデータのパス
SAVE_DATA_PATH = "data/save.csv"  #セーブデータのパス

# GUI
CANVAS_WIDTH = 690  #キャンバスの幅
CANVAS_HEIGHT = 690  #キャンバスの高さ
CANVAS_COLOR = "grey20"  #キャンバスの色
BOARD_X = 25  #フィールドの位置
BOARD_Y = 25  #フィールドの位置
PANEL_SIZE = 80  #パネルの辺の長さ
PANEL_COLOR = "dark green"  #パネルの色
PUTTABLE_COLOR = "greenyellow"  #置けるパネルの色
PIECE_SIZE = 36  #駒の半径
PIECE_COLOR_A = "grey15"  #先攻の色
PIECE_COLOR_B = "grey90"  #後攻の色
TURN_CYCLE = 50  #ターン周期(ms)
LIMITER = False  #時間制限
LIMIT = 10000  #制限時間(ms)
RETURN_TWICE = True  #置き直し

#----------agents----------
# AlphaBeta_Agent
BOARD_EVALUATION = [[120,-40, 20,  5,  5, 20,-40,120],
                    [-40,-80, -1, -1, -1, -1,-80,-40],
                    [ 20, -1,  5,  1,  1,  5, -1, 20],
                    [  5, -1,  1,  0,  0,  1, -1,  5],
                    [  5, -1,  1,  0,  0,  1, -1,  5],
                    [ 20, -1,  5,  1,  1,  5, -1, 20],
                    [-40,-80, -1, -1, -1, -1,-80,-40],
                    [120,-40, 20,  5,  5, 20,-40,120]]  #盤面評価値
HEIGHT = 5  #探索木の高さ

#----------dqn----------
# params
INPUT_SHAPE = (5, BOARD_SIZE, BOARD_SIZE)  # 入力形状(状態の形状)
OUTPUT_SIZE = 4  # 出力サイズ(行動のサイズ)
LR = 0.001  # 学習係数
GAMMA = 0.95  # 割引係数
MEMORY_SIZE = 10000  # メモリサイズ
BATCH_SIZE = 500  # バッチサイズ
# train
DQN_MODEL_NAME = 'dqn'  # DQNモデルの名前
DQN_MODE = 0  # DQNモード(0:DDQN, 1:DQN)
SAVE_CYCLE = 100  # 定期セーブ周期
NUM_EPISODES = 1000  # エピソード数
MAX_STEPS = 500  # 最大ステップ数
GOAL_AVG_REWARD = 100  # 目標平均報酬
AVG_SIZE = 10  # 平均する集合の大きさ
