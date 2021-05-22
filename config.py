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
# Q_Network
INPUT_SHAPE = (5, BOARD_SIZE, BOARD_SIZE)  # 入力形状(状態の形状)
HIDDEN_SIZE = 128  # 隠れ層のニューロンの数
OUTPUT_SIZE = 64  # 出力サイズ(行動のサイズ)
LR = 0.001  # 学習係数
GAMMA = 0.90  # 割引係数
DQN_PATH = {'model':'models/ddqn_model500.h5',
            'network':'models/Q-network500.png',
            'data':'data/train500.csv',
            'result':'models/result500.png'}  # DQNのパス

# Memory
MEMORY_SIZE = 10000  # メモリサイズ
BATCH_SIZE = 32  # バッチサイズ

# Trainer
DQN_MODE = 0  # DQNモード(0:DDQN, 1:DQN)
LOAD_MODEL = True  # モデルのロード
SAVE_MODEL = True  # モデルのセーブ
SAVE_CYCLE = 20  # 定期セーブ周期
NUM_EPISODES = 500  # エピソード数
MAX_STEPS = 100  # 最大ステップ数
GOAL_AVG_REWARD = 1  # 目標平均報酬
AVG_SIZE = 10  # 平均する集合の大きさ
