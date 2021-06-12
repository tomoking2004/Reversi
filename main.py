# coding: utf-8
from game import Reversi, GUI
from agents import Human_Agent, Random_Agent, AlphaBeta_Agent
#from dqn import DQN_Agent


def app():
    """アプリケーション"""
    # 環境生成
    env = Reversi(None)
    gui = GUI(env)

    # エージェント生成
    human = Human_Agent(env)
    random = Random_Agent(env)
    ab = AlphaBeta_Agent(env)
    #dqn = DQN_Agent(env)
    #dqn.load("models/ddqn_model500.h5")

    # 機能拡張
    env.gui = gui
    env.agent_a = None
    env.agent_b = ab

    # 起動
    env.play()
    gui.mainloop()


if __name__ == "__main__":
    app()
