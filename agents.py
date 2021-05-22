# coding: utf-8
import random

from config import *


class AlphaBeta_Agent:
    
    def __init__(self, env):
        self.env = env

    def evaluation_function(self, color):
        """静的評価関数"""
        board_eval = color * sum([self.env.board[x][y] * BOARD_EVALUATION[x][y]
         for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)])
        puttable_eval = len(self.env.get_actions(color))*5
        value = board_eval + puttable_eval
        return value

    def nega_alpha(self, node, depth, alpha, beta):
        """ネガ・アルファ法で探索する"""
        if depth==0 or len(node)==0:
            return self.evaluation_function(self.env.color)

        evals = {}
        random.shuffle(node)
        for action in node:
            board = self.env.get_board()
            x,y = (action//BOARD_SIZE, action%BOARD_SIZE)
            self.env.board[x][y] = self.env.color
            for _x,_y in self.env.judge(action, "r", self.env.color): self.env.board[_x][_y] *= -1
            self.env.color *= -1
            _node = self.env.get_actions(self.env.color)
            alpha = max(alpha, -1*self.nega_alpha(_node, depth-1, -beta, -alpha))
            self.env.color *= -1
            evals[action] = alpha
            self.env.set_board(board)
            if alpha>=beta: break

        if depth==HEIGHT:
            return [kv[0] for kv in evals.items() if kv[1] == max(evals.values())][0]
        else:
            return alpha

    def alphabeta(self, node, depth):
        """アルファベータ法で探索する"""
        return self.nega_alpha(node, depth, -10000, 10000)

    def get_action(self):
        """行動を返す"""
        return self.alphabeta(self.env.actions, HEIGHT)


class Random_Agent:

    def __init__(self, env):
        self.env = env

    def get_action(self):
        """行動を返す"""
        return random.choice(self.env.actions)


class Human_Agent:

    def __init__(self, env):
        self.env = env

    def get_action(self):
        """行動を返す"""
        while True:
            inputed = input('>')
            if inputed=='q': exit()
            try: action = tuple(map(int, inputed.split()))
            except: continue
            _action = action[0] * BOARD_SIZE + action[1]
            if _action in self.env.actions: break
        return _action
