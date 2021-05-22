# coding: utf-8
import random, os, csv, datetime
from tkinter.font import names
import numpy as np
import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog

from config import *


class SimpleReversi:
    """簡易オセロ環境"""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.board_history = []
        self.w_data = []
        self.cnt = 0
        self.color = -1
        self.board[PIECE_XY][PIECE_XY] = 1
        self.board[PIECE_XY+1][PIECE_XY] = -1
        self.board[PIECE_XY][PIECE_XY+1] = -1
        self.board[PIECE_XY+1][PIECE_XY+1] = 1
        self.actions = self.get_actions(self.color)
        self.board_history.append(self.get_board())

    def get_board(self, k=0):
        if k==0:  # 現在の盤面
            return self.board.copy()
        elif 0 <= self.cnt+k < len(self.board_history):  # cnt+k手目の盤面
            return self.board_history[self.cnt+k].copy()
        else:  # ゼロ詰の盤面(cnt+k手目が存在しない)
            return np.zeros((BOARD_SIZE, BOARD_SIZE))

    def set_board(self, board):
        self.board = board

    def get_actions(self, color):
        return [action for action in range(BOARD_SIZE**2) if self.judge(action, "p", color)]

    def judge(self, action, cnf, color):
        _action = (action//BOARD_SIZE, action%BOARD_SIZE)
        x,y = _action
        if cnf=="p" and self.board[x][y]!=0: return False
        reverses = []
        for _x,_y in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
            r = 0
            x,y = _action
            rs = []
            while 0<=x+_x<BOARD_SIZE and 0<=y+_y<BOARD_SIZE:
                r += 1
                x += _x
                y += _y
                if self.board[x][y]==color*-1: rs.append((x,y))
                else: break
            if self.board[x][y]==color and r!=1:
                if cnf=="p": return True
                if cnf=="r": reverses.extend(rs)
        if cnf=="p": return False
        if cnf=="r": return reverses

    def put(self, action):
        x,y = (action//BOARD_SIZE, action%BOARD_SIZE)
        if action!=BOARD_SIZE**2:  # パスでない
            self.board[x][y] = self.color
            for _x,_y in self.judge(action, "r", self.color): self.board[_x][_y] *= -1
        self.cnt += 1
        self.color *= -1
        self.actions = self.get_actions(self.color)
        self.w_data.append(action)
        self.board_history.append(self.get_board())

    def get_state(self):
        state = []
        for k in range(2):  # k手前まで追加
            state.append(np.where(self.get_board(-k)<0, 1, 0))  # 黒石の位置
            state.append(np.where(self.get_board(-k)>0, 1, 0))  # 白石の位置
        if self.color==-1:
            state.append(np.ones((BOARD_SIZE, BOARD_SIZE)))  # 黒の手番
        else:
            state.append(np.zeros((BOARD_SIZE, BOARD_SIZE)))  # 白の手番
        return np.reshape(state, (1, *INPUT_SHAPE))

    def get_reward(self, terminal):  # 報酬設定
        if terminal:
            result = np.reshape(self.get_board(), (BOARD_SIZE**2, )).sum()
            if result>0: return 1
            elif result<0: return -1
            else: return 0
        else:
            return 0

    def is_terminal(self):
        if len(self.get_actions(self.color))==0\
         and len(self.get_actions(self.color*-1))==0:
            return True
        else:
            return False

    # メインメソッド
    def write_data(self, path):
        if os.path.isfile(path): mode = "a"
        else: mode = "w"
        with open(path, mode) as f:
            writer = csv.writer(f)
            writer.writerow(self.w_data)

    def step(self, action):  # 遷移関数
        if not(action==BOARD_SIZE**2 or action in self.actions):  # 反則したら減点して終了
            next_state = self.get_state()
            return next_state, -1, True, None
        self.put(action)
        terminal = self.is_terminal()
        reward = self.get_reward(terminal)
        next_state = self.get_state()
        if len(self.actions) or terminal:  # 可動or終端
            return next_state, reward, terminal, None
        else:
            return self.step(BOARD_SIZE**2)  # 自動でパスして遷移

    def reset(self):
        self.__init__()
        state = self.get_state()
        return state


class Reversi(SimpleReversi):
    """オセロ環境"""

    def __init__(self, gui, agent_a=None, agent_b=None):
        super().__init__()
        self.r_data = []
        self.gui = gui
        self.agent_a = agent_a
        self.agent_b = agent_b

    def play(self):
        self.game()

    def save_data(self):
        if os.path.isfile(SAVE_DATA_PATH): mode = "a"
        else: mode = "w"
        with open(SAVE_DATA_PATH, mode) as f:
            writer = csv.writer(f)
            now = datetime.datetime.now()
            data = [now.strftime('%Y%m%d_%H%M%S')] + self.w_data
            writer.writerow(data)

    def load_data(self, idx):#
        if os.path.isfile(SAVE_DATA_PATH):
            self.restart()
            with open(SAVE_DATA_PATH, "r") as f:
                reader = csv.reader(f)
                alldata = [row for row in reader]
                self.r_data = [int(col) for col in alldata[idx][1:]]

    def write_data(self):
        if os.path.isfile(REPLAY_DATA_PATH): mode = "a"
        else: mode = "w"
        with open(REPLAY_DATA_PATH, mode) as f:
            writer = csv.writer(f)
            now = datetime.datetime.now()
            data = [now.strftime('%Y%m%d_%H%M%S')] + self.w_data
            writer.writerow(data)

    def read_data(self, idx):#
        if os.path.isfile(REPLAY_DATA_PATH):
            self.restart()
            with open(REPLAY_DATA_PATH, "r") as f:
                reader = csv.reader(f)
                alldata = [row for row in reader]
                self.r_data = [int(col) for col in alldata[idx][1:]]

    def delete_data(self):
        if os.path.isfile(REPLAY_DATA_PATH):
            os.remove(REPLAY_DATA_PATH)
        if os.path.isfile(SAVE_DATA_PATH):
            os.remove(SAVE_DATA_PATH)

    def put(self, action):  # 改
        del self.w_data[self.cnt:]
        del self.board_history[self.cnt:]
        self.board_history.append(self.get_board())
        x,y = (action//BOARD_SIZE, action%BOARD_SIZE)
        if action!=BOARD_SIZE**2:  # パスでない
            self.board[x][y] = self.color
            for _x,_y in self.judge(action, "r", self.color): self.board[_x][_y] *= -1
        else:
            self.gui.pass_message()
        self.cnt += 1
        self.color *= -1
        self.actions = self.get_actions(self.color)
        self.w_data.append(action)
        self.board_history.append(self.get_board())
        self.game()

    def random_put(self):
        self.put(random.choice(self.actions))

    def set_turn(self, k):
        if 0<=self.cnt+k<=len(self.board_history):
            self.set_board(self.get_board(k))
            self.cnt += k
            self.actions = self.get_actions(self.color)
            self.game()

    def turn(self):
        if self.is_terminal():  # 終了
            self.winner_judge()
            self.gui.write_ask()
            self.gui.restart_ask()
            self.gui.actable = True
        elif len(self.r_data):  # ロード
            action = self.r_data[self.cnt]
            if self.cnt==len(self.r_data)-1: self.r_data.clear()
            self.put(action)
        elif len(self.actions)==0:  # パス
            self.put(BOARD_SIZE**2)
        elif self.color==-1 and self.agent_a is not None:  # aのターン
            self.put(self.agent_a.get_action())
        elif self.color==1 and self.agent_b is not None:  # bのターン
            self.put(self.agent_b.get_action())
        else:  # guiのターン
            self.gui.set_limit()
            self.gui.actable = True

    def winner_judge(self):
        difference = int(np.reshape(self.get_board(), [BOARD_SIZE**2, ]).sum())
        self.gui.result_message(difference)

    def game(self):
        self.gui.actable = False
        self.gui.cancel_limit()
        self.gui.canvas_update()
        self.gui.after(TURN_CYCLE,self.turn)

    def restart(self):
        self.__init__(self.gui, self.agent_a, self.agent_b)
        self.game()


class GUI(tk.Tk):
    """グラフィカル・ユーザ・インターフェース"""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.canvas = tk.Canvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, 
                                bg=CANVAS_COLOR, highlightthickness=0)
        self.menu_top = tk.Menu(self, tearoff=False)
        self.coordinates = [[(BOARD_X+PANEL_SIZE*y, BOARD_Y+PANEL_SIZE*x)
            for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]
        self.show_puttable = False
        self.actable = False
        self.limiter = id(None)
        self.initialize()
        
    def initialize(self):
        menu_file = tk.Menu(self.menu_top, tearoff=0)
        menu_open = tk.Menu(self.menu_top, tearoff=0)
        menu_config = tk.Menu(self.menu_top, tearoff=0)
        self.menu_top.add_cascade(label="ファイル...", menu=menu_file)
        self.menu_top.add_cascade(label="環境設定...", menu=menu_config)
        self.menu_top.add_separator()
        self.menu_top.add_command(label="再起動", command=self.env.restart)
        self.menu_top.add_command(label="強制終了", command=exit)
        menu_file.add_cascade(label="開く...", menu=menu_open)
        menu_file.add_command(label="保存", command=self.env.save_data)
        menu_open.add_command(label="続きからプレイする", command=self.load_data)
        menu_open.add_command(label="リプレイを見る", command=self.read_data)
        menu_config.add_command(label="パネル変色", command=self.turn_show_puttable)
        menu_config.add_command(label="データ削除", command=self.delete_ask)
        self.bind("<Button-1>", self.put)
        self.bind("<Button-2>", self.show_popup)
        if RETURN_TWICE:
            self.bind("<Left>", self.back_twice)
            self.bind("<Right>", self.forward_twice)
        self.protocol("WM_DELETE_WINDOW", self.save_ask)
        self.canvas.pack()

    def canvas_update(self):
        # タイトル更新
        if self.env.color==-1:
            self.title("黒のターン({}手目)".format(self.env.cnt+1))
        else:
            self.title("白のターン({}手目)".format(self.env.cnt+1))
        # 盤面更新
        self.canvas.delete("all") #削除
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                _x,_y = self.coordinates[x][y]
                # パネル更新
                if self.show_puttable and BOARD_SIZE*x+y in self.env.actions: fill = PUTTABLE_COLOR
                else: fill = PANEL_COLOR
                self.canvas.create_rectangle(_x,_y,_x+PANEL_SIZE,_y+PANEL_SIZE,fill=fill,tag="all")
                # 駒更新
                if self.env.board[x][y]==-1: fill = PIECE_COLOR_A
                elif self.env.board[x][y]==1: fill = PIECE_COLOR_B
                else: fill = None
                self.canvas.create_oval(_x+PANEL_SIZE//2-PIECE_SIZE, _y+PANEL_SIZE//2-PIECE_SIZE,
                    _x+PANEL_SIZE//2+PIECE_SIZE, _y+PANEL_SIZE//2+PIECE_SIZE, fill=fill, width=0, tag="all")

    def set_limit(self):
        if LIMITER:
            self.limiter = self.after(LIMIT, self.env.random_put)

    def cancel_limit(self):
        if LIMITER:
            self.after_cancel(self.limiter)

    def turn_show_puttable(self):
        if self.show_puttable:
            self.show_puttable = False
        else:
            self.show_puttable = True
        self.canvas_update()

    # message
    def pass_message(self):
        if self.env.color==-1:
            tk.messagebox.showinfo("Reversi","黒は何処にも置けないためパスします")
        else:
            tk.messagebox.showinfo("Reversi","白は何処にも置けないためパスします")

    def result_message(self, difference):
        if difference<0:
            tk.messagebox.showinfo("Reversi","黒の{}石勝ち".format(difference*-1))
        elif difference>0:
            tk.messagebox.showinfo("Reversi","白の{}石勝ち".format(difference))
        else:
            tk.messagebox.showinfo("Reversi","引き分け")

    # ask
    def write_ask(self):
        ans = tk.messagebox.askyesno("Reversi", "後でリプレイを見ることができます。\nリプレイデータを書き込みますか？")
        if ans: self.env.write_data()

    def save_ask(self):
        if self.env.cnt>0:
            ans = tk.messagebox.askyesnocancel("Reversi", "後で続きからプレイできます。\nセーブして終了しますか？")
            if ans is not None:
                if ans: self.env.save_data()
                exit()
        else:
            exit()

    def delete_ask(self):
        ans = tk.messagebox.askyesno("Reversi", "削除したデータは元に戻りません。\n本当に削除しますか？")
        if ans:
            self.env.delete_data()
            tk.messagebox.showinfo("Reversi", "データを削除しました。")

    def restart_ask(self):
        ans = tk.messagebox.askyesno("Reversi", "もう一度プレイしますか？")
        if ans: self.env.restart()

    # listbox
    def listbox(self, names, func):
        master = tk.Tk()
        master.title("Reversi")
        tk.Label(master, text="データを選択").grid(row=1, column=3, sticky="n")
        frame = tk.Frame(master)
        frame.grid(row=2, column=3, padx=10, pady=10)
        lb = tk.Listbox(frame)
        for name in names:
            lb.insert(tk.END, name)
        lb.pack(side=tk.LEFT)
        bar =tk.Scrollbar(frame, command=lb.yview)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        lb.config(yscrollcommand=bar.set)
        def cmd():
            name = lb.get(tk.ACTIVE)
            if name!='':
                idx = names.index(name)
                func(idx)
            master.destroy()
        btn = tk.Button(master, text="OK", width=10, command=cmd)
        btn.grid(row=5, column=3, pady=5)

    def load_data(self):
        if not os.path.isfile(SAVE_DATA_PATH):
            open(SAVE_DATA_PATH, "w")
        with open(SAVE_DATA_PATH, "r") as f:
            reader = csv.reader(f)
            names = [row[0] for row in reader]
        self.listbox(names, self.env.load_data)

    def read_data(self):
        if not os.path.isfile(REPLAY_DATA_PATH):
            open(REPLAY_DATA_PATH, "w")
        with open(REPLAY_DATA_PATH, "r") as f:
            reader = csv.reader(f)
            names = [row[0] for row in reader]
        self.listbox(names, self.env.read_data)

    # events
    def put(self, event):
        if self.actable:
            for action in self.env.actions:
                x,y = (action//BOARD_SIZE, action%BOARD_SIZE)
                _x,_y = self.coordinates[x][y]
                if _x<event.x<_x+PANEL_SIZE and _y<event.y<_y+PANEL_SIZE:
                    self.env.put(action)

    def show_popup(self, event):
        if self.actable:
            self.menu_top.post(event.x_root, event.y_root)

    def back_twice(self, event):
        if self.actable:
            self.env.set_turn(-2)

    def forward_twice(self, event):
        if self.actable:
            self.env.set_turn(2)
