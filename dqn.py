# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.losses import huber_loss
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

from config import *
from game import SimpleReversi
from agents import Random_Agent


class Q_Network:

    def __init__(self, input_shape=(5,8,8), hidden_size=128, output_size=64, lr=0.001):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.model = self.build_model()

    def build_model(self):  # ネットワーク構築
        model = Sequential()
        # when loading model, faild with InputLayer.
        model.add(Conv2D(self.hidden_size, 3, padding='same', activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(self.hidden_size, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(self.hidden_size, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(self.output_size, activation='softmax'))
        optimizer = RMSprop(learning_rate=self.lr)
        # when loading model, faild with huber loss function.
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def replay(self, memory, batch_size, gamma, targetQN):
        # [0]は行ベクトルを一次元配列に変換している
        inputs = np.zeros((batch_size, *self.input_shape))  #状態を格納
        targets = np.zeros((batch_size, self.output_size))  #行動のQ値を格納
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all():
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号

        self.model.fit(inputs, targets, epochs=1, verbose=0)  # 重みの学習


class Memory:

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class DQN_Agent:

    def __init__(self, env):
        self.env = env
        self.mainQN = Q_Network(INPUT_SHAPE, HIDDEN_SIZE, OUTPUT_SIZE, LR)
        self.targetQN = Q_Network(INPUT_SHAPE, HIDDEN_SIZE, OUTPUT_SIZE, LR)
        self.memory = Memory(MEMORY_SIZE)
        self.path = DQN_PATH['model']

    def info(self):
        self.mainQN.model.summary()
        plot_model(self.mainQN.model, to_file=DQN_PATH['network'], show_shapes=True)

    def remember(self, experience):
        self.memory.push(experience)

    def learn(self):
        if self.memory.__len__() > BATCH_SIZE:
            self.mainQN.replay(self.memory, BATCH_SIZE, GAMMA, self.targetQN)

    def update_target(self):
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())

    def get_action(self, epsilon=0):
        # 徐々に最適行動のみをとる、ε-greedy法
        state = self.env.get_state()

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = self.mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            while action not in self.env.actions:  #合法手でない場合選び直す
                retTargetQs[action] = 0
                action = np.argmax(retTargetQs)
        else:
            action = np.random.choice(self.env.actions)  # ランダムに行動を選択する

        return action

    def load(self):
        try:
            self.mainQN.model = load_model(self.path)
            print('Loaded DQN model.')
        except:
            print('Failed loading DQN model.')

    def save(self):
        try:
            self.mainQN.model.save(self.path)
            print('Saved DQN model.')
        except:
            print('Failed saving DQN model.')


def train():
    # クラス生成
    env = SimpleReversi()
    agent_a = DQN_Agent(env)
    agent_b = Random_Agent(env)

    # モデルのロード
    if LOAD_MODEL:
        agent_a.load()

    # メインルーチン
    total_reward = np.array([])
    total_mean = np.array([])
    islearned = 0

    agent_a.info()

    for episode in range(NUM_EPISODES):

        state = env.reset()  # 環境初期化
        agent_a.update_target()
        episode_reward = 0

        for t in range(0, MAX_STEPS):

            if env.color==1:  # DQN_Agent
                epsilon = 0.001 + 0.9 / (1.0+episode)
                action = agent_a.get_action(epsilon)
            else:  # Other_Agent
                action = agent_b.get_action()

            next_state, reward, done, _ = env.step(action)
            agent_a.remember((state, action, reward, next_state))
            state = next_state
            episode_reward += reward

            if not islearned:
                agent_a.learn()

            if DQN_MODE:
                agent_a.update_target()

            if done:
                env.write_data(DQN_PATH['data'])
                break

        total_reward = np.hstack((total_reward, episode_reward))  # 報酬を記録
        mean = total_reward[-AVG_SIZE:].mean()
        total_mean = np.hstack((total_mean, mean))
        print('Episode %d, finished after %2d time steps | reward %2d | mean %.1f' % (episode, t+1, episode_reward, mean))

        # 複数施行の平均報酬で終了を判断
        if mean >= GOAL_AVG_REWARD and episode+1 >= AVG_SIZE:
            print('Episode %d, trained successfuly!' % episode)
            islearned = 1

        # モデルの定期セーブ
        if SAVE_MODEL and episode%SAVE_CYCLE==0 and episode!=0:
            agent_a.save()

    # モデルのセーブ
    if SAVE_MODEL:
        agent_a.save()

    # 結果のプロット
    plt.plot(np.arange(NUM_EPISODES), total_mean)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Result")
    plt.savefig(DQN_PATH['result'])
    plt.show()



if __name__=="__main__":
    train()
