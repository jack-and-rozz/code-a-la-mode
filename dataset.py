# coding: utf-8
import sys, os, json, collections, re, random
import numpy as np
from utils import dotDict, recDotDict, recDotDefaultDict, flatten 
from pprint import pprint
from play import tensor2state, state2tensor

rewards_expr=re.compile("T \'(.+?)\'")

def parse_rewards_sum(log_strs):
  system_turns = set([0, 1, 202, 203, 404, 405, 606])

  def _parse_turn(i, view_str):
    view_str = view_str.split('\n')[1]
    view_str = json.loads(view_str)['entitymodule']
    if 'T \'' not in view_str:
      return 0
    else:
      shown_scores = rewards_expr.search(view_str).group(1).split()
      if i in range(2, 202):
        score = int(shown_scores[0])
      elif i in range(204, 404):
        score = int(shown_scores[1])
      else:
        score = int(shown_scores[2])
      return score

  def _scores_to_rewards(scores):
    reward_turns = [i for i, r in enumerate(scores) if r != 0]
    for i in reversed(range(len(reward_turns)-1)):
      scores[reward_turns[i+1]] -= scores[reward_turns[i]]
    # Sum up rewards obtained in each player's turn.
    rewards = [scores[i] + scores[i+1] for i in range(0, 200, 2)]
    return rewards # [100]

  scores = [_parse_turn(i, _str) for i, _str in enumerate(log_strs) if i not in system_turns] 

  rewards = [_scores_to_rewards(scores[i:i+200]) for i in range(0, 600, 200)] # tuple of reward lists in each game. [3, 100]
  return rewards
    


def parse_log(path):
  '''
  A game contains 607 turns. Since turns 0-1, 202-203, 404-405, and 606 are for setup and closing, agents send no logs. 
  Player 1 joins in the 1st and 2nd games. 
  Player 2 joins in the 1st and 3rd games.
  '''
  d = json.load(open(path))
  # Player 1 and Player 2 must be the agents made by this code.
  player1 = [recDotDict(json.loads(turn_log)) for turn_log in d['errors']['0'] if turn_log] # logs in turns [2, 4, ... 200] and [205, 207, ... 403].
  player2 = [recDotDict(json.loads(turn_log)) for turn_log in d['errors']['1'] if turn_log] # logs in turns [3, 5, ... 201] and [406, 408, ... 604].

  # Player 3 is in charge of logs in turns [204, 206, ... 402] and [407, 409, ... 605] but they are not available as we use other's agents as Player 3.
  player3 = [recDotDict(json.loads(turn_log)) for turn_log in d['errors']['2'] if turn_log]

  rewards = parse_rewards_sum(d['views']) #[3, 200]

  for i in range(100):
    player1[i].rewards = rewards[0][i]
    player2[i].rewards = rewards[0][i]
    player3[i].rewards = rewards[1][i]

  for i in range(100, 200):
    player1[i].rewards = rewards[1][i-100]
    player2[i].rewards = rewards[2][i-100]
    player3[i].rewards = rewards[2][i-100]

  return (player1[:100], player1[100:200], player2[:100], player2[100:200], player3[:100], player3[100:200])


def read_score(file_path):
  return [int(l) for l in open(file_path)]

def print_batch(batch):
  for k, v in batch.items():
    print(k, v.shape)

class Dataset():
  def __init__(self, log_files, batch_size):
    self.batch_size = batch_size
    self.scores = [read_score(file_path[:-5] + '.score') for file_path in log_files]
    self.data = [parse_log(file_path) for i, file_path in enumerate(log_files) if -1 not in self.scores[i]] # 6 * num_games (player1-round1, player2-round1, player1-round2, player2-round3)
    self.scores = [s for s in self.scores if -1 not in s]
    idxs = list(range(len(self.data)))
    random.shuffle(idxs)
    self.data = flatten([self.data[i] for i in idxs])

    self.average_score = int(sum([s[0] for s in self.scores])/len(self.scores))
    self.enemy_average_score = int(sum([s[2] for s in self.scores])/len(self.scores))

  def __iter__(self):
    for i in range(0, len(self.data), self.batch_size):
      yield self.create_batch(self.data[i:i+self.batch_size])

  def create_batch(self, data):
    batch = dotDict()
    games = [self.game2tensor(game) for game in data]
    for k in games[0].keys():
      batch[k] = np.array([game[k] for game in games])
    return batch
  
  def game2tensor(self, game):
    game = [state2tensor(g) for g in game]
    game_state = dotDict()
    for k in game[0].keys():
      game_state[k] = np.array([turn[k] for turn in game])
    return game_state
