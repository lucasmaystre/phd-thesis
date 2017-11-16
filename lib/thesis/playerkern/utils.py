import enum
import collections
import numpy as np

from .data_models import Game, Team

from math import log


class Outcome(enum.Enum):
    """Simple enumeration of the outcomes of a game."""
    win = 1
    tie = 2
    loss = 3


class TestSet:

    def __init__(self, games):
        self._games = games

    @classmethod
    def from_games(cls, where_clause):
        games = list()
        for game in Game.select(Game, Team).join(Team).where(where_clause):
            games.append(game)
        return cls(games)

    def evaluate_fct(self, predictor):
        results = collections.OrderedDict()
        for game in self._games:
            if game.score_home_ft > game.score_away_ft:
                outcome = Outcome.win
            elif game.score_home_ft < game.score_away_ft:
                outcome = Outcome.loss
            else:
                outcome = Outcome.tie
            results[game.id] = (game, outcome, predictor(game))
        return TestResults(results)

    def evaluate(self, model):
        return self.evaluate_fct(model.predict)


class TestResults:

    def __init__(self, results):
        self._results = results

    def __len__(self):
        return len(self._results)

    @property
    def zero_one_loss(self):
        loss = 0
        for game, outcome, pred in self._results.values():
            if np.argmax(pred) == 0 and outcome is not Outcome.win:
                loss += 1
            elif np.argmax(pred) == 1 and outcome is not Outcome.tie:
                loss += 1
            elif np.argmax(pred) == 2 and outcome is not Outcome.loss:
                loss += 1
        return loss

    @property
    def log_loss(self):
        loss = 0
        for game, outcome, pred in self._results.values():
            try:
                if outcome is Outcome.win:
                    loss += -log(pred[0])
                elif outcome is Outcome.tie:
                    loss += -log(pred[1])
                else:  # outcome is Outcome.loss
                    loss += -log(pred[2])
            except ValueError:
                # We probably tried to compute `log(0)`.
                return float("inf")
        return loss

    def print_summary(self):
        print("number of samples:", len(self))
        print("0-1 loss: {:.3f}".format(self.zero_one_loss))
        print("log loss: {:.3f}".format(self.log_loss))


class Features:

    def __init__(self):
        self._cnter = 0
        self._idx = dict()
        self._groups = collections.defaultdict(list)

    def __len__(self):
        return self._cnter

    def add(self, feature_name, group="default"):
        if feature_name not in self._idx:
            self._idx[feature_name] = self._cnter
            self._groups[group].append(self._cnter)
            self._cnter += 1

    def get_group(self, group):
        return np.array(self._groups[group], dtype=int)

    def get_idx(self, feature_name):
        return self._idx[feature_name]

    def new_vector(self):
        return FeatureVector(self)


class FeatureVector:

    def __init__(self, features, base=None):
        if base is not None:
            self._vec = np.copy(base)
        else:
            self._vec = np.zeros(len(features), dtype=float)
        self._feats = features

    def __setitem__(self, key, val):
        self._vec[self._feats.get_idx(key)] = val

    def __getitem__(self, key):
        return self._vec[self._feats.get_idx(key)]

    def as_array(self):
        return self._vec
