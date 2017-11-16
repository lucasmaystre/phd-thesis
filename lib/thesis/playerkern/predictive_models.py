import abc
import GPy
import numpy as np

from .data_models import Actor, Game, Team
from .utils import Features

from GPy.inference.latent_function_inference import expectation_propagation


class PredictiveModel(metaclass=abc.ABCMeta):

    """A predictive model."""

    @abc.abstractmethod
    def fit(self, games):
        """Fit a model given a set of games."""

    @abc.abstractmethod
    def predict(self, game):
        """Compute outcome probabilities for a game."""


class ActorGPModel(PredictiveModel):

    def __init__(self, alpha, home_advantage=True, **hyperparams):
        self.alpha = alpha
        self.home_advantage = home_advantage
        self.hyperparams = hyperparams
        self.features = Features()
        self.gp = None

    def _build_features(self):
        self.features.add("home_adv")
        for actor in Actor.select():
            self.features.add(actor.id, group="actors")

    def _build_featmat(self, games):
        # Each row is a game, each column is a player.
        featmat = list()
        for game in games:
            if len(game.participants) == 0:
                # There were no players for this game.
                continue
            if game.score_home_ft >= game.score_away_ft:
                vec = self.features.new_vector()
                for p in game.participants:
                    if p.ratio == 0:
                        continue
                    elif p.team_id == game.team_home_id:
                        vec[p.actor_id] = +p.ratio
                    else:
                        vec[p.actor_id] = -p.ratio
                if not game.is_neutral:
                    vec["home_adv"] = +1.0
                featmat.append(vec.as_array())
            if game.score_home_ft <= game.score_away_ft:
                vec = self.features.new_vector()
                for p in game.participants:
                    if p.ratio == 0:
                        continue
                    elif p.team_id == game.team_home_id:
                        vec[p.actor_id] = -p.ratio
                    else:
                        vec[p.actor_id] = +p.ratio
                if not game.is_neutral:
                    vec["home_adv"] = -1.0
                featmat.append(vec.as_array())
        self.featmat = np.array(featmat)

    def fit(self, games, parallel=True):
        self._build_features()
        self._build_featmat(games)
        # Kernel for the teams.
        indices = self.features.get_group("actors")
        var = self.hyperparams.get("actors_var", 1.0)
        k_actors = GPy.kern.Linear(
                input_dim=len(indices), active_dims=indices, variances=var,
                name="actors")
        # Kernel for the home advantage.
        idx = self.features.get_idx("home_adv")
        var = self.hyperparams.get("home_adv_var", 1.0)
        k_home_adv = GPy.kern.Linear(
                input_dim=1, active_dims=[idx], variances=var, name="home_adv")
        # Putting it all together.
        kernel = k_actors
        if self.home_advantage:
            kernel += k_home_adv
        link = GPy.likelihoods.link_functions.RaoKupper(self.alpha)
        likelihood = GPy.likelihoods.Bernoulli(gp_link=link)
        method = expectation_propagation.EP(parallel_updates=parallel)
        n = self.featmat.shape[0]
        self.gp = GPy.core.GP(
                X=self.featmat, Y=np.ones((n, 1)), kernel=kernel,
                likelihood=likelihood, inference_method=method)

    def predict(self, game):
        vec = self.features.new_vector()
        for p in game.participants:
            if p.is_starter and p.team_id == game.team_home_id:
                vec[p.actor_id] = +1.0
            elif p.is_starter and p.team_id == game.team_away_id:
                vec[p.actor_id] = -1.0
        if not game.is_neutral:
            vec["home_adv"] = +1.0
        # Second, we compute the predictive mean and variance
        Xnew = np.array([vec.as_array()])
        mean, _ = self.gp.predict(Xnew)
        prob_win = mean[0, 0]
        mean, _ = self.gp.predict(-Xnew)
        prob_lose = mean[0, 0]
        prob_draw = 1.0 - prob_win - prob_lose
        return (prob_win, prob_draw, prob_lose)


class TeamGPModel(PredictiveModel):

    def __init__(self, alpha, home_advantage=True, **hyperparams):
        self.alpha = alpha
        self.home_advantage = home_advantage
        self.hyperparams = hyperparams
        self.features = Features()
        self.gp = None

    def _build_features(self, games):
        self.features.add("home_adv")
        # TODO Maybe here we should just use all the teams in the DB.
        for game in games:
            self.features.add(game.team_home.name, group="teams")
            self.features.add(game.team_away.name, group="teams")
        self.features.add("Poland", group="teams")
        self.features.add("Ukraine", group="teams")

    def _build_featmat(self, games):
        featmat = list()
        for game in games:
            if game.score_home_ft >= game.score_away_ft:
                vec = self.features.new_vector()
                vec[game.team_home.name] = +1.0
                vec[game.team_away.name] = -1.0
                if not game.is_neutral:
                    vec["home_adv"] = +1.0
                featmat.append(vec.as_array())
            if game.score_home_ft <= game.score_away_ft:
                vec = self.features.new_vector()
                vec[game.team_home.name] = -1.0
                vec[game.team_away.name] = +1.0
                if not game.is_neutral:
                    vec["home_adv"] = -1.0
                featmat.append(vec.as_array())
        self.featmat = np.array(featmat)

    def fit(self, games, parallel=True):
        self._build_features(games)
        self._build_featmat(games)
        # Kernel for the teams.
        indices = self.features.get_group("teams")
        var = self.hyperparams.get("teams_var", 1.0)
        k_teams = GPy.kern.Linear(input_dim=len(indices), active_dims=indices,
                variances=var, name="teams")
        # Kernel for the home advantage.
        idx = self.features.get_idx("home_adv")
        var = self.hyperparams.get("home_adv_var", 1.0)
        k_home_adv = GPy.kern.Linear(input_dim=1, active_dims=[idx],
                variances=var, name="home_adv")
        # Putting it all together.
        kernel = k_teams
        if self.home_advantage:
            kernel += k_home_adv
        link = GPy.likelihoods.link_functions.RaoKupper(self.alpha)
        likelihood = GPy.likelihoods.Bernoulli(gp_link=link)
        method = expectation_propagation.EP(parallel_updates=parallel)
        n = self.featmat.shape[0]
        self.gp = GPy.core.GP(
                X=self.featmat, Y=np.ones((n, 1)), kernel=kernel,
                likelihood=likelihood, inference_method=method)

    def predict(self, game):
        vec = self.features.new_vector()
        vec[game.team_home.name] = +1.0
        vec[game.team_away.name] = -1.0
        if not game.is_neutral:
            vec["home_adv"] = +1.0
        # Second, we compute the predictive mean and variance
        Xnew = np.array([vec.as_array()])
        mean, _ = self.gp.predict(Xnew)
        prob_win = mean[0, 0]
        mean, _ = self.gp.predict(-Xnew)
        prob_lose = mean[0, 0]
        prob_draw = 1.0 - prob_win - prob_lose
        return (prob_win, prob_draw, prob_lose)
