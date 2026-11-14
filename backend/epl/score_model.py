"""Time-decayed Poisson and Dixon–Coles score models, using results only."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import poisson


class ScoreModel:
    """Jointly fit centered attack/defence, home advantage and low-score dependence."""

    classes_ = np.array(["A", "D", "H"])

    def __init__(self, half_life=365, corrected=True):
        if not np.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life
        self.corrected = corrected

    @staticmethod
    def _rho(parameters, attack, defence):
        """A differentiable bounded rho keeps every team's low-score cells positive."""
        count = len(attack)
        intercept, home = parameters[2 * count:2 * count + 2]
        attack_max, defence_max = attack.argmax(), defence.argmax()
        strength_max = (attack + defence).argmax()
        max_rate = np.exp(intercept + max(home, 0) + attack[attack_max] + defence[defence_max])
        max_product = np.exp(2 * intercept + home + 2 * (attack + defence)[strength_max])
        lower = -.95 / max_rate
        upper = min(.95, .95 / max_product)
        share = expit(parameters[-1])
        rho = lower + (upper - lower) * share
        gradient = np.zeros_like(parameters)
        lower_gradient = np.zeros_like(parameters)
        lower_gradient[:count] = lower / count
        lower_gradient[attack_max] -= lower
        lower_gradient[count:2 * count] = lower / count
        lower_gradient[count + defence_max] -= lower
        lower_gradient[2 * count] = -lower
        lower_gradient[2 * count + 1] = -lower if home > 0 else 0
        upper_gradient = np.zeros_like(parameters)
        if max_product > 1:
            upper_gradient[:count] = 2 * upper / count
            upper_gradient[strength_max] -= 2 * upper
            upper_gradient[count:2 * count] = 2 * upper / count
            upper_gradient[count + strength_max] -= 2 * upper
            upper_gradient[2 * count] = -2 * upper
            upper_gradient[2 * count + 1] = -upper
        gradient += (1 - share) * lower_gradient + share * upper_gradient
        gradient[-1] = (upper - lower) * share * (1 - share)
        return rho, gradient

    def fit(self, matches, cutoff=None):
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        if not required.issubset(matches.columns):
            raise ValueError("Score fitting requires dates, teams and full-time goals")
        frame = matches.copy()
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        reference = pd.Timestamp(cutoff) if cutoff is not None else frame.date.max() + pd.Timedelta(days=1)
        frame = frame[frame.date < reference]
        if "season" in frame:
            frame = frame[frame.season.isin(sorted(frame.season.unique())[-5:])]
        else:
            frame = frame[frame.date >= reference - pd.DateOffset(years=5)]
        if frame.empty or frame.date.isna().any():
            raise ValueError("No dated matches before the training cutoff")
        goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=float)
        if not np.isfinite(goals).all() or (goals < 0).any() or (goals != np.floor(goals)).any():
            raise ValueError("Goals must be finite nonnegative integers")
        if (frame.home_team == frame.away_team).any() or frame[["home_team", "away_team"]].isna().any().any():
            raise ValueError("Invalid teams")
        teams = sorted(set(frame.home_team) | set(frame.away_team))
        self.teams_ = {team: index for index, team in enumerate(teams)}
        count = len(teams)
        home_index = frame.home_team.map(self.teams_).to_numpy()
        away_index = frame.away_team.map(self.teams_).to_numpy()
        hg, ag = goals.T
        weights = np.exp2(-(reference - frame.date).dt.total_seconds().to_numpy() / (86400 * self.half_life))
        weights /= weights.sum()
        initial = np.zeros(2 * count + 2 + int(self.corrected))
        initial[2 * count] = np.clip(np.log(max(np.average(ag, weights=weights), .15)), -2, 2)
        initial[2 * count + 1] = np.clip(np.log(max(np.average(hg, weights=weights), .15)) - initial[2 * count], -1, 1)
        if self.corrected:
            # Start at independence despite the asymmetric admissible rho interval.
            initial[-1] = -8
            lower, _ = self._rho(initial, initial[:count], initial[count:2 * count])
            initial[-1] = 8
            upper, _ = self._rho(initial, initial[:count], initial[count:2 * count])
            share = np.clip(-lower / (upper - lower), 1e-4, 1 - 1e-4)
            initial[-1] = np.log(share / (1 - share))
        constants = gammaln(hg + 1) + gammaln(ag + 1)
        zero_zero, zero_one = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1)
        one_zero, one_one = (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

        def objective(parameters):
            attack = parameters[:count] - parameters[:count].mean()
            defence = parameters[count:2 * count] - parameters[count:2 * count].mean()
            intercept, home = parameters[2 * count:2 * count + 2]
            log_home = intercept + home + attack[home_index] + defence[away_index]
            log_away = intercept + attack[away_index] + defence[home_index]
            lam, mu = np.exp(log_home), np.exp(log_away)
            log_tau = np.zeros(len(frame))
            derivative_home = np.zeros(len(frame))
            derivative_away = np.zeros(len(frame))
            derivative_rho = np.zeros(len(frame))
            if self.corrected:
                rho, rho_gradient = self._rho(parameters, attack, defence)
                tau = np.ones(len(frame))
                tau[zero_zero] = 1 - lam[zero_zero] * mu[zero_zero] * rho
                tau[zero_one] = 1 + lam[zero_one] * rho
                tau[one_zero] = 1 + mu[one_zero] * rho
                tau[one_one] = 1 - rho
                if (tau <= 0).any():
                    raise ValueError("Invalid Dixon–Coles likelihood correction")
                log_tau = np.log(tau)
                derivative_home[zero_zero] = -lam[zero_zero] * mu[zero_zero] * rho / tau[zero_zero]
                derivative_home[zero_one] = lam[zero_one] * rho / tau[zero_one]
                derivative_away[zero_zero] = derivative_home[zero_zero]
                derivative_away[one_zero] = mu[one_zero] * rho / tau[one_zero]
                derivative_rho[zero_zero] = -lam[zero_zero] * mu[zero_zero] / tau[zero_zero]
                derivative_rho[zero_one] = lam[zero_one] / tau[zero_one]
                derivative_rho[one_zero] = mu[one_zero] / tau[one_zero]
                derivative_rho[one_one] = -1 / tau[one_one]
            loss = np.sum(weights * (lam + mu - hg * log_home - ag * log_away + constants - log_tau))
            # Penalize strengths, not identities; the centered parameters have a unique interpretation.
            loss += .01 * (np.mean(attack ** 2) + np.mean(defence ** 2))
            home_gradient = weights * (lam - hg - derivative_home)
            away_gradient = weights * (mu - ag - derivative_away)
            gradient = np.zeros_like(parameters)
            gradient[:count] = np.bincount(home_index, home_gradient, minlength=count) + np.bincount(away_index, away_gradient, minlength=count)
            gradient[count:2 * count] = np.bincount(away_index, home_gradient, minlength=count) + np.bincount(home_index, away_gradient, minlength=count)
            gradient[:count] -= gradient[:count].mean()
            gradient[count:2 * count] -= gradient[count:2 * count].mean()
            gradient[:count] += .02 * attack / count
            gradient[count:2 * count] += .02 * defence / count
            gradient[2 * count] = np.sum(home_gradient + away_gradient)
            gradient[2 * count + 1] = np.sum(home_gradient)
            if self.corrected:
                gradient += -np.sum(weights * derivative_rho) * rho_gradient
            return float(loss), gradient

        bounds = [(-1.5, 1.5)] * (2 * count) + [(-2, 2), (-1, 1)]
        if self.corrected:
            bounds += [(-8, 8)]
        result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6})
        if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError(f"Score model optimization failed: {result.message}")
        self.attack_ = result.x[:count] - result.x[:count].mean()
        self.defence_ = result.x[count:2 * count] - result.x[count:2 * count].mean()
        self.intercept_, self.home_advantage_ = result.x[2 * count:2 * count + 2]
        self.rho_ = self._rho(result.x, self.attack_, self.defence_)[0] if self.corrected else 0.0
        self.training_cutoff_ = str(frame.date.max().date())
        self.training_seasons_ = sorted(frame.season.unique().tolist()) if "season" in frame else []
        self.n_matches_ = len(frame)
        self.optimization_iterations_ = int(result.nit)
        return self

    def predict_proba(self, fixtures):
        if not hasattr(self, "attack_"):
            raise ValueError("Score model is not fitted")
        output = []
        for fixture in fixtures.itertuples():
            if not isinstance(fixture.home_team, str) or not isinstance(fixture.away_team, str) or fixture.home_team == fixture.away_team:
                raise ValueError("Choose two valid different teams")
            home, away = self.teams_.get(fixture.home_team), self.teams_.get(fixture.away_team)
            home_attack = self.attack_[home] if home is not None else 0.0
            home_defence = self.defence_[home] if home is not None else 0.0
            away_attack = self.attack_[away] if away is not None else 0.0
            away_defence = self.defence_[away] if away is not None else 0.0
            lam = np.exp(self.intercept_ + self.home_advantage_ + home_attack + away_defence)
            mu = np.exp(self.intercept_ + away_attack + home_defence)
            limit = max(1, int(poisson.ppf(1 - 1e-10, max(lam, mu))))
            goals = np.arange(limit + 1)
            grid = poisson.pmf(goals, lam)[:, None] * poisson.pmf(goals, mu)[None, :]
            correction = np.array([[1 - lam * mu * self.rho_, 1 + lam * self.rho_],
                                   [1 + mu * self.rho_, 1 - self.rho_]])
            if not np.isfinite(correction).all() or (correction <= 0).any():
                raise ValueError("Invalid Dixon–Coles prediction correction")
            grid[:2, :2] *= correction
            retained = grid.sum()
            if not np.isfinite(retained) or abs(retained - 1) > 1e-8:
                raise ValueError("Score grid omitted excessive probability")
            output.append([np.triu(grid, 1).sum() / retained, np.trace(grid) / retained,
                           np.tril(grid, -1).sum() / retained])
        return np.array(output).reshape(-1, 3)
