# standard
import warnings
from dataclasses import dataclass

# external
import pandas as pd
import numpy as np


@dataclass(order=True)
class State:
    name: str
    char: str
    reward: float = None


@dataclass(order=True)
class Step:
    state: State
    reward: float
    done: bool
    info: dict = None

    def __repr__(self):
        action = repr((self.info or {}).get("action", ""))
        return f"{self.__class__.__name__}(action={action},state={self.state},reward={self.reward})"


# ############################################################################## #
# ############################################################################## #


class Game:
    _blank_state = State(
        name="blank",
        char=" ",
        reward=0,
    )
    _outofbounds_state = State(
        name="outofbounds",
        char=" ",
        reward=0,
    )

    def __init__(self, rows=5, cols=5, random_seed=55):
        self.rows = rows
        self.cols = cols
        self.random_seed = random_seed

        self._rng = np.random.default_rng(random_seed)

        self.items = {}
        self._add_items(1, name="start", char="S", reward=0)
        self.user_prev = None
        self.user_loc = list(self.items.keys())[0]
        self.user_score = 50
        self._add_items(1, name="end", char="E", reward=100)
        self._add_items(2, name="bomb", char="B", reward=-10)
        self._add_items(2, name="powerup", char="+", reward=10)

    def copy(self):
        return self.__class__(
            rows=self.rows,
            cols=self.cols,
            random_seed=self.random_seed,
        )

    def _random_loc(self, unique=True):
        cells = self.rows * self.cols
        for _ in range(cells * 2):
            row = self._rng.integers(self.rows)
            col = self._rng.integers(self.cols)
            loc = (row, col)
            if loc in self.items and unique:
                continue
            return loc
        raise ValueError("")

    def _add_items(self, count, **state):
        for _ in range(count):
            loc = self._random_loc()
            self.items[loc] = State(**state)

    def _formatted(self, delim="|"):
        lines = []
        for row in range(self.rows):
            elements = []
            for col in range(self.cols):
                loc = (row, col)
                item_char = self.items.get(loc, self._blank_state).char
                if loc == self.user_prev:
                    user_char = "-"
                elif loc == self.user_loc:
                    user_char = "o"
                else:
                    user_char = " "
                elements.append(item_char + user_char)
            line = delim + delim.join(elements) + delim
            lines.append(line)
        return "\n".join(lines)

    def _next_user_loc(self, action):
        row, col = self.user_loc
        if action == "up":
            row -= 1
        elif action == "down":
            row += 1
        elif action == "left":
            col -= 1
        elif action == "right":
            col += 1
        else:
            raise ValueError(action)

        next_row = max(0, min(self.rows - 1, row))
        next_col = max(0, min(self.cols - 1, col))
        outofbounds = (row != next_row) or (col != next_col)
        return (next_row, next_col), outofbounds

    def get_state(self):
        return self.items.get(self.user_loc, self._blank_state).name

    def step(self, action):
        loc = self.user_loc
        next_loc, outofbounds = self._next_user_loc(action)
        next_state = self.items.get(next_loc, self._blank_state)

        if outofbounds:
            state_name = self._outofbounds_state.name
            reward = 0
            user_prev = None
            user_loc = loc
        else:
            state_name = next_state.name
            reward = next_state.reward
            user_prev = loc
            user_loc = next_loc

        self.user_score += reward
        self.user_prev = user_prev
        self.user_loc = user_loc

        done = (self.user_score < 0) | (next_state.name == "end")
        return Step(
            state=state_name,
            reward=reward,
            done=done,
            info=dict(
                action=action,
            ),
        )

    def render(self, mode="human"):
        print(self._formatted())

    def __repr__(self):
        return self._formatted()

    @property
    def action_space(self):
        return (
            "left",
            "right",
            "up",
            "down",
        )

    @property
    def state_space(self):
        return set([self._blank_state.name, self._outofbounds_state.name] + [
            item.name for item in self.items.values()
        ])


# ############################################################################## #
# ############################################################################## #


class Agent:
    def __init__(self, state_space, action_space, random_seed=123, learning=True):
        self.qtable = pd.DataFrame(
            0,
            index=action_space,
            columns=state_space,
        )
        self.gamma = 0.9
        self.alpha = 1.0
        self.random_seed = random_seed
        self.learning = learning
        self.rng = np.random.default_rng(random_seed)

    def update(self, state, next_state, action, reward, info=None):
        if not self.learning:
            return
        # Q(S, A) = Q(S, A) + alpha[R + gamma * max_a Q(S', A) - Q(S, A)]
        current_reward = self.qtable.loc[action, state]
        self.qtable.loc[action, state] = current_reward + (
            self.alpha * (
                reward
                + self.gamma * (
                    self.qtable[next_state].max()
                    - current_reward
                )
            )
        )

    def get_action(self, state):
        if self.learning:
            action = self.rng.choice(self.qtable.index)
        else:
            action = self.qtable[state].idxmax()
        return action


# ############################################################################## #
# ############################################################################## #


@dataclass
class TrainingStep:
    step: Step
    action: str
    episode: int = 0
    episode_step: int = 0


def train_agent(agent, env, num_episodes=10, num_steps_max=1000):
    agent.learning = True
    for episode in range(num_episodes):
        _env = env.copy()
        for episode_step in range(num_steps_max):
            state = _env.get_state()
            action = agent.get_action(state)
            step = _env.step(action)
            agent.update(
                state=state,
                next_state=step.state,
                action=action,
                reward=step.reward,
                info=step.info
            )
            yield TrainingStep(
                step=step,
                action=action,
                episode=episode,
                episode_step=episode_step,
            )
            if step.done:
                break
        else:
            raise ValueError(num_steps_max)


def test_agent(agent, env, num_steps_max=100):
    agent.learning = False
    for _ in range(num_steps_max):
        state = env.get_state()
        action = agent.get_action(state)
        step = env.step(action)
        if step.done:
            break
        yield step
    else:
        warnings.warn(f"Did not complete after {num_steps_max} steps")
