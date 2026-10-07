"""Progress semantics and discounted replay accounting, without simulator assets."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from roboscope.envs.libero_reward import LiberoGraspProgress  # noqa: E402
from roboscope.rl.collector import collect_episode  # noqa: E402
from roboscope.rl.replay import episode_transitions  # noqa: E402
from roboscope.rl.rewards import grasp_progress, validate_reward  # noqa: E402
from roboscope.workflows.config import validate_rlt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/libero_spatial/smolvla_rlt_hf_progress.json").read_text())


def test_progress_recipe_preserves_baseline_learning_parameters():
    cfg = config()
    validate_rlt(cfg)
    baseline = json.loads((ROOT / "configs/libero_spatial/smolvla_rlt_hf.json").read_text())
    assert {k: v for k, v in cfg.items() if k != "reward"} == baseline


@pytest.mark.parametrize(
    "change",
    [
        {"scale": float("nan")},
        {"scale": -1},
        {"scale": 0},
        {"mode": "typo"},
        {"reach_weight": 1},
        {"transport_distance_m": 0},
        {"unexpected": 1},
        {"grasp_confirmation_steps": 0},
    ],
)
def test_invalid_reward_config_fails_closed(change):
    cfg = config()
    cfg["reward"].update(change)
    with pytest.raises(ValueError):
        validate_reward(cfg)


def test_grasp_progress_distinguishes_support_contact_rim_grasp_and_drop():
    reward = config()["reward"]

    def phi(distance=0.05, finger=True, supported=False, goal=0.3):
        return grasp_progress(
            eef_distance=distance,
            goal_distance=goal,
            finger_contact=finger,
            external_contact=supported,
            grasp_contact_steps=3,
            reward=reward,
        )

    far, approach = phi(distance=0.3, finger=False), phi(finger=False)
    on_table, co_lifted_ramekin = phi(supported=True), phi(supported=True, goal=0.01)
    rim_grasp, over_plate = phi(), phi(goal=0.01)
    dropped = phi(distance=0.18, finger=False)
    assert far["potential"] < approach["potential"]
    assert on_table["pickup"] == co_lifted_ramekin["pickup"] == 0
    assert rim_grasp["pickup"] == 1  # does not require both finger pads
    assert on_table["potential"] < rim_grasp["potential"] < over_plate["potential"]
    assert dropped["potential"] < rim_grasp["potential"]
    assert phi(distance=0.3)["pickup"] == 0


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("length", [3, 10, 23])
def test_each_overlapping_chunk_equals_discounted_physical_step_rewards(success, length):
    cfg = config()
    gamma, scale = cfg["gamma"], cfg["reward"]["scale"]
    potential = np.random.default_rng(42).uniform(size=length + 1)
    original = potential.copy()
    absorbing_phi = potential.copy()
    absorbing_phi[-1] = 0
    shaping = scale * (gamma * absorbing_phi[1:] - absorbing_phi[:-1])
    sparse = np.zeros(length)
    sparse[-1] = success
    features = {i: (np.zeros(3), np.zeros((10, 7))) for i in range(length)}
    rows = list(
        episode_transitions(
            np.zeros((length, 7)), success, features, cfg, 0, 1, 2, True, potentials=potential
        )
    )
    np.testing.assert_array_equal(potential, original)
    for row in rows:
        start, n = row["control_step"], row["duration"]
        discounts = gamma ** np.arange(n)
        assert row["reward"] == pytest.approx(
            np.dot(discounts, (sparse + shaping)[start : start + n]), abs=1e-7
        )
        assert row["shaping_reward"] == pytest.approx(np.dot(discounts, shaping[start : start + n]), abs=1e-7)
        if start + n == length:
            assert row["next_potential"] == row["discount"] == 0
    # Terminal rollback removes the incentive to finish a timeout holding a bowl.
    assert np.dot(gamma ** np.arange(length), shaping) == pytest.approx(-scale * potential[0])


@pytest.mark.parametrize("potential", [None, [0.2], [0.2, float("nan")], [0.2, 1.01]])
def test_missing_or_invalid_potentials_cannot_silently_become_sparse_reward(potential):
    with pytest.raises(ValueError, match="potential"):
        list(
            episode_transitions(
                np.zeros((1, 7)),
                False,
                {0: (np.zeros(3), np.zeros((10, 7)))},
                config(),
                0,
                0,
                0,
                True,
                potentials=potential,
            )
        )


def test_native_contact_adapter_selects_goal_target_and_blocks_other_supports():
    # A distractor's finger contact must not count as grasping the target.
    names = {"target": 0, "plate": 1, "left": 2, "right": 3, "ramekin": 4, "distractor": 5}
    objects = {n: SimpleNamespace(root_body=n, contact_geoms=[n]) for n in ("target", "plate")}
    data = SimpleNamespace(body_xpos=np.array([[0.0, 0.0, 1.2], [0.3, 0.0, 0.8]]), ncon=0, contact=[])
    env = SimpleNamespace(
        parsed_problem={"goal_state": [["on", "target", "plate"]]},
        sim=SimpleNamespace(
            model=SimpleNamespace(geom_name2id=names.__getitem__, body_name2id=names.__getitem__), data=data
        ),
        get_object=objects.__getitem__,
        robots=[
            SimpleNamespace(
                gripper=SimpleNamespace(important_geoms={"left_finger": ["left"], "right_finger": ["right"]})
            )
        ],
    )
    adapter = LiberoGraspProgress(env, config()["reward"])

    def measure(pairs):
        data.contact = [SimpleNamespace(geom1=names[a], geom2=names[b]) for a, b in pairs]
        data.ncon = len(pairs)
        return adapter.measure([0, 0, 1.25])

    assert measure([("distractor", "left")])["pickup"] == 0
    assert measure([("left", "target")])["pickup"] == 0
    assert measure([("left", "target")])["pickup"] == 0
    assert measure([("left", "target")])["pickup"] == 1
    assert measure([("left", "target"), ("target", "ramekin")])["pickup"] == 0
    assert measure([("left", "target")])["pickup"] == 0  # contact spike after co-lift
    assert measure([("left", "target"), ("target", "plate")])["destination_contact"]
    # Support-independent: a bowl can be grasped from a cabinet or table height.
    data.body_xpos[0, 2] = 0.8
    data.contact, data.ncon = [SimpleNamespace(geom1=0, geom2=2)], 1
    assert adapter.measure([0, 0, 0.85])["pickup"] == 0
    assert adapter.measure([0, 0, 0.85])["pickup"] == 0
    assert adapter.measure([0, 0, 0.85])["pickup"] == 1
    # A new episode never inherits confirmed grasp history.
    assert LiberoGraspProgress(env, config()["reward"]).measure([0, 0, 0.85])["pickup"] == 0


def test_collector_carries_every_step_potential_without_exposing_it_to_actor():
    cfg = {**config(), "rollout_horizon": 13, "amp": False}

    class Pool:
        def __init__(self):
            self.connections = [self]

        def send(self, message):
            self.step = 0 if message[0] == "reset_training" else self.step + 1

        def receive(self, slot):
            return {
                "state": np.zeros(8, np.float32),
                "agentview_rgb": np.zeros((4, 4, 3), np.uint8),
                "eye_in_hand_rgb": np.zeros((4, 4, 3), np.uint8),
                "reward_progress": {"potential": self.step / 20, "pickup": int(self.step > 5)},
            }, False

    class Policy:
        def describe(self, batch, noise):
            assert set(batch) == {"state", "task_id", "remaining_steps", "agentview_rgb", "eye_in_hand_rgb"}
            n = len(batch["state"])
            return torch.zeros(n, 3), torch.zeros(n, 10, 7)

    rows, record = collect_episode(Pool(), Policy(), {"id": 0}, 0, cfg, torch.device("cpu"), True)
    assert rows[0]["next_potential"] == 0.5  # t=10, not last frame in the episode
    assert rows[-1]["next_potential"] == 0
    assert record["steps"] == 13 and record["pickup_proxy_steps"] == 8
    assert record["pickup_proxy_entries"] == 1
    assert record["discounted_shaping_return"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("old_mode", ["sparse", "changed_scale"])
def test_resume_rejects_replay_from_another_reward_contract(tmp_path, old_mode):
    from roboscope.trainers.smolvla_rlt import train_online

    cfg = config()
    previous = {**cfg, "reward": {**cfg["reward"], "scale": 0.2}}
    if old_mode == "sparse":
        previous.pop("reward")
    torch.save({"config": previous}, tmp_path / "last.pt")
    policy = SimpleNamespace(base=torch.nn.Linear(1, 1), token=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="changed on resume"):
        train_online(tmp_path, cfg, {}, policy, None, torch.device("cpu"), resume=True)


@pytest.mark.local
@pytest.mark.skipif(
    not os.environ.get("ROBOSCOPE_RLT_HF_REWARD_RUN"), reason="Requires HF assets and EGL binding"
)
def test_native_worker_collector_replay_and_shaped_actor_critic_update(tmp_path):
    from roboscope.envs.pool import EnvPool
    from roboscope.rl.learner import RLTAgent
    from roboscope.rl.replay import ReplayBuffer

    source = Path(os.environ["ROBOSCOPE_RLT_HF_REWARD_RUN"])
    cfg = {
        **json.loads((source / "env_config.json").read_text()),
        **config(),
        "rollout_horizon": 13,
        "amp": False,
    }
    task = json.loads((source / "manifest.json").read_text())["tasks"][0]

    class Policy:
        def describe(self, batch, noise):
            n = len(batch["state"])
            return torch.zeros(n, cfg["token_dim"] + cfg["state_dim"] + 1), torch.zeros(n, 10, 7)

    pool = EnvPool(cfg, tmp_path, 1)
    try:
        rows, record = collect_episode(pool, Policy(), task, 0, cfg, torch.device("cpu"), True)
    finally:
        pool.close()
    assert record["reward_mode"] == "grasp_progress_v1"
    assert len(rows) > 0 and rows[-1]["next_potential"] == 0
    buffer = ReplayBuffer(100)
    for row in rows:
        buffer.add(row)
    agent = RLTAgent(cfg)
    for _ in range(2):
        metrics = agent.update(buffer.sample(4, "cpu"))
        assert np.isfinite(list(metrics.values())).all()
    assert "actor_loss" in metrics
