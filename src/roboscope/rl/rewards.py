"""Training-only potential shaping; no privileged state enters the policy."""

import math

PROGRESS_MODE = "grasp_progress_v1"


def validate_reward(cfg):
    reward = cfg.get("reward", {"mode": "sparse"})
    if not isinstance(reward, dict):
        raise ValueError("reward must be a mapping")
    mode = reward.get("mode")
    if mode == "sparse":
        if set(reward) != {"mode"}:
            raise ValueError("Sparse reward accepts only mode")
        return
    if mode != PROGRESS_MODE or cfg.get("environment_backend") != "hf_libero":
        raise ValueError("Progress reward requires grasp_progress_v1 and hf_libero")
    positive = ("scale", "reach_distance_m", "transport_distance_m", "max_grasp_distance_m")
    weights = ("reach_weight", "grasp_weight", "transport_weight")
    if set(reward) != {"mode", "grasp_confirmation_steps", *positive, *weights}:
        raise ValueError("Progress reward requires an explicit, versioned parameter set")
    if type(reward["grasp_confirmation_steps"]) is not int or reward["grasp_confirmation_steps"] < 1:
        raise ValueError("reward.grasp_confirmation_steps must be a positive integer")
    for key in (*positive, *weights):
        value = reward[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid reward.{key}")
        if key in positive and value == 0:
            raise ValueError(f"reward.{key} must be positive")
    if not math.isclose(sum(reward[key] for key in weights), 1.0):
        raise ValueError("Progress potential weights must sum to 1")


def progress_enabled(cfg):
    validate_reward(cfg)
    return cfg.get("reward", {}).get("mode") == PROGRESS_MODE


def grasp_progress(
    *, eef_distance, goal_distance, finger_contact, external_contact, grasp_contact_steps, reward
):
    """A bounded potential over physical state + consecutive-contact counter.

    Any target contact with a non-finger geom blocks the pickup term, including
    a ramekin lifted together with the bowl. One finger is sufficient for rim
    grasps; contact alone is insufficient while the bowl still has support.
    This is a pickup proxy, not a guarantee of a mechanically secure grasp.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (eef_distance, goal_distance)):
        raise ValueError("Nonfinite or negative reward geometry")
    reach = 1.0 - math.tanh(eef_distance / reward["reach_distance_m"])
    candidate = bool(
        finger_contact and not external_contact and eef_distance <= reward["max_grasp_distance_m"]
    )
    pickup = float(candidate and grasp_contact_steps >= reward["grasp_confirmation_steps"])
    transport = pickup * (1.0 - math.tanh(goal_distance / reward["transport_distance_m"]))
    potential = (
        reward["reach_weight"] * reach
        + reward["grasp_weight"] * pickup
        + reward["transport_weight"] * transport
    )
    return {
        "potential": potential,
        "reach": reach,
        "pickup": pickup,
        "transport": transport,
        "pickup_candidate": candidate,
    }
