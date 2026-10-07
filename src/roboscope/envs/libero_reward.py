"""Privileged HF LIBERO geometry used only to construct training rewards."""

import numpy as np

from roboscope.rl.rewards import grasp_progress


class LiberoGraspProgress:
    def __init__(self, native_env, reward):
        # Construct AFTER every hard reset: MuJoCo geom IDs can be rebuilt.
        self.env, self.reward = native_env, reward
        self.grasp_contact_steps = 0
        goals = native_env.parsed_problem["goal_state"]
        on_goals = [goal for goal in goals if len(goal) == 3 and goal[0].lower() == "on"]
        if len(on_goals) != 1:
            raise ValueError("grasp_progress_v1 requires exactly one On(object, destination) goal")
        _, self.target, self.destination = on_goals[0]
        model = native_env.sim.model

        def geoms(names):
            return {model.geom_name2id(name) for name in names}

        target = native_env.get_object(self.target)
        destination = native_env.get_object(self.destination)
        self.target_body = model.body_name2id(target.root_body)
        self.destination_body = model.body_name2id(destination.root_body)
        self.target_geoms = geoms(target.contact_geoms)
        self.destination_geoms = geoms(destination.contact_geoms)
        fingers = native_env.robots[0].gripper.important_geoms
        self.finger_geoms = geoms([*fingers["left_finger"], *fingers["right_finger"]])
        if not self.target_geoms or not self.finger_geoms:
            raise ValueError("Missing target or finger collision geometry")

    def measure(self, eef_pos):
        data = self.env.sim.data
        target = data.body_xpos[self.target_body]
        destination = data.body_xpos[self.destination_body]
        contacts = set()
        for contact in data.contact[: data.ncon]:
            # Count native contact pairs (including near-contact constraints),
            # conservatively treating all non-finger objects as support/obstruction.
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.target_geoms and b not in self.target_geoms:
                contacts.add(b)
            if b in self.target_geoms and a not in self.target_geoms:
                contacts.add(a)
        distance = float(np.linalg.norm(target - np.asarray(eef_pos)))
        goal_distance = float(np.linalg.norm(target[:2] - destination[:2]))
        finger_contact = bool(contacts & self.finger_geoms)
        external_contact = bool(contacts - self.finger_geoms)
        candidate = (
            finger_contact and not external_contact and distance <= self.reward["max_grasp_distance_m"]
        )
        # Reject isolated solver/contact gaps (e.g. bowl still riding a ramekin).
        # Clear immediately on loss; never retain a maximum-ever pickup milestone.
        self.grasp_contact_steps = (
            min(self.grasp_contact_steps + 1, self.reward["grasp_confirmation_steps"]) if candidate else 0
        )
        return {
            **grasp_progress(
                eef_distance=distance,
                goal_distance=goal_distance,
                finger_contact=finger_contact,
                external_contact=external_contact,
                grasp_contact_steps=self.grasp_contact_steps,
                reward=self.reward,
            ),
            "finger_contact": finger_contact,
            "grasp_contact_steps": self.grasp_contact_steps,
            "external_contact": external_contact,
            "destination_contact": bool(contacts & self.destination_geoms),
            "eef_distance_m": distance,
            "goal_distance_m": goal_distance,
        }
