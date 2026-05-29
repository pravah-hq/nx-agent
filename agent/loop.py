from __future__ import annotations

from agent.environment import World
from agent.observations import print_observation
from agent.policy import Policy, apply_consideration
from agent.types import Action, ActionType, AgentState, StepRecord


class AgentLoop:
    def __init__(self, world: World, policy: Policy | None = None) -> None:
        self.world = world
        self.policy = policy

    def step(self, state: AgentState, action: Action | None = None) -> tuple[AgentState, StepRecord]:
        poles_in_view = self.world.poles_in_view(state)
        state = (
            apply_consideration(state, self.world, self.policy, poles_in_view)
            if self.policy
            else state
        )

        if action is None:
            if self.policy is None:
                raise ValueError("No action provided and no policy configured.")
            action = self.policy.choose(self.world, state, poles_in_view)

        before = state.copy()
        after, message = self.world.apply_action(state, action)
        record_step = getattr(self.policy, "record_step", None)
        if callable(record_step):
            record_step(before, action, after)
        record = StepRecord(
            step=0,
            action=action,
            state_before=before,
            state_after=after,
            poles_in_view=poles_in_view,
            message=message,
        )
        return after, record

    def run(
        self,
        state: AgentState,
        *,
        max_steps: int = 500,
        verbose: bool = True,
        json_obs: bool = False,
    ) -> tuple[AgentState, list[StepRecord]]:
        if self.policy is None:
            raise ValueError("Autonomous run requires a policy.")

        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

        history: list[StepRecord] = []
        current = state

        for step_index in range(1, max_steps + 1):
            poles_in_view = self.world.poles_in_view(current)
            if verbose:
                print(f"\n--- step {step_index} ---")
                print_observation(self.world, current, poles_in_view, as_json=json_obs)

            current = apply_consideration(current, self.world, self.policy, poles_in_view)
            action = self.policy.choose(self.world, current, poles_in_view)
            before = current.copy()
            current, message = self.world.apply_action(current, action)
            record_step = getattr(self.policy, "record_step", None)
            if callable(record_step):
                record_step(before, action, current)

            record = StepRecord(
                step=step_index,
                action=action,
                state_before=before,
                state_after=current.copy(),
                poles_in_view=poles_in_view,
                message=message,
            )
            history.append(record)

            if verbose:
                print(f"action: {action.type.value} -> {message}")

            if action.type == ActionType.CLASSIFY_OR_STOP and action.stop_after:
                break
            if self.world.is_task_complete(current):
                break

        return current, history
