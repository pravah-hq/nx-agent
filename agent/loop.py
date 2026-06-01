"""
Main control loop per step:

  apply_consideration (pick target pole)
  -> observe / resolve_pole_in_clear_view (VLM: is target in clear view?)
  -> choose (classify if clear, else navigate)
  -> apply_action

AgentLoop.run() is used by `python -m agent run`; step() for single-step debugging.
"""

from __future__ import annotations

from agent.clear_view import geometric_pole_in_clear_view
from agent.environment import World
from agent.observations import print_observation, print_vlm_step_responses
from agent.vlm_policy import VlmPolicy
from agent.policy import Policy, apply_consideration
from agent.types import Action, ActionType, AgentState, StepRecord
def resolve_pole_in_clear_view(
    world: World,
    policy: Policy | None,
    state: AgentState,
) -> bool:
    """
    Before choose(): is the target pole unambiguously identifiable?

    VlmPolicy.observe(): VLM clear-view check (after target is chosen).
    """
    if policy is None:
        return False
    if isinstance(policy, VlmPolicy):
        return policy.observe(world, state)
    return geometric_pole_in_clear_view(world, state)


class AgentLoop:
    def __init__(self, world: World, policy: Policy | None = None) -> None:
        self.world = world
        self.policy = policy

    def step(self, state: AgentState, action: Action | None = None) -> tuple[AgentState, StepRecord]:
        """One step; pass action to override policy (manual testing)."""
        if isinstance(self.policy, VlmPolicy):
            self.policy.begin_agent_step()
        if self.policy:
            state = apply_consideration(state, self.world, self.policy)
        pole_in_clear_view = resolve_pole_in_clear_view(self.world, self.policy, state)

        if action is None:
            if self.policy is None:
                raise ValueError("No action provided and no policy configured.")
            action = self.policy.choose(self.world, state, pole_in_clear_view)
            print_vlm_step_responses(self.policy)

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
            pole_in_clear_view=pole_in_clear_view,
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
        """Autonomous episode until classify+stop, max_steps, or all poles classified."""
        if self.policy is None:
            raise ValueError("Autonomous run requires a policy.")

        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

        history: list[StepRecord] = []
        current = state

        for step_index in range(1, max_steps + 1):
            if isinstance(self.policy, VlmPolicy):
                self.policy.begin_agent_step()
            current = apply_consideration(current, self.world, self.policy)
            pole_in_clear_view = resolve_pole_in_clear_view(
                self.world, self.policy, current
            )
            if verbose:
                print(f"\n--- step {step_index} ---")
                print_observation(
                    self.world,
                    current,
                    pole_in_clear_view=pole_in_clear_view,
                    as_json=json_obs,
                    policy=self.policy,
                )

            action = self.policy.choose(self.world, current, pole_in_clear_view)
            if verbose:
                print_vlm_step_responses(self.policy)
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
                pole_in_clear_view=pole_in_clear_view,
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
