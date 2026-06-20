"""Rescue-window state machine for Second Bell."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RescueState:
    state: str
    allowed_actions: List[str]
    blocked_actions: List[str]
    requires_human_approval: bool
    explanation: str


def classify_rescue_state(row: Dict) -> RescueState:
    sealed = bool(row.get("sealed_or_whole", 0))
    cold = bool(row.get("cold_chain_required", 0))
    monitor = bool(row.get("share_table_monitor", 0))
    cooler = int(row.get("cooler_capacity", 0))
    predicted_returns = float(row.get("predicted_return_q50", 0))

    if predicted_returns <= 3:
        return RescueState(
            state="low_surplus_watch",
            allowed_actions=["monitor only", "log actual returns"],
            blocked_actions=["production cut", "after-school route"],
            requires_human_approval=False,
            explanation="Predicted recoverable surplus is too small for an operational intervention.",
        )

    if not sealed:
        return RescueState(
            state="not_recoverable_for_share_table",
            allowed_actions=["batch staging", "compost planning", "menu adjustment review"],
            blocked_actions=["share-table reuse", "after-school redistribution"],
            requires_human_approval=True,
            explanation="Prepared or opened items are not routed through the share-table workflow in this MVP.",
        )

    if cold and (not monitor or cooler <= 0):
        return RescueState(
            state="cold_chain_blocked",
            allowed_actions=["two-wave staging", "manager review", "log as cold-chain miss"],
            blocked_actions=["unmonitored share table", "after-school cold route"],
            requires_human_approval=True,
            explanation="Cold items need monitored cold storage before reuse can be considered.",
        )

    if sealed and monitor:
        return RescueState(
            state="share_table_eligible",
            allowed_actions=["deploy share-table cooler", "after-school demand match", "line placement adjustment", "two-wave staging"],
            blocked_actions=["automatic safety approval"],
            requires_human_approval=True,
            explanation="The item is sealed or whole and a monitor is available, so the AI may recommend a rescue action but cannot approve safety.",
        )

    return RescueState(
        state="whole_item_monitor_needed",
        allowed_actions=["line placement adjustment", "manual audit"],
        blocked_actions=["automatic redistribution"],
        requires_human_approval=True,
        explanation="The item may be recoverable, but the school needs a monitored process before reuse.",
    )
