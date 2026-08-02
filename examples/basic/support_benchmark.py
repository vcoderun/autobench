from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, TypeAdapter

from autobench import Case, RunContext

RoutingProfile = Literal["baseline", "priority"]
ROUTING_PROFILE_ADAPTER = TypeAdapter(RoutingProfile)


class Ticket(BaseModel):
    subject: str
    customer_tier: Literal["standard", "enterprise"]


class RoutingDecision(BaseModel):
    queue: Literal["account", "billing", "incident", "technical"]
    handled: bool = True
    matched_rule: str


def run(ctx: RunContext, case: Case) -> RoutingDecision:
    ticket = Ticket.model_validate(case.input)
    profile = ROUTING_PROFILE_ADAPTER.validate_python(ctx.factor("routing_profile"))
    with ctx.span(
        "route_ticket",
        kind="workflow",
        input=ticket.model_dump(),
        attributes={"profile": profile},
    ) as span:
        decision = _route(ticket, profile=profile)
        span.set_output(decision.model_dump())
        span.artifact(
            "routing_decision",
            decision.model_dump(),
            media_type="application/yaml",
        )
    return decision


def _route(ticket: Ticket, *, profile: RoutingProfile) -> RoutingDecision:
    subject = ticket.subject.casefold()
    if profile == "priority" and ticket.customer_tier == "enterprise" and "outage" in subject:
        return RoutingDecision(queue="incident", matched_rule="enterprise_outage")
    if "refund" in subject or "charge" in subject:
        return RoutingDecision(queue="billing", matched_rule="billing_keyword")
    if "sign in" in subject or "password" in subject:
        return RoutingDecision(queue="account", matched_rule="account_keyword")
    return RoutingDecision(queue="technical", matched_rule="fallback")


__all__ = ("run",)
