from typing import Any

from app.agent.graph import CampusAffairsGraph
from app.agent.state import AgentState
from app.core.security import RequestContext


def build_langgraph_app(graph: CampusAffairsGraph, context: RequestContext) -> Any:
    """Build a LangGraph StateGraph when the optional dependency is installed."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "langgraph is not installed. Install requirements.txt in production "
            "or keep using CampusAffairsGraph for local development."
        ) from exc

    async def input_guard(state: AgentState) -> AgentState:
        graph._input_guard(state)
        return state

    async def classify_intent(state: AgentState) -> AgentState:
        graph._classify_intent(state)
        return state

    async def business_guard(state: AgentState) -> AgentState:
        graph._business_guard(state)
        return state

    async def retrieve_policy(state: AgentState) -> AgentState:
        graph._retrieve_policy(state, context)
        return state

    async def collect_slots(state: AgentState) -> AgentState:
        graph._collect_slots(state)
        return state

    async def generate_answer(state: AgentState) -> AgentState:
        await graph._generate_answer(state)
        return state

    async def output_guard(state: AgentState) -> AgentState:
        graph._output_guard(state)
        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("input_guard", input_guard)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("business_guard", business_guard)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("collect_slots", collect_slots)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("output_guard", output_guard)

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        lambda state: END if state.blocked_reason else "classify_intent",
    )
    workflow.add_edge("classify_intent", "business_guard")
    workflow.add_conditional_edges(
        "business_guard",
        lambda state: END if state.blocked_reason else "retrieve_policy",
    )
    workflow.add_edge("retrieve_policy", "collect_slots")
    workflow.add_edge("collect_slots", "generate_answer")
    workflow.add_edge("generate_answer", "output_guard")
    workflow.add_edge("output_guard", END)
    return workflow.compile()

