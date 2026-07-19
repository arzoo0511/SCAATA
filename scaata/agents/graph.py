"""LangGraph StateGraph wiring the inner loop's real feedback edge:

    scraper -> normalizer -> regime_classifier -> meta_selector -> critique
                                                        ^                |
                                                        |__(continue)____|
                                                                |
                                                             (done) -> END

This is the concrete realization of the paper's dual-loop diagram's
Critique -> Meta-Selector feedback edge, bounded by `MAX_GRAPH_ITERATIONS`
(LLM/training call cost) or early convergence (see `critique_node`).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from scaata.agents.nodes.critique_node import critique_node
from scaata.agents.nodes.meta_selector_node import meta_selector_node
from scaata.agents.nodes.normalizer_node import normalizer_node
from scaata.agents.nodes.regime_classifier_node import regime_classifier_node
from scaata.agents.nodes.scraper_node import scraper_node
from scaata.agents.state import AgentState
from scaata.config import MAX_GRAPH_ITERATIONS


def _should_continue(state: AgentState) -> str:
    if state.get("converged") or state.get("iteration", 0) >= state.get("max_iterations", MAX_GRAPH_ITERATIONS):
        return "done"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("normalizer", normalizer_node)
    graph.add_node("regime_classifier", regime_classifier_node)
    graph.add_node("meta_selector", meta_selector_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("scraper")
    graph.add_edge("scraper", "normalizer")
    graph.add_edge("normalizer", "regime_classifier")
    graph.add_edge("regime_classifier", "meta_selector")
    graph.add_edge("meta_selector", "critique")
    graph.add_conditional_edges("critique", _should_continue, {"continue": "meta_selector", "done": END})

    return graph.compile()
