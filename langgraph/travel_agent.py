"""
Travel Planning Multi-Agent
=======================================================
"""

import os
import sys
import json
import warnings
from typing import Literal, Annotated, Sequence, TypedDict



# Force UTF-8 so prints involving LLM output or currency symbols never blow up.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()


# ─────────────────────────── helpers ───────────────────────────

def banner(msg: str, char: str = "-") -> None:
    line = char * 72
    print(f"\n{line}\n{msg}\n{line}")


_tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ──────────────────────────── tools ────────────────────────────
# Each tool is a single, well-scoped capability. The @tool decorator
# turns the docstring + type hints into a JSON schema the LLM can read.

@tool
def get_weather(city: str, when: str = "current") -> str:
    """Get weather for a city.

    Args:
        city: City name, e.g. "Pune".
        when: "current" for live forecast, or a month name like "June" for seasonal context.
    """
    print(f"   [TOOL]  get_weather(city='{city}', when='{when}')")
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        data = r.json()
        current = data["current_condition"][0]
        result = {
            "city": city,
            "asked_for": when,
            "current": {
                "temp_C": current["temp_C"],
                "condition": current["weatherDesc"][0]["value"],
                "humidity": current["humidity"],
            },
            "next_3_days": [
                {
                    "date": d["date"],
                    "max_C": d["maxtempC"],
                    "min_C": d["mintempC"],
                    "condition": d["hourly"][4]["weatherDesc"][0]["value"],
                }
                for d in data["weather"][:3]
            ],
            "note": (
                f"Live data shown. For travel in {when}, reason about the typical climate "
                f"of {city} in that month (e.g. monsoon, summer, winter)."
            ),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"weather lookup failed: {e}"


@tool
def search_hotels(city: str, nights: int, budget_inr: int = 5000) -> str:
    """Find hotels in a city for N nights within a per-night INR budget.

    Args:
        city: Destination city.
        nights: Number of nights to stay.
        budget_inr: Max per-night budget in INR (default 5000).
    """
    print(f"   [TOOL]  search_hotels(city='{city}', nights={nights}, budget_inr={budget_inr})")
    # Mock catalogue — swap with Booking.com / Amadeus / EaseMyTrip in production.
    catalogue = {
        "Pune": [
            {"name": "The Westin Pune Koregaon Park", "price_per_night": 8500, "rating": 4.6, "area": "Koregaon Park"},
            {"name": "Hyatt Pune",                    "price_per_night": 7200, "rating": 4.5, "area": "Kalyani Nagar"},
            {"name": "Hotel Sagar Plaza",             "price_per_night": 4200, "rating": 4.1, "area": "Pune Station"},
            {"name": "Lemon Tree Hotel",              "price_per_night": 3800, "rating": 4.0, "area": "Hinjewadi"},
            {"name": "Treebo Trend Crystal Inn",      "price_per_night": 2400, "rating": 3.8, "area": "Shivajinagar"},
        ],
    }
    options = catalogue.get(city, [
        {"name": f"{city} Grand Hotel",  "price_per_night": 5500, "rating": 4.3, "area": "City Centre"},
        {"name": f"{city} Comfort Inn",  "price_per_night": 3200, "rating": 4.0, "area": "Downtown"},
        {"name": f"{city} Budget Stay",  "price_per_night": 1800, "rating": 3.6, "area": "Suburbs"},
    ])
    matches = [h for h in options if h["price_per_night"] <= budget_inr]
    if not matches:
        matches = sorted(options, key=lambda h: h["price_per_night"])[:2]
    for h in matches:
        h["total_cost_inr"] = h["price_per_night"] * nights
    return json.dumps(
        {"city": city, "nights": nights, "budget_inr": budget_inr, "options": matches},
        indent=2,
    )


@tool
def find_attractions(city: str, interests: str = "popular tourist places") -> str:
    """Search for top attractions in a city.

    Args:
        city: Destination city.
        interests: Kind of attractions, e.g. "historic places", "food", "nature".
    """
    print(f"   [TOOL]  find_attractions(city='{city}', interests='{interests}')")
    try:
        results = _tavily.search(
            query=f"top {interests} to visit in {city} for a 2-day trip",
            max_results=5,
            search_depth="basic",
        )
        summary = [
            {"title": r["title"], "snippet": r["content"][:280], "url": r["url"]}
            for r in results.get("results", [])
        ]
        return json.dumps({"city": city, "interests": interests, "results": summary}, indent=2)
    except Exception as e:
        return f"attractions search failed: {e}"


@tool
def plan_route(origin: str, destination: str, mode: str = "road") -> str:
    """Plan a journey between two cities. mode is 'road', 'rail', or 'air'.

    Args:
        origin: Starting city.
        destination: Ending city.
        mode: Travel mode — 'road', 'rail', or 'air'.
    """
    print(f"   [TOOL]  plan_route(origin='{origin}', destination='{destination}', mode='{mode}')")
    try:
        results = _tavily.search(
            query=f"distance and travel time from {origin} to {destination} by {mode}, best route and tips",
            max_results=4,
            search_depth="basic",
        )
        summary = [
            {"title": r["title"], "snippet": r["content"][:280], "url": r["url"]}
            for r in results.get("results", [])
        ]
        return json.dumps(
            {"origin": origin, "destination": destination, "mode": mode, "route_info": summary},
            indent=2,
        )
    except Exception as e:
        return f"route planning failed: {e}"


# ───────────────────────── specialist agents ────────────────────────
# Each specialist is a tiny ReAct agent that owns exactly one tool.
# Keeping them single-tool makes the reasoning loop short and easy to demo.

llm        = ChatOpenAI(model="gpt-4o", temperature=0.2)   # used by router + specialists
synth_llm  = ChatOpenAI(model="gpt-4o", temperature=0.5)   # used by final synthesizer

weather_agent = create_react_agent(
    llm, tools=[get_weather],
    prompt=(
        "You are a weather specialist. Call get_weather, then summarise in 2-3 sentences "
        "what the climate will be like during the user's travel window."
    ),
)

hotel_agent = create_react_agent(
    llm, tools=[search_hotels],
    prompt=(
        "You are a hotel-booking specialist. Call search_hotels with sensible defaults "
        "(budget ≈ ₹5000/night unless told otherwise). Return the top 2-3 options with "
        "name, area, per-night price, and total cost."
    ),
)

attractions_agent = create_react_agent(
    llm, tools=[find_attractions],
    prompt=(
        "You are a local-attractions specialist. Use find_attractions, then recommend "
        "4-6 must-see places that fit the trip duration. Cite source URLs."
    ),
)

transport_agent = create_react_agent(
    llm, tools=[plan_route],
    prompt=(
        "You are a transport planner. Use plan_route to research the journey, then summarise "
        "distance, estimated travel time, and any practical tips (rest stops, tolls, alt routes)."
    ),
)


# ────────────────────────── supervisor ──────────────────────────

AGENTS = ["weather", "hotels", "attractions", "transport"]


class Route(BaseModel):
    """Routing decision produced by the supervisor each turn."""
    next: Literal["weather", "hotels", "attractions", "transport", "FINISH"] = Field(
        description="Which specialist runs next, or FINISH when enough info has been gathered."
    )
    reason: str = Field(description="One-sentence rationale for this routing choice.")


SUPERVISOR_PROMPT = """You are the orchestrator of a travel-planning team.

Specialists you can dispatch:
  • weather      — climate / forecast at the destination
  • hotels       — places to stay
  • attractions  — things to do
  • transport    — how to get there (road / rail / air)

Rules:
  • Each specialist may be called AT MOST ONCE.
  • Decide who to run NEXT based on the user's query and what's still missing.
  • When everything needed for a full itinerary has been gathered, reply with FINISH.
  • Always include a short reason — this teaches workshop participants how supervisors reason.
"""

structured_router = llm.with_structured_output(Route)


class TravelState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    called: list[str]   # specialists that have already run
    next: str           # routing decision for the next edge


def supervisor_node(state: TravelState) -> dict:
    banner(">> SUPERVISOR thinking...")
    remaining = [a for a in AGENTS if a not in state["called"]]
    context = HumanMessage(content=(
        f"Specialists already called: {state['called'] or 'none'}\n"
        f"Still available:           {remaining or 'none — you must FINISH'}\n\n"
        "Which specialist should run NEXT (or FINISH)?"
    ))
    decision: Route = structured_router.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"], context]
    )
    print(f"   -> decision: {decision.next}   ({decision.reason})")
    return {"next": decision.next}


def make_specialist_node(name: str, agent):
    """Wrap a ReAct agent so it slots into the StateGraph as a node."""
    def node(state: TravelState) -> dict:
        banner(f">> {name.upper()} AGENT working...")
        # Give the specialist ONLY the original user query — keeps prompts small
        # and demonstrates clean separation of concerns.
        original = state["messages"][0]
        result = agent.invoke({"messages": [original]})
        final = result["messages"][-1]
        tagged = AIMessage(content=f"[{name} agent]\n{final.content}", name=name)
        preview = final.content.replace("\n", " ")[:140]
        print(f"   [OK] {name} replied: {preview}...")
        return {"messages": [tagged], "called": state["called"] + [name]}
    return node


def synthesizer_node(state: TravelState) -> dict:
    banner(">> SYNTHESIZER composing final itinerary...")
    instruction = HumanMessage(content=(
        "Compose the final travel plan as a clean markdown itinerary using ONLY the "
        "information already gathered above. Structure it as:\n\n"
        "  ## Trip Summary\n"
        "  ## Getting There\n"
        "  ## Weather & What to Pack\n"
        "  ## Where to Stay (top picks with prices)\n"
        "  ## Day-by-Day Plan\n"
        "  ## Estimated Total Cost\n\n"
        "Be specific and actionable. Do NOT invent facts not present in the prior messages."
    ))
    reply = synth_llm.invoke([*state["messages"], instruction])
    return {"messages": [reply]}


def route_from_supervisor(state: TravelState) -> str:
    return "synthesizer" if state["next"] == "FINISH" else state["next"]


# ─────────────────────────── graph wiring ───────────────────────────

def build_graph():
    g = StateGraph(TravelState)
    g.add_node("supervisor",  supervisor_node)
    g.add_node("weather",     make_specialist_node("weather",     weather_agent))
    g.add_node("hotels",      make_specialist_node("hotels",      hotel_agent))
    g.add_node("attractions", make_specialist_node("attractions", attractions_agent))
    g.add_node("transport",   make_specialist_node("transport",   transport_agent))
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "weather":     "weather",
            "hotels":      "hotels",
            "attractions": "attractions",
            "transport":   "transport",
            "synthesizer": "synthesizer",
        },
    )
    # After each specialist, return to the supervisor for the next decision.
    for a in AGENTS:
        g.add_edge(a, "supervisor")
    g.add_edge("synthesizer", END)
    return g.compile()


# ──────────────────────────────── main ────────────────────────────────

DEFAULT_QUERY = (
    "what is a weather in Pune now")


def main():
    import sys
    user_query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    print(f"\nUSER QUERY:\n   {user_query}")

    graph = build_graph()
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_query)], "called": [], "next": ""},
        {"recursion_limit": 25},
    )

    banner("FINAL ITINERARY", "=")
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    main()
