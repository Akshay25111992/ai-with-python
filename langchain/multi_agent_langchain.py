# Supervisor-based Multi-Agent System with Langchain

# %pip install -U langchain langchain-openai wikipedia on terminal OR

# python -m pip install -U langchain langchain-openai  then
# python -c "import langchain; print(langchain.__version__)" LangChain 1.3.18 - then below should support
# from langchain.agents import create_agent

# Do not hard-code API keys in this file.
# Set OPENAI_API_KEY in the environment, or replace the
# ChatOpenAI configuration with your provider's base_url/API key.
# pip install -q python-dotenv on terminal
# python -m pip install langchain-tavily

from dotenv import load_dotenv

load_dotenv()
import os
import wikipedia

import requests
import json
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch

import warnings

warnings.filterwarnings("ignore")

# print(os.getenv("OPENAI_API_KEY")) to test if .env file is working
tavily_api = TavilySearch(
    api_key=os.environ["TAVILY_API_KEY"],
    max_results=5,
    search_depth="basic")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------
# Step 1: Initialize the LLM
# For standard OpenAI:
#   os.environ["OPENAI_API_KEY"] = "your-key"
#   model="gpt-4o-mini"
#
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


# use below 2 lines of code to test connection first time
# response = llm.invoke("What is artificial intelligence? Explain in one sentence.")
# print(response.content)

# ------Step 2: Define tools------------------------------------------------------

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
        print(f"   [WEATHER ERROR] {type(e).__name__}: {e}")
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
            {"name": "Hyatt Pune", "price_per_night": 7200, "rating": 4.5, "area": "Kalyani Nagar"},
            {"name": "Hotel Sagar Plaza", "price_per_night": 4200, "rating": 4.1, "area": "Pune Station"},
            {"name": "Lemon Tree Hotel", "price_per_night": 3800, "rating": 4.0, "area": "Hinjewadi"},
            {"name": "Treebo Trend Crystal Inn", "price_per_night": 2400, "rating": 3.8, "area": "Shivajinagar"},
        ],
    }
    options = catalogue.get(city, [
        {"name": f"{city} Grand Hotel", "price_per_night": 5500, "rating": 4.3, "area": "City Centre"},
        {"name": f"{city} Comfort Inn", "price_per_night": 3200, "rating": 4.0, "area": "Downtown"},
        {"name": f"{city} Budget Stay", "price_per_night": 1800, "rating": 3.6, "area": "Suburbs"},
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
        results = tavily_api.invoke({
            "query": f"top {interests} to visit in {city} for a 2-day trip"
        }

        )
        summary = [
            {"title": r["title"], "snippet": r["content"][:280], "url": r["url"]}
            for r in results.get("results", [])
        ]
        return json.dumps({"city": city, "interests": interests, "results": summary}, indent=2)
    except Exception as e:
        print(f"   [ATTRACTIONS ERROR] {type(e).__name__}: {e}")
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
        results = tavily_api.invoke(
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


# ──Step 3 ─────────────────────── specialist agents ────────────────────────
# Each specialist is a tiny ReAct agent that owns exactly one tool.
# Keeping them single-tool makes the reasoning loop short and easy to demo.


weather_agent = create_agent(
    llm, tools=[get_weather],
    system_prompt="""
       You are a Weather Specialist.

    IMPORTANT RULES:
    1. Call get_weather exactly ONCE.
    2. After receiving the tool result, immediately provide the final answer.
    3. Never call get_weather a second time.
    4. Do not retry the tool.
    5. If the tool returns an error, report the error clearly.
    6. Do not call any other tools.
    """
)

hotel_agent = create_agent(
    llm, tools=[search_hotels],
    system_prompt=(
        "You are a hotel-booking specialist. Call search_hotels with sensible defaults "
        "(budget ≈ ₹5000/night unless told otherwise). Return the top 2-3 options with "
        "name, area, per-night price, and total cost."
    ),
)

attractions_agent = create_agent(
    llm, tools=[find_attractions],
    system_prompt="""
    You are an Attractions Specialist.

    IMPORTANT RULES:
    1. Call find_attractions exactly ONCE.
    2. After receiving the tool result, immediately provide the final answer.
    3. Never call find_attractions a second time.
    4. Do not retry the tool.
    5. If the tool returns an error, report the error clearly.
    """
)

transport_agent = create_agent(
    llm, tools=[plan_route],
    system_prompt=(
        "You are a transport planner. Use plan_route to research the journey, then summarise "
        "distance, estimated travel time, and any practical tips (rest stops, tolls, alt routes)."
    ),
)


# ─────Step 4──────────────────── defining all specialist as tools ────────────────────────
# in this LangChain-only design, create_agent() expects a list of tools it can choose from.
# To let the supervisor dynamically delegate to another agent, we expose that agent as a tool..

@tool
def ask_weather_specialist(query: str) -> str:
    """Ask the Weather Specialist about weather conditions."""

    print("\n>>> WEATHER AGENT CALLED")

    result = weather_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config={"recursion_limit": 10}
    )

    return result["messages"][-1].content


@tool
def ask_hotel_specialist(query: str) -> str:
    """Ask the Hotel Specialist for accommodation recommendations."""

    print("\n>>> HOTEL AGENT CALLED")

    result = hotel_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    },
        config={"recursion_limit": 10}
    )

    return result["messages"][-1].content


@tool
def ask_attractions_specialist(query: str) -> str:
    """Ask the Attractions Specialist about places to visit."""

    print("\n>>> ATTRACTIONS AGENT CALLED")

    result = attractions_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    return result["messages"][-1].content


@tool
def ask_transport_specialist(query: str) -> str:
    """Ask the Transport Specialist for transportation planning."""

    print("\n>>> TRANSPORT AGENT CALLED")

    result = transport_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    return result["messages"][-1].content


# --------Step 5: Create supervisor agent----------------------------------------------------

specialist_tools = [
    ask_weather_specialist,
    ask_hotel_specialist,
    ask_attractions_specialist,
    ask_transport_specialist,
]

supervisor_agent = create_agent(
    model=llm,
    tools=specialist_tools,
    system_prompt="""
# ROLE
You are a Travel Planning Supervisor Agent.
# AVAILABLE SPECIALISTS

You can delegate work to:

1. Weather Specialist
   - Weather and climate information

2. Hotel Specialist
   - Accommodation recommendations

3. Attractions Specialist
   - Places to visit

4. Transport Specialist
   - Travel and transportation planning

# INSTRUCTIONS

1. Understand the user's complete request.
2. Decide which specialist agents are required.
3. Call only the specialists needed.
4. Call multiple specialists when necessary.
5. Use the returned information to create one complete response.
6. Do not call a specialist unnecessarily.

# CONSTRAINTS

- You are responsible for coordination.
- Specialist agents are responsible for their own domains.
- Combine specialist responses into a coherent answer.
- Do not invent information that was not returned.

# OUTPUT

Provide a clear and concise travel plan.
"""
)


# ------Step 6: Run the multi agent------------------------------------------------------


def ask_supervisor(question: str) -> str:
    """Send a question and display agent tool decisions."""

    result = supervisor_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    print("\n" + "-" * 60)
    print("AGENT EXECUTION TRACE")
    print("-" * 60)

    for message in result["messages"]:
        print("\nMESSAGE CLASS:", type(message))
        print("MESSAGE TYPE:", message.type)
        print("CONTENT:", message.content)

        # Show tool calls selected by the agent
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("TOOL CALLS:", message.tool_calls)

            for tool_call in message.tool_calls:
                print("\n AGENT DECISION")
                print(f"Tool selected: {tool_call['name']}")
                print(f"Input: {tool_call['args']}")

        # Show tool results
        if message.type == "tool":
            print("\n TOOL EXECUTED")
            print(f"Tool: {message.name}")
            print(f"Output: {message.content}")

    print("\n" + "-" * 60)

    return result["messages"][-1].content


# ------Step 7: Demonstrations------------------------------------------------------


if __name__ == "__main__":

    queries = [
        "What is the weather in Pune?",
        "I am travelling to Pune. What is the weather and what are some good attractions?",
        "Plan a trip from Mumbai to Pune. Include: transportation, weather, hotels, attractions",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        print(f"Answer: {ask_supervisor(query)}")

# python -m pip install --upgrade aiohttp openai langchain-openai on terminal in case you get an error
# AttributeError: module aiohttp has no attribute SocketTimeoutError
# result["messages"] contains 4 messages in a squence like below
# 1. HumanMessage  → User question
# 2. AIMessage     → LLM decides to call calculator
# 3. ToolMessage   → Calculator returns result
# 4. AIMessage     → LLM produces final answer

