# Simple LangChain Agent with Tools
# Works with the current LangChain 1.x agent API.

# %pip install -U langchain langchain-openai wikipedia on terminal OR

#python -m pip install -U langchain langchain-openai  then
# python -c "import langchain; print(langchain.__version__)" LangChain 1.3.18 - then below should support
# from langchain.agents import create_agent

# Do not hard-code API keys in this file.
# Set OPENAI_API_KEY in the environment, or replace the
# ChatOpenAI configuration with your provider's base_url/API key.
#pip install -q python-dotenv on terminal

from dotenv import load_dotenv
load_dotenv()
import os
import wikipedia

#print(os.getenv("OPENAI_API_KEY"))
#print(os.getenv("OPENAI_BASE_URL"))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------
# Step 1: Initialize the LLM
# For standard OpenAI:
#   os.environ["OPENAI_API_KEY"] = "your-key"
#   model="gpt-4o-mini"
# For Vocareum:
#   set OPENAI_API_KEY and OPENAI_BASE_URL in the environment.
#
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
# use below 2 lines of code to test connection first time
#response = llm.invoke("What is artificial intelligence? Explain in one sentence.") 
#print(response.content)

# ------------------------------------------------------------
# Step 2: Define tools
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Perform a simple arithmetic calculation.


    Examples:
        10*10
        345*23+99
    """
    try:
        # Suitable only for a controlled classroom demo.
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short summary for a general-knowledge query."""
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return f"Your query is ambiguous. Possible topics include: {options}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page was found for: {query}"
    except Exception as e:
        return f"Wikipedia search error: {e}"


@tool
def weather_search(location: str) -> str:
    """Return demo weather information for selected cities."""
    weather = {
        "london": "Partly cloudy, 15°C.",
        "new york": "Sunny, 22°C.",
    }

    result = weather.get(location.strip().lower())
    if result:
        return f"Demo weather for {location}: {result}"

    return f"No demo weather data is available for {location}."


tools = [calculator, wikipedia_search, weather_search]


# ------------------------------------------------------------
# Step 3: Create the agent
# ------------------------------------------------------------
# create_agent is the current LangChain 1.x approach.
# The agent decides which tool(s) to use based on the question.

system_prompt = """
# ROLE
You are a helpful and reliable AI assistant.

# OBJECTIVE
Answer the user's question accurately and concisely.

# AVAILABLE TOOLS
You have access to the following tools:
- Calculator: Use for arithmetic and mathematical calculations.
- Wikipedia Search: Use for general knowledge questions.
- Weather Search: Use for weather-related questions supported by the tool.

# INSTRUCTIONS
1. Understand the user's request before responding.
2. Decide whether a tool is needed.
3. Select the most appropriate tool or tools.
4. Use multiple tools when the request requires multiple pieces of information.
5. Use the tool results when forming your final answer.
6. If no tool is needed, answer directly.

# CONSTRAINTS
- Do not perform arithmetic mentally when the Calculator tool is appropriate.
- Do not invent information that a tool should provide.
- Do not claim that you used a tool if you did not use it.
- Use only the available tools.
- If a tool cannot provide the requested information, clearly state the limitation.

# OUTPUT FORMAT
- Give a concise and clear final answer.
- Do not expose internal reasoning or chain-of-thought.
"""


#
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# ------------------------------------------------------------
# Step 4: Run the agent
# ------------------------------------------------------------

def ask_agent(question: str) -> str:
    """Send a question and display agent tool decisions."""

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    print("\n" + "=" * 60)
    print("AGENT EXECUTION TRACE")
    print("=" * 60)

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

    print("\n" + "=" * 60)

    return result["messages"][-1].content

# ------------------------------------------------------------
# Step 5: Demonstrations
# ------------------------------------------------------------

if __name__ == "__main__":

    queries = [
        "What is 109 * 10?",
        "What is the capital of India?",
        "What is the weather in London and what is 5 + 7?",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        print(f"Answer: {ask_agent(query)}")



#python -m pip install --upgrade aiohttp openai langchain-openai on terminal in case you get an error 
#AttributeError: module aiohttp has no attribute SocketTimeoutError
#python /voc/work/simple_agent_with_tools_V1.py - use path in case of error

#result["messages"] contains 4 messages in a squence like below
#1. HumanMessage  → User question
#2. AIMessage     → LLM decides to call calculator
#3. ToolMessage   → Calculator returns result
#4. AIMessage     → LLM produces final answer

