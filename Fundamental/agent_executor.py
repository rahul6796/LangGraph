
from langchain import hub 
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
fr


from typing import TypedDict, Annotated, Union, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

import os
load_dotenv()

tools = [TavilySearchResults(max_results=1)]

llm = ChatOllama(model="mistral")
prompt = hub.pull("hwchase17/react")

agent_runnable = create_react_agent(llm, tools, prompt)



class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

tool_executor = ToolExecutor(tools)

## define the agent:

def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

# define the execute tool
def execute_tools(data):
    print("Called `execute_tools`")
    messages = [data["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(data["agent_outcome"], response)]}



## define the logic that will be used to determin the condition edge to go down

def should_continue(data):

    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"
    


### Define the Graph

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
  
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")
app = workflow.compile()


inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")





