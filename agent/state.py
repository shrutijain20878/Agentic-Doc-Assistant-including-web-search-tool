from typing import TypedDict, List

class AgentState(TypedDict):

    question: str
    tool: str
    answer: str