


from typing import TypedDict


class GraphState(TypedDict):
    input: str
    assigned_node: int 
    prev_node_to_write: str


