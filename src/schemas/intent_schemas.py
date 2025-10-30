"""This schema is used by the retiever to decide on the intent of the user query"""

from pydantic import BaseModel
from typing import List, Optional

class RoutingDecision(BaseModel):
    use_tools: List[str]     # tools to call in order
    reason: str              # explanation for debugging
