from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The Url of the srouce")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The response to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the response",
    )
