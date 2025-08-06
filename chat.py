 from pydantic import BaseModel                                                                                                                       │
 │     2                                                                                                                                                      │
 │     3 class Message(BaseModel):                                                                                                                            │
 │     4     text: str                                                                                                                                        │
 │     5     sender: str                                                                                                                                      │
 │     6                                                                                                                                                      │
 │     7 class ChatRequest(BaseModel):                                                                                                                        │
 │     8     message: str                                                                                                                                     │
 │     9                                                                                                                                                      │
 │    10 class ChatResponse(BaseModel):                                                                                                                       │
 │    11     response: str  