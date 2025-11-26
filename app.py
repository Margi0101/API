from fastapi import FastAPI
from typing import Annotated

app = FastAPI()

@app.post("/Registration")
def register(id: Annotated[int,"Form"], name: Annotated[str, "Form"]):

    return{"message":f"welcome!{name} to the futuretech"}