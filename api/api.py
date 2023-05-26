from fastapi import FastAPI
from api.database import send_query


app = FastAPI()


@app.get("/test")
async def test():
    result = send_query("select * from user")
    return result

