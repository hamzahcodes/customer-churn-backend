import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.router import user, authentication, files
app = FastAPI()

origins = [ "*" ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(user.router)
app.include_router(authentication.router)
app.include_router(files.router)

@app.get("/")
def home():
    path = os.getcwd()
    x = path.replace("\\", "/")
    return {"data" : x}
