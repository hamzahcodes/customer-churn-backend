from pydantic import BaseModel
from typing import List

class User(BaseModel):
    user_id: int
    username: str
    email: str
    password: str

class LoginUser(BaseModel):
    email: str
    password: str

class Item(BaseModel):
    rows: int
    cols: int
    predt: List[int] = []
    predf: List[int] = []
    predL: List[int] = []
    tree_error: float
    forest_error: float
    logistic_reg_error: float