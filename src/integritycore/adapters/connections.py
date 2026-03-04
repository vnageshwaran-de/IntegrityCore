import json
import os
from typing import List
from pydantic import BaseModel
import uuid

class DBConnection(BaseModel):
    id: str
    name: str
    dialect: str
    host: str
    username: str
    password: str

class ConnectionManager:
    """Manages local storage of database connections."""
    def __init__(self, filepath: str = os.path.expanduser("~/.integritycore/connections.json")):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def load_connections(self) -> List[DBConnection]:
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            return [DBConnection(**conn) for conn in data]

    def add_connection(self, name: str, dialect: str, host: str, username: str, password: str) -> DBConnection:
        conns = [c.model_dump() for c in self.load_connections()]
        new_conn = {
            "id": str(uuid.uuid4()),
            "name": name,
            "dialect": dialect,
            "host": host,
            "username": username,
            "password": password
        }
        conns.append(new_conn)
        
        with open(self.filepath, 'w') as f:
            json.dump(conns, f, indent=4)
            
        return DBConnection(**new_conn)
