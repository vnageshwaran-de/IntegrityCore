import json
import os
from typing import List, Optional
from pydantic import BaseModel
import uuid

class DBConnection(BaseModel):
    id: str
    name: str
    dialect: str
    host: Optional[str] = None
    port: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    account: Optional[str] = None
    warehouse: Optional[str] = None
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    service_account_json: Optional[str] = None

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

    def add_connection(self, name: str, dialect: str, **kwargs) -> DBConnection:
        conns = [c.model_dump() for c in self.load_connections()]
        new_conn = {
            "id": str(uuid.uuid4()),
            "name": name,
            "dialect": dialect,
            **kwargs
        }
        conns.append(new_conn)
        
        with open(self.filepath, 'w') as f:
            json.dump(conns, f, indent=4)
            
        return DBConnection(**new_conn)

    def delete_connection(self, conn_id: str) -> bool:
        conns = self.load_connections()
        initial_count = len(conns)
        conns = [c for c in conns if c.id != conn_id]

        if len(conns) == initial_count:
            return False

        with open(self.filepath, 'w') as f:
            json.dump([c.model_dump() for c in conns], f, indent=4)

        return True

    def update_connection(self, conn_id: str, **fields) -> Optional[DBConnection]:
        conns = self.load_connections()
        updated = None
        for c in conns:
            if c.id == conn_id:
                data = c.model_dump()
                for k, v in fields.items():
                    if k in data and v is not None:
                        data[k] = v
                updated = DBConnection(**data)
                break

        if updated is None:
            return None

        with open(self.filepath, 'w') as f:
            json.dump([c.model_dump() if c.id != conn_id else updated.model_dump()
                       for c in conns], f, indent=4)

        return updated
