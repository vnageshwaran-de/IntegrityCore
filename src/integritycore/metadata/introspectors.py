"""Introspection engine: extensible base and dialect-specific implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from integritycore.metadata.models import (
    ColumnMetadata,
    ConstraintMetadata,
    ConstraintType,
    TableMetadata,
)

log = logging.getLogger("integritycore.metadata.introspectors")


class BaseIntrospector(ABC):
    """Extensible base for database introspection (column metadata, constraints, comments)."""

    def __init__(self, conn: Any, conn_id: str = ""):
        self.conn = conn
        self.conn_id = conn_id or getattr(conn, "id", "") or getattr(conn, "name", "")

    @abstractmethod
    def get_columns(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[ColumnMetadata]:
        """Return column-level metadata (ordinal position, is_nullable, data_type, comment)."""
        pass

    @abstractmethod
    def get_constraints(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[ConstraintMetadata]:
        """Return primary/foreign key constraints to build relational graph."""
        pass

    @abstractmethod
    def get_tables(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> List[tuple]:
        """Return list of (schema_name, table_name) or (database, schema_name, table_name)."""
        pass

    def get_table_comment(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> Optional[str]:
        """Override in subclass if supported. Default None."""
        return None

    def introspect_table(
        self,
        database: Optional[str] = None,
        schema: str = "PUBLIC",
        table_name: str = "",
    ) -> Optional[TableMetadata]:
        """Full introspection for one table: columns, constraints, comment."""
        columns = self.get_columns(database, schema, table_name)
        if not columns and table_name:
            return None
        constraints = self.get_constraints(database, schema, table_name)
        comment = self.get_table_comment(database, schema, table_name)
        return TableMetadata(
            conn_id=self.conn_id,
            database=database,
            schema_name=schema or "",
            table_name=table_name or (columns[0].table_name if columns else ""),
            columns=columns,
            constraints=constraints,
            table_comment=comment,
        )


class SnowflakeIntrospector(BaseIntrospector):
    """Snowflake-specific introspection: columns, PK/FK, comments from information_schema."""

    def _cursor(self):
        try:
            import snowflake.connector
        except ImportError:
            raise RuntimeError("snowflake-connector-python is required for Snowflake introspection")
        if hasattr(self.conn, "cursor"):
            return self.conn.cursor()
        return snowflake.connector.connect(
            user=self.conn.username,
            password=self.conn.password,
            account=self.conn.account,
            database=self.conn.database or "",
            warehouse=self.conn.warehouse or "",
        ).cursor()

    def get_tables(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> List[tuple]:
        """List tables: (database, schema_name, table_name) or (schema_name, table_name)."""
        cur = self._cursor()
        try:
            db = database or (self.conn.database if hasattr(self.conn, "database") else None)
            if db and schema:
                cur.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_catalog = %s AND table_schema = %s AND table_type = 'BASE TABLE'",
                    (db, schema),
                )
            elif schema:
                cur.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                    (schema,),
                )
            else:
                cur.execute(
                    "SELECT table_catalog, table_schema, table_name FROM information_schema.tables "
                    "WHERE table_type = 'BASE TABLE' LIMIT 500"
                )
            rows = cur.fetchall()
            if db and schema and rows:
                return [(r[0], r[1]) for r in rows]
            if rows and len(rows[0]) == 3:
                return [(r[0], r[1], r[2]) for r in rows]
            return [(r[0], r[1]) for r in rows]
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def get_columns(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[ColumnMetadata]:
        """Column-level metadata: ordinal position, is_nullable, data_type, comment."""
        cur = self._cursor()
        try:
            if table and schema:
                cur.execute(
                    """
                    SELECT ordinal_position, column_name, data_type, is_nullable, comment
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (schema, table),
                )
            else:
                return []
            out = []
            for row in cur.fetchall():
                out.append(
                    ColumnMetadata(
                        name=row[1],
                        ordinal_position=int(row[0]) if row[0] is not None else 0,
                        data_type=row[2] or "",
                        is_nullable=(row[3] == "YES") if row[3] else True,
                        comment=row[4] if len(row) > 4 else None,
                        table_schema=schema or "",
                        table_name=table or "",
                    )
                )
            return out
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def get_constraints(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[ConstraintMetadata]:
        """Primary and foreign key constraints for relational graph."""
        cur = self._cursor()
        try:
            constraints = []
            if not schema:
                return constraints
            # Snowflake: constraint info from information_schema
            cur.execute(
                """
                SELECT tc.constraint_name, tc.constraint_type, tc.table_schema, tc.table_name,
                       kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                  AND tc.table_name = kcu.table_name
                WHERE tc.table_schema = %s AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
                """ + (" AND tc.table_name = %s" if table else "") + " ORDER BY tc.constraint_name, kcu.ordinal_position",
                (schema, table) if table else (schema,),
            )
            rows = cur.fetchall()
            # Group by constraint
            by_key: dict = {}
            for r in rows:
                cname, ctype, sch, tbl, col = r[0], r[1], r[2], r[3], r[4]
                key = (cname, sch, tbl)
                if key not in by_key:
                    by_key[key] = {"type": ctype, "cols": [], "ref": None}
                by_key[key]["cols"].append(col)

            for (cname, sch, tbl), v in by_key.items():
                if table and tbl != table:
                    continue
                ct = ConstraintType.PRIMARY_KEY if v["type"] == "PRIMARY KEY" else ConstraintType.FOREIGN_KEY
                ref_schema, ref_table, ref_cols = None, None, None
                if ct == ConstraintType.FOREIGN_KEY:
                    ref_schema, ref_table, ref_cols = self._get_fk_reference(cur, schema, tbl, cname)
                constraints.append(
                    ConstraintMetadata(
                        constraint_name=cname,
                        constraint_type=ct,
                        table_schema=sch,
                        table_name=tbl,
                        column_names=v["cols"],
                        ref_table_schema=ref_schema,
                        ref_table_name=ref_table,
                        ref_column_names=ref_cols,
                    )
                )
            return constraints
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def _get_fk_reference(
        self,
        cur,
        schema: str,
        table: str,
        constraint_name: str,
    ) -> tuple:
        try:
            cur.execute(
                """
                SELECT referenced_table_schema, referenced_table_name
                FROM information_schema.referential_constraints
                WHERE constraint_schema = %s AND constraint_name = %s
                """,
                (schema, constraint_name),
            )
            row = cur.fetchone()
            if not row:
                return None, None, None
            ref_schema, ref_table = row[0], row[1]
            cur.execute(
                """
                SELECT referenced_column_name FROM information_schema.key_column_usage
                WHERE constraint_schema = %s AND constraint_name = %s
                ORDER BY ordinal_position
                """,
                (schema, constraint_name),
            )
            ref_cols = [r[0] for r in cur.fetchall() if r[0]]
            return ref_schema, ref_table, ref_cols
        except Exception:
            return None, None, None

    def get_table_comment(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
    ) -> Optional[str]:
        """Table comment from Snowflake (SHOW TABLES or information_schema)."""
        if not schema or not table:
            return None
        cur = self._cursor()
        try:
            cur.execute(
                "SELECT comment FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                (schema, table),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None
        finally:
            try:
                cur.close()
            except Exception:
                pass
