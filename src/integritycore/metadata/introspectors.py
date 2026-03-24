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
        """Primary and foreign key constraints for relational graph.
        Uses TABLE_CONSTRAINTS + REFERENTIAL_CONSTRAINTS per Snowflake Information Schema.
        Snowflake does not have KEY_COLUMN_USAGE; column names from SHOW PRIMARY KEYS / SHOW IMPORTED KEYS.
        """
        cur = self._cursor()
        try:
            constraints = []
            if not schema:
                return constraints
            db = database or (self.conn.database if hasattr(self.conn, "database") else None)
            # Use TABLE_CONSTRAINTS only (no KEY_COLUMN_USAGE - not in Snowflake per docs)
            # per https://www.snowflake.com/en/blog/using-snowflake-information-schema/
            qual = f'"{db}".information_schema.table_constraints' if db else "information_schema.table_constraints"
            cur.execute(
                f"""
                SELECT constraint_name, constraint_type, table_schema, table_name
                FROM {qual}
                WHERE table_schema = %s AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
                """ + (" AND table_name = %s" if table else "") + " ORDER BY constraint_name",
                (schema, table) if table else (schema,),
            )
            rows = cur.fetchall()
            # Build table qualifier for SHOW commands
            tbl_qual = f'"{db}"."{schema}"."{table}"' if db and table else (f'"{schema}"."{table}"' if table else None)
            pk_cols = self._get_pk_columns(cur, db, schema, table, tbl_qual)
            fk_cols = self._get_fk_columns(cur, db, schema, table, tbl_qual)
            for r in rows:
                cname, ctype, sch, tbl = r[0], r[1], r[2], r[3]
                if table and tbl != table:
                    continue
                ct = ConstraintType.PRIMARY_KEY if ctype == "PRIMARY KEY" else ConstraintType.FOREIGN_KEY
                cols = pk_cols.get((cname, sch, tbl), []) if ct == ConstraintType.PRIMARY_KEY else fk_cols.get((cname, sch, tbl), [])
                ref_schema, ref_table, ref_cols = None, None, None
                if ct == ConstraintType.FOREIGN_KEY:
                    ref_schema, ref_table, ref_cols = self._get_fk_reference(cur, db, schema, tbl, cname)
                constraints.append(
                    ConstraintMetadata(
                        constraint_name=cname,
                        constraint_type=ct,
                        table_schema=sch,
                        table_name=tbl,
                        column_names=cols,
                        ref_table_schema=ref_schema,
                        ref_table_name=ref_table,
                        ref_column_names=ref_cols,
                    )
                )
            return constraints
        except Exception as e:
            log.debug("get_constraints failed (returning empty): %s", e)
            return []
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def _get_pk_columns(
        self, cur, database: Optional[str], schema: str, table: Optional[str], tbl_qual: Optional[str]
    ) -> dict:
        """Get PK column names via SHOW PRIMARY KEYS (Snowflake has no KEY_COLUMN_USAGE)."""
        out: dict = {}
        try:
            if tbl_qual:
                cur.execute(f"SHOW PRIMARY KEYS IN TABLE {tbl_qual}")
            elif database and schema:
                cur.execute(f'SHOW PRIMARY KEYS IN SCHEMA "{database}"."{schema}"')
            else:
                return out
            desc = [d[0].lower() for d in (cur.description or [])]
            for row in cur.fetchall():
                r = dict(zip(desc, row)) if desc else {}
                cname, col = r.get("constraint_name"), r.get("column_name")
                sch = r.get("schema_name") or schema
                tbl = r.get("table_name") or table
                seq = r.get("key_sequence") or 0
                if cname and col and sch and tbl:
                    key = (cname, sch, tbl)
                    if key not in out:
                        out[key] = []
                    out[key].append((int(seq) if seq is not None else 0, col))
            for k in list(out.keys()):
                out[k] = [c for _, c in sorted(out[k], key=lambda x: x[0])]
        except Exception:
            pass
        return out

    def _get_fk_columns(
        self, cur, database: Optional[str], schema: str, table: Optional[str], tbl_qual: Optional[str]
    ) -> dict:
        """Get FK column names via SHOW IMPORTED KEYS."""
        out: dict = {}
        try:
            if tbl_qual:
                cur.execute(f"SHOW IMPORTED KEYS IN TABLE {tbl_qual}")
            elif database and schema:
                cur.execute(f'SHOW IMPORTED KEYS IN SCHEMA "{database}"."{schema}"')
            else:
                return out
            desc = [d[0].lower() for d in (cur.description or [])]
            for row in cur.fetchall():
                r = dict(zip(desc, row)) if desc else {}
                cname = r.get("constraint_name")
                col = r.get("column_name") or r.get("fk_column_name")
                sch = r.get("schema_name") or (r.get("fk_schema_name") if "fk_schema_name" in r else schema)
                tbl = r.get("table_name") or (r.get("fk_table_name") if "fk_table_name" in r else table)
                if cname and col and sch and tbl:
                    key = (cname, sch, tbl)
                    if key not in out:
                        out[key] = []
                    out[key].append(col)
        except Exception:
            pass
        return out

    def _get_fk_reference(
        self,
        cur,
        database: Optional[str],
        schema: str,
        table: str,
        constraint_name: str,
    ) -> tuple:
        """Snowflake: REFERENTIAL_CONSTRAINTS + TABLE_CONSTRAINTS for ref table (no KEY_COLUMN_USAGE)."""
        try:
            qual = f'"{database}".information_schema' if database else "information_schema"
            cur.execute(
                f"""
                SELECT rc.unique_constraint_schema, tc.table_name
                FROM {qual}.referential_constraints rc
                JOIN {qual}.table_constraints tc
                  ON tc.constraint_name = rc.unique_constraint_name
                  AND tc.constraint_schema = rc.unique_constraint_schema
                WHERE rc.constraint_schema = %s AND rc.constraint_name = %s
                """,
                (schema, constraint_name),
            )
            row = cur.fetchone()
            if not row:
                return None, None, None
            return row[0], row[1], None  # ref_cols not available without KEY_COLUMN_USAGE
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
