#!/usr/bin/env python3
"""
Trace the flow for a prompt like "pull data from India table".
Prints output at each step: grounding, validation, etc.
Run from IntegrityCore root: python scripts/trace_flow.py
"""
import os
import sys

# Ensure we can import integritycore
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "pull data from India table"
    print("=" * 60)
    print(f"TRACING FLOW FOR: \"{prompt}\"")
    print("=" * 60)

    # 1. Load connections
    from integritycore.adapters.connections import ConnectionManager
    conn_mgr = ConnectionManager()
    connections = conn_mgr.load_connections()
    snowflake = [c for c in connections if getattr(c, "dialect", "").upper() == "SNOWFLAKE"]

    print("\n--- 1. CONNECTIONS ---")
    if not connections:
        print("No connections configured. Add a connection in the UI first.")
        return
    print(f"Found {len(connections)} connection(s): {[c.name for c in connections]}")
    if not snowflake:
        print("No Snowflake connection. Harvest and grounding require Snowflake.")
        return
    src = snowflake[0]
    conn_id = src.id or src.name
    print(f"Using source: {src.name} (conn_id={conn_id})")

    # 2. MetadataManager
    print("\n--- 2. METADATA MANAGER ---")
    try:
        from integritycore.metadata.manager import MetadataManager
        mgr = MetadataManager()
        print(f"DuckDB: {mgr.duckdb_path}")
        print(f"LanceDB: {mgr.lancedb_path}")
        print(f"LanceDB table exists: {mgr._lance_table is not None}")
    except Exception as e:
        print(f"MetadataManager failed: {e}")
        return

    # 3. Grounding: vector search
    print("\n--- 3. GROUNDING: VECTOR SEARCH ---")
    hits = mgr.search_by_semantics(prompt, conn_id=conn_id, top_k=10)
    if hits:
        print(f"Vector search found {len(hits)} hit(s):")
        for h in hits:
            print(f"  - {h[1]}.{h[2]} (conn={h[0]}, distance={h[3]:.4f})")
    else:
        print("No vector search hits.")

    # 4. Grounding: fallback table match
    print("\n--- 4. GROUNDING: FALLBACK TABLE MATCH ---")
    tables = mgr.list_tables(conn_id)
    print(f"Catalog has {len(tables)} table(s) for conn_id={conn_id}")
    if tables:
        words = set(w.lower().strip(".,;:!?") for w in prompt.split() if len(w) > 2)
        print(f"Prompt words (len>2): {words}")
        for sch, tbl in tables[:15]:
            if any(w in (sch or "").lower() or w in (tbl or "").lower() for w in words):
                print(f"  Match: {sch}.{tbl}")

    # 5. Full GroundingEngine.retrieve()
    print("\n--- 5. GROUNDING ENGINE: retrieve() ---")
    from integritycore.core.grounding import GroundingEngine
    engine = GroundingEngine(mgr)
    result = engine.retrieve(user_prompt=prompt, conn_id=conn_id, expand_fk=False, top_k=10)

    print(f"related_tables: {result.related_tables}")
    print(f"confidence: {result.confidence:.2f}")
    print(f"closeness_matches: {result.closeness_matches}")
    print(f"verified_schema_fragment (length): {len(result.verified_schema_fragment)} chars")
    if result.verified_schema_fragment:
        print("verified_schema_fragment (first 500 chars):")
        print(result.verified_schema_fragment[:500])
    print(f"semantic_mappings: {result.semantic_mappings}")

    # 6. Simulate ground_context_node logic
    print("\n--- 6. GROUND CONTEXT NODE (simulated) ---")
    if not result.related_tables and not result.verified_schema_fragment:
        import re
        stop = {"pull", "data", "from", "the", "table", "to", "into", "load", "copy", "a", "an", "and", "or", "of", "for"}
        words = [w for w in re.findall(r'\b\w+\b', prompt.lower()) if w not in stop]
        table_hint = words[0] if words else "your query"
        print(f"NO_CATALOG_MATCH: No tables matching '{table_hint}' in catalog.")
        print("  -> User would see: Run harvest or specify SCHEMA.TABLE")
    elif result.related_tables:
        selected = result.related_tables[0]
        print(f"AUTO_SELECT: Using first match: {selected}")
        print(f"  -> Proceeds to validation (target table needed if not in prompt)")

    # 7. Harvest stats
    print("\n--- 7. HARVEST STATS ---")
    stats = mgr.get_harvest_stats(conn_id)
    print(f"table_count: {stats['table_count']}")
    print(f"columns_count: {stats['columns_count']}")
    print(f"last_harvest_at: {stats['last_harvest_at']}")

    print("\n" + "=" * 60)
    print("TRACE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
