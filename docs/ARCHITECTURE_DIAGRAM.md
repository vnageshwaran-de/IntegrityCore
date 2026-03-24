# IntegrityCore Architecture — Detailed Mermaid Diagrams

## 1. System Context (C4 Level 1)

```mermaid
flowchart TB
    subgraph External["External"]
        User["👤 Data Engineer"]
        Snowflake[(Snowflake)]
        LiteLLM[LiteLLM / LLM Provider]
        APScheduler[APScheduler]
    end

    subgraph IntegrityCore["IntegrityCore"]
        ETLAgent["ETL Agent\nLangGraph: parse → ground → validate → generate → critique → verify → execute"]
    end

    User -->|"Creates jobs, dry-runs, confirms source/target"| ETLAgent
    User -->|"Triggers manual runs"| ETLAgent
    ETLAgent -->|"Introspect, execute SQL"| Snowflake
    ETLAgent -->|"Parse prompt, generate SQL, critique"| LiteLLM
    APScheduler -->|"Triggers scheduled runs"| ETLAgent
```

---

## 2. High-Level Component Architecture

```mermaid
flowchart TB
    subgraph UI["UI Layer"]
        direction TB
        WebApp["React SPA (index.html)"]
        FastAPI["FastAPI App (api.py)"]
        WebApp --> FastAPI
    end

    subgraph API["API Endpoints"]
        direction TB
        JobsAPI["/api/jobs"]
        RunsAPI["/api/runs"]
        CatalogAPI["/api/catalog"]
        ConnAPI["/api/connections"]
        ExploreAPI["/api/explore"]
        StatsAPI["/api/stats"]
    end

    subgraph Agents["Agent Layer"]
        direction TB
        LangGraph["LangGraph ETL Graph"]
        ParseNode["parse_prompt_node"]
        GroundNode["ground_context_node"]
        ValidateNode["validate_prompt_node"]
        GenNode["generate_sql_node"]
        CritiqueNode["critique_sql_node"]
        VerifyNode["verify_node"]
        ExecNode["execute_node"]
        RepairNode["execution_repair_node"]
    end

    subgraph Core["Core Services"]
        direction TB
        Grounding["GroundingEngine"]
        Verifier["LogicVerifier (Z3/SMT)"]
    end

    subgraph Adapters["Adapters"]
        direction TB
        ConnMgr["ConnectionManager"]
        Executor["DatabaseExecutor"]
        Cloud["CloudAdapter"]
    end

    subgraph Metadata["Metadata"]
        direction TB
        MetaMgr["MetadataManager"]
        Introspector["SnowflakeIntrospector"]
    end

    subgraph Storage["Storage"]
        direction TB
        SQLite[(SQLite\njobs, runs, gold_queries)]
        DuckDB[(DuckDB\ncatalog)]
        LanceDB[(LanceDB\nvector store)]
    end

    subgraph External["External"]
        Snowflake[(Snowflake)]
        LiteLLM[LiteLLM]
    end

    FastAPI --> API
    JobsAPI --> LangGraph
    RunsAPI --> LangGraph
    CatalogAPI --> MetaMgr
    ConnAPI --> ConnMgr
    ExploreAPI --> Executor

    LangGraph --> ParseNode
    LangGraph --> GroundNode
    LangGraph --> ValidateNode
    LangGraph --> GenNode
    LangGraph --> CritiqueNode
    LangGraph --> VerifyNode
    LangGraph --> ExecNode
    LangGraph --> RepairNode

    GroundNode --> Grounding
    Grounding --> MetaMgr
    MetaMgr --> DuckDB
    MetaMgr --> LanceDB
    MetaMgr --> Introspector

    GenNode --> LiteLLM
    GenNode --> Executor
    VerifyNode --> Verifier
    VerifyNode --> Executor
    ExecNode --> Executor

    Executor --> Snowflake
    Introspector --> Snowflake
    ConnMgr --> Snowflake

    LangGraph --> SQLite
```

---

## 3. ETL Graph Flow (Detailed)

```mermaid
flowchart TB
    START([START]) --> Parse

    Parse["parse_prompt_node\nLLM: source_database, source_schema, source_table, target_*"]
    Parse --> Ground

    Ground["ground_context_node\nretrieve_single_table OR vector search → grounded_ddl, source_tables_to_confirm"]
    Ground --> Validate

    Validate["validate_prompt_node\nPre-flight: catalog match, user confirmation for source & target"]
    Validate --> RouteVal["route_after_validation"]

    RouteVal -->|"is_valid_prompt=false"| END1([END])
    RouteVal -->|"is_valid_prompt=true"| Generate

    Generate["generate_sql_node\nLiteLLM + tools + Gold Query few-shot"]
    Generate --> Critique

    Critique["critique_sql_node\nSenior Data Engineer: Cartesian, PII, Snowflake patterns"]
    Critique --> RouteCrit["route_after_critique"]

    RouteCrit -->|"critique_passed=true"| Verify
    RouteCrit -->|"issues + attempts < 2"| Generate
    RouteCrit -->|"issues + max attempts"| Verify

    Verify["verify_node\nSMT (Z3) + executor.compile_only"]
    Verify --> RouteVer["route_after_verification"]

    RouteVer -->|"verified + dry_run"| END2([END])
    RouteVer -->|"verified + !dry_run"| Execute
    RouteVer -->|"failed + repair < 3"| PrepareRepair
    RouteVer -->|"failed + max repairs"| END3([END])

    PrepareRepair["prepare_repair_node"] --> Generate

    Execute["execute_node\nDatabaseExecutor.execute on target"]
    Execute --> RouteExec["route_after_execution"]

    RouteExec -->|"success"| END5([END])
    RouteExec -->|"failed + repair < 3"| ExecRepair
    RouteExec -->|"failed + max repairs"| END6([END])

    ExecRepair["execution_repair_node"] --> Generate
```

---

## 4. ETL State Machine (ETLState)

```mermaid
stateDiagram-v2
    [*] --> Parse

    Parse --> Ground: parsed_source_table, parsed_target_table

    Ground --> Validate: grounding_result, validation_result, source_tables_to_confirm

    Validate --> Generate: is_valid_prompt=true
    Validate --> [*]: is_valid_prompt=false

    Generate --> Critique: sql, messages, schema_ddl_used

    Critique --> Generate: critique_passed=false, repair
    Critique --> Verify: critique_passed=true

    Verify --> Execute: verified=true
    Verify --> Generate: verified=false, prepare_repair

    Execute --> [*]: success
    Execute --> Generate: failed, execution_repair

    note right of Generate
        repair_attempts < 3
        max_repairs = 3
    end note
```

---

## 5. Data Model (Entity Relationship)

```mermaid
erDiagram
    Job ||--o{ JobRun : "has many"
    Job {
        string id PK
        string name
        string description
        string source_conn_id
        string target_conn_id
        string prompt
        string selected_source_table
        string selected_target_table
        string schedule_cron
        string schedule_label
        string status
        datetime created_at
        datetime updated_at
    }

    JobRun {
        string id PK
        string job_id FK
        string status
        string triggered_by
        datetime started_at
        datetime finished_at
        float duration_seconds
        int rows_processed
        text logs
        text error_msg
        text generated_sql
        boolean verified
        string llm_model_version
        text input_ddl_context
        text source_target_map
        text logic_verification_result
    }

    GoldQuery {
        string id PK
        text problem_description
        text sql_query
        string dialect
        datetime created_at
    }

    DBConnection {
        string id
        string name
        string dialect
        string host
        string database
        string account
        string warehouse
    }

    Job }o--|| DBConnection : "source_conn"
    Job }o--|| DBConnection : "target_conn"
```

---

## 6. ETL Graph State (ETLState TypedDict)

```mermaid
flowchart LR
    subgraph State["ETLState"]
        direction TB

        subgraph Prompt["Prompt & Config"]
            P1["prompt"]
            P2["source_dialect"]
            P3["target_dialect"]
            P4["model_name"]
            P5["strategy"]
        end

        subgraph Parsed["LLM-Parsed"]
            PA1["parsed_source_database"]
            PA2["parsed_source_schema"]
            PA3["parsed_source_table"]
            PA4["parsed_target_schema"]
            PA5["parsed_target_table"]
        end

        subgraph Grounding["Grounding"]
            G1["grounding_result"]
            G2["grounded_ddl"]
            G3["schema_ddl_used"]
            G4["semantic_mappings"]
            G5["selected_source_table"]
            G6["selected_target_table"]
        end

        subgraph Validation["Validation"]
            V1["validation_result"]
            V2["is_valid_prompt"]
            V3["validation_error"]
        end

        subgraph Generation["Generation"]
            GEN1["sql"]
            GEN2["messages"]
            GEN3["repair_attempts"]
        end

        subgraph Critique["Critique"]
            C1["critique_passed"]
            C2["critique_issues"]
            C3["critique_repair_attempts"]
        end

        subgraph Verification["Verification"]
            VER1["verified"]
            VER2["verification_details"]
            VER3["special_action"]
        end

        subgraph Execution["Execution"]
            E1["execution_result"]
            E2["source_conn"]
            E3["target_conn"]
            E4["executor"]
        end

        subgraph Meta["Metadata"]
            M1["metadata_manager"]
        end
    end
```

---

## 7. API Endpoints Map

```mermaid
flowchart TB
    subgraph Catalog["Catalog API"]
        C1["GET /api/catalog/tables"]
        C2["GET /api/catalog/table/{conn_id}/{schema}/{table}"]
        C3["POST /api/catalog/harvest"]
        C4["POST /api/catalog/harvest/cancel"]
        C5["GET /api/catalog/harvest-status"]
        C6["GET /api/catalog/search"]
    end

    subgraph Jobs["Jobs API"]
        J1["GET /api/jobs"]
        J2["POST /api/jobs"]
        J3["GET /api/jobs/{id}"]
        J4["PUT /api/jobs/{id}"]
        J5["DELETE /api/jobs/{id}"]
        J6["POST /api/jobs/{id}/pause"]
        J7["POST /api/jobs/{id}/resume"]
        J8["POST /api/jobs/dry-run"]
        J9["POST /api/jobs/{id}/run"]
        J10["GET /api/jobs/{id}/runs"]
    end

    subgraph Runs["Runs API"]
        R1["GET /api/runs/{id}"]
        R2["GET /api/runs/{id}/lineage"]
        R3["POST /api/runs/{id}/promote-to-gold"]
        R4["GET /api/runs/{id}/logs"]
    end

    subgraph Connections["Connections API"]
        Conn1["GET /api/connections"]
        Conn2["POST /api/connections"]
        Conn3["PUT /api/connections/{id}"]
        Conn4["DELETE /api/connections/{id}"]
        Conn5["POST /api/connections/test"]
    end

    subgraph Explore["Explore API"]
        E1["GET /api/explore/schemas"]
        E2["GET /api/explore/tables"]
        E3["GET /api/explore/columns"]
        E4["GET /api/explore/preview"]
        E5["POST /api/explore/query"]
    end

    subgraph Stats["Stats API"]
        S1["GET /api/stats"]
    end

    J8 --> LangGraph["build_etl_graph"]
    J9 --> Runner["execute_run"]
    Runner --> LangGraph
```

---

## 8. Grounding Engine Flow

```mermaid
flowchart TB
    subgraph Input["Input"]
        Prompt["user_prompt"]
        ConnId["conn_id"]
        Parsed["parsed_source_table"]
    end

    subgraph GroundingEngine["GroundingEngine"]
        direction TB

        subgraph Retrieve["Retrieve"]
            R1["retrieve_single_table(schema, table)"]
            R2["OR vector search"]
            R3["search_by_semantics"]
            R4["_fallback_table_match"]
        end

        subgraph Expand["Expand"]
            E1["FK graph expansion"]
            E2["tables_to_include"]
        end

        subgraph Build["Build"]
            B1["get_table_metadata"]
            B2["_table_to_clean_ddl"]
            B3["_get_card"]
            B4["semantic_mappings"]
        end

        subgraph Output["Output"]
            O1["GroundingResult"]
            O2["verified_schema_fragment"]
            O3["related_tables"]
            O4["semantic_mappings"]
            O5["confidence"]
            O6["closeness_matches"]
        end
    end

    subgraph MetadataManager["MetadataManager"]
        MM1["search_by_semantics"]
        MM2["list_tables"]
        MM3["get_table_metadata"]
    end

    subgraph Storage["Storage"]
        DuckDB[(DuckDB)]
        LanceDB[(LanceDB)]
    end

    Prompt --> R1
    Parsed --> R1
    R1 -->|"no match"| R2
    R2 --> R3
    R3 -->|"no hits"| R4
    R3 -->|"hits"| E1
    E1 --> B1
    B1 --> B2
    B1 --> B3
    B2 --> O2
    B3 --> O4
    B1 --> O3

    MM1 --> LanceDB
    MM2 --> DuckDB
    MM3 --> DuckDB
```

---

## 9. Dry Run vs Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant API
    participant Graph
    participant Executor
    participant Snowflake

    User->>UI: Create job, run dry-run
    UI->>API: POST /api/jobs/dry-run
    API->>Graph: build_etl_graph().invoke(state_input)
    Note over Graph: parse → ground → validate → generate → critique → verify
    Graph->>Graph: is_dry_run=true
    Graph->>API: final_state (no execute)
    API->>UI: sql, verified, logs, validation_result

    alt needs_interaction
        API->>UI: source_tables_to_confirm / target_table_suggestions
        User->>UI: Select source & target
        UI->>API: POST /api/jobs/dry-run (with selected_*)
        API->>Graph: invoke again
    end

    User->>UI: Create job, trigger run
    UI->>API: POST /api/jobs/{id}/run
    API->>Graph: execute_run (subprocess)
    Graph->>Graph: parse → ground → validate → generate → critique → verify
    Graph->>Executor: execute(sql, target_conn)
    Executor->>Snowflake: Execute SQL
    Snowflake-->>Executor: Result
    Executor-->>Graph: execution_result
    alt Execution failed
        Graph->>Graph: execution_repair_node
        Graph->>Graph: generate_sql_node (retry)
    end
    Graph->>API: SUCCESS_RUN / FAILED_RUN
    API->>UI: Run status, logs
```

---

## 10. Scheduler & Runner Flow

```mermaid
flowchart TB
    subgraph Scheduler["Scheduler"]
        APScheduler["APScheduler"]
        AddJob["add_job(execute_run)"]
        Cron["schedule_cron"]
    end

    subgraph Runner["execute_run"]
        direction TB
        R1["Fetch Job from DB"]
        R2["Load connections"]
        R3["Create temp Python script"]
        R4["subprocess.run(script, payload)"]
        R5["Parse SUCCESS_RUN / FAILED_RUN"]
        R6["Persist JobRun to DB"]
    end

    subgraph Subprocess["Subprocess Script"]
        S1["json.loads(payload)"]
        S2["build_etl_graph().invoke(state_input)"]
        S3["print SUCCESS_RUN/FAILED_RUN"]
    end

    subgraph DB["Database"]
        Job[(Job)]
        JobRun[(JobRun)]
    end

    APScheduler --> AddJob
    Cron --> AddJob
    AddJob --> R1
    R1 --> Job
    R1 --> R2
    R2 --> R3
    R3 --> R4
    R4 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> R5
    R5 --> R6
    R6 --> JobRun
```

---

## 11. External Integrations

```mermaid
flowchart LR
    subgraph IntegrityCore["IntegrityCore"]
        Agents["Agents"]
        Meta["Metadata"]
        Exec["Executor"]
    end

    subgraph External["External Systems"]
        Snowflake[(Snowflake)]
        LiteLLM[LiteLLM]
        Z3[Z3 Solver]
        GCP[GCP Secret Manager]
    end

    subgraph Storage["Storage"]
        SQLite[(SQLite)]
        DuckDB[(DuckDB)]
        LanceDB[(LanceDB)]
    end

    Agents -->|"completion"| LiteLLM
    Agents -->|"embeddings"| LiteLLM
    Meta -->|"introspect"| Snowflake
    Meta -->|"vector search"| LanceDB
    Meta -->|"catalog"| DuckDB
    Exec -->|"execute, compile"| Snowflake
    Agents -->|"verify"| Z3
    Exec -->|"secrets"| GCP
```

---

## 12. Module Dependency Graph

```mermaid
flowchart TB
    subgraph Entry["Entry Points"]
        CLI["cli.py"]
        API["ui/api.py"]
        Runner["scheduler/runner.py"]
    end

    subgraph Agents["agents"]
        Graph["graph.py"]
    end

    subgraph Core["core"]
        Grounding["grounding.py"]
        Verifier["verifier.py"]
    end

    subgraph Adapters["adapters"]
        Connections["connections.py"]
        Executor["executor.py"]
        Cloud["cloud.py"]
    end

    subgraph Metadata["metadata"]
        Manager["manager.py"]
        Introspectors["introspectors.py"]
        Models["models.py"]
    end

    subgraph DB["db"]
        Models["models.py"]
        Engine["engine.py"]
        GoldQueries["gold_queries.py"]
    end

    CLI --> Graph
    API --> Graph
    API --> Connections
    API --> Manager
    Runner --> Graph
    Runner --> Connections

    Graph --> Grounding
    Graph --> Verifier
    Graph --> Connections
    Graph --> Executor
    Graph --> GoldQueries

    Grounding --> Manager
    Manager --> Introspectors
    Manager --> Models
    Executor --> Connections
    Introspectors --> Connections
```

---

## Usage

Copy any diagram block into a Mermaid-compatible viewer (e.g. [Mermaid Live Editor](https://mermaid.live), GitHub, Notion, Confluence) to render.
