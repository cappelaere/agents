# AIS Agent Architecture

This document describes the architecture of the AIS (Automatic Identification System) Agent, including:

- How the **AIS Agent API** (FastAPI) relates to the **MCP server** and **WatsonX Orchestrate**
- The **OpenAPI contract** (`ais_openapi.yaml`) used for LLM tool-calls
- End-to-end request flows (including auth)
- Deployment on AWS ECS (`gres-cluster-A`)
- Data flow and governance / observability hooks

The goal is to keep **MCP tools**, **FastAPI implementation**, and **WatsonX Orchestrate tool usage** in sync via a single, curated contract.

---

## 1. Components Overview

### 1.1 AIS Agent API (FastAPI MCP Bridge)

- Implemented in `app.py`
- Exposes HTTP endpoints under `/mcp/ais/*` that conform to `ais_openapi.yaml`
- Delegates actual logic to the `ais_agent` module (`ais_agent.py`, `ais_vessel_info.py`, etc.)
- Serves the OpenAPI contract for orchestrators:

  - JSON: `/mcp/openapi.json`
  - YAML: `/mcp/openapi.yaml`
  - Docs: `/mcp/docs` (Swagger), `/mcp/redoc` (ReDoc)

The **server URL** in `ais_openapi.yaml` is:

```yaml
servers:
  - url: http://localhost:8200/mcp/ais
    description: MCP bridge to the local AIS agent (via mcp_server)
````

At runtime, Orchestrate sees paths like:

* `GET /vessels/nearby`
* `GET /vessels/aoi`
* `GET /vessel/info`
* `GET /vessel/photo`
* `GET /vessel/track`
* `GET /aoi`, `GET /aoi/{aoi_id}`
* `GET /vessel/events`
* `GET /vessel/portcalls`
* `GET /portcalls`
* `GET /routing/distance_to_port`
* `GET /routing/vessel_route_to_port`
* `GET /health`

…with the base URL `http://<host>/mcp/ais`.

---

### 1.2 MCP Server & AIS Logic

* `ais_agent.py` implements the **core AIS logic**, including:

  * Fetching vessel info by MMSI/IMO/name (`ais_vessel_info.py`)
  * Querying external AIS provider APIs
  * Normalizing & enriching AIS data
  * Tracing (e.g., Langfuse) via helper utilities (e.g., `trace_start`, `trace_end`)

* The MCP server exposes these capabilities as **tools** to higher-level agent runtimes (not just HTTP).

In this architecture:

* FastAPI is a **bridge** that:

  * Accepts HTTP requests shaped by `ais_openapi.yaml`
  * Calls MCP tools / logic in `ais_agent`
  * Returns normalized JSON responses

---

### 1.3 WatsonX Orchestrate & LLM Tool Use

WatsonX Orchestrate:

* Fetches the AIS Agent contract from `/mcp/openapi.json` or `/mcp/openapi.yaml`
* Registers each path as a **tool** the LLM can call
* Uses `tags`, `summaries`, `descriptions`, and **schemas** from `ais_openapi.yaml`
* Uses `x-llm-usage` metadata to improve tool selection

Example excerpt:

```yaml
x-llm-usage:
  guidance: |
    Use this API to answer questions about vessels, their positions, events, and port calls...
  canonical_examples:
    - user: "Show me nearby vessels around 64.5N, -170W within 50nm"
      tools:
        - name: GET /vessels/nearby
          params: { lat: 64.5, lon: -170, radius_nm: 50 }
```

This metadata makes the spec **LLM-tuned**, not just machine-readable.

---

## 2. API Contract (`ais_openapi.yaml`)

The AIS Agent API contract is defined in `ais_openapi.yaml`.

It defines:

* Endpoint structure & parameters
* Response schemas & examples
* LLM usage guidance metadata
* Sessions, tracing, and invocation patterns

Example header:

```yaml
openapi: 3.0.3
info:
  title: AIS Agent API
  version: 1.0.0
```

---

## 3. High-Level Architecture

### 3.1 Traditional REST Architecture

```mermaid
flowchart LR
    Client["Browser / App / Service"] -->|HTTP| API["REST API Server"]
    API --> DB["Database"]
    API --> External["External Services"]
```

### 3.2 Pure MCP Server

```mermaid
flowchart LR
    LLM["LLM / Agent Runtime"] --> WS["MCP Server (WebSocket)"]
    WS --> LLM
    WS --> Tools["Tools / Functions"]
    WS --> Data["Data Sources"]
```

### 3.3 Hybrid AIS Architecture (What We Run)

```mermaid
flowchart LR
    Orchestrate["WatsonX Orchestrate / Agent"] --> FastAPI["FastAPI AIS Agent (OpenAPI)"]
    FastAPI --> MCP["MCP Server (AIS Tools)"]
    MCP --> Logic["Domain Logic / Vendor APIs / Adapters"]
    Logic --> Data["AIS Data / Cache / DB"]
```

---

## 4. End-to-End Request Flows

### 4.1 Nearby Vessels (`GET /vessels/nearby`)

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant WO as WatsonX Orchestrate
    participant LLM as LLM
    participant ToolRT as Tool Runtime
    participant API as AIS Agent API
    participant MCP as MCP Server
    participant Vendor as External AIS Provider
    participant DB as Cache

    User->>WO: Query: nearby vessels
    WO->>LLM: Provide OpenAPI context
    LLM-->>WO: Select /vessels/nearby tool
    WO->>ToolRT: Construct tool call
    ToolRT->>API: GET /vessels/nearby?lat=64.5&lon=-170&radius_nm=50

    API->>MCP: call_tool("get_vessels_nearby")
    MCP->>DB: Check cache
    alt cache hit
        DB-->>MCP: cached results
    else cache miss
        MCP->>Vendor: API request
        Vendor-->>MCP: Raw AIS data
        MCP->>MCP: Normalize + filter
        MCP->>DB: Store cache
    end

    MCP-->>API: VesselList JSON
    API-->>ToolRT: HTTP 200
    ToolRT-->>WO: Structured tool response
    WO-->>User: Natural language summary + results
```

### 4.2 Vessel Track with Auth

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant WO as WatsonX Orchestrate
    participant ToolRT as Tool Runtime
    participant API as AIS Agent API
    participant AuthAPI as Auth / Secrets
    participant MCP as MCP Server
    participant Vendor as AIS Provider

    User->>WO: Get vessel track
    WO->>ToolRT: Select /vessel/track
    ToolRT->>API: GET /vessel/track?mmsi=...
    API->>AuthAPI: Validate token
    API->>MCP: call_tool("get_vessel_track")
    MCP->>AuthAPI: Fetch provider token
    MCP->>Vendor: Query provider
    Vendor-->>MCP: Track data
    MCP-->>API: Normalized response
    API-->>ToolRT: HTTP 200
    ToolRT-->>WO: Tool result
    WO-->>User: Track summary + recommended actions
```

---

## 5. Deployment on AWS ECS (`gres-cluster-A`)

```mermaid
flowchart TB
    subgraph SaaS["IBM Cloud / WatsonX SaaS"]
        WO["WatsonX Orchestrate\nLLM + Tool Runtime"]
    end

    subgraph AWS["AWS Account"]
        subgraph VPC["VPC"]
            ALB["ALB\nais-agent.example.gov"]

            subgraph ECS["Amazon ECS Cluster\ngres-cluster-A"]
                subgraph AIS["AIS Agent Service"]
                    Task1["Task: ais_agent\nFastAPI + MCP"]
                    Task2["Task: ais_agent\nFastAPI + MCP"]
                end
                Metoc["metoc_agent"]
                Ports["ports_agent"]
                SeaIce["seaice_agent"]
            end

            subgraph DataTier["Data Storage"]
                RDS["PostgreSQL"]
                Redis["Elasticache"]
                S3["S3 Buckets"]
            end
        end

        VendorAIS["External AIS Provider API"]
    end

    WO -->|HTTPS| ALB
    ALB --> AIS
    AIS --> RDS
    AIS --> Redis
    AIS -->|API Key| VendorAIS
```

---

## 6. Data Flow & Governance

```mermaid
flowchart LR
    User["User"] --> WO["WatsonX Orchestrate"]
    WO -->|Tool Calls| API["AIS Agent API"]
    API --> MCP["MCP Tools"]
    MCP --> Vendor["AIS Provider API"]
    MCP --> Data["RDS / Redis / S3"]
    API --> Obs["Observability\nLangfuse / OpenTelemetry"]
    Obs --> Gov["WatsonX Governance"]
```

---

## 7. Source of Truth & Workflow

| Artifact            | Role             | Maintained By                   |
| ------------------- | ---------------- | ------------------------------- |
| `ais_openapi.yaml`  | **API contract** | Edited manually (authoritative) |
| `app.py`            | FastAPI server   | Implementation follows contract |
| `ais_agent.py`      | MCP tool logic   | Implements tool calls           |
| `/mcp/openapi.yaml` | Served spec      | Mirrors contract file           |

### Recommended Workflow

1. Design & modify **`ais_openapi.yaml` first**
2. Generate/update FastAPI signatures to match
3. Implement MCP logic behind routes
4. Test with `curl` + Orchestrate tool calls
5. Monitor using Langfuse / Governance telemetry
6. Iterate spec based on usage

---

**End of Document**
