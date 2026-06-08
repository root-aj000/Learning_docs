import os

out = "/home/aj-000/aj/Learning_docs/clg/sdn/answers/answer5.md"

content = """---

## Q8a) Write short note on Floodlight Controller

### 1. Introduction

The **Floodlight Controller** is an open-source, Java-based SDN controller that emerged in 2012 from **Big Switch Networks** (founded by ex-Stanford SDN researchers Rob Sherwood and Glen Gibb). Floodlight was released under the Apache 2.0 license and became one of the first production-grade, community-driven OpenFlow controllers available for research, education, and commercial deployment. While newer controllers (ONOS, ODL) have since gained greater enterprise prominence, Floodlight played a pivotal role in the early SDN ecosystem and remains widely used in academic and research environments today.

### 2. Floodlight Architecture

Floodlight is built on a **modular, service-oriented architecture** in Java:

**Core Modules:**
- **REST API Module:** Exposes RESTful HTTP API on port 8080 for topology, device, and flow management.
- **OpenFlow Protocol Module:** Handles switch connection management, flow rule installation/deletion, Packet-In/Packet-Out processing.
- **Topology Manager:** Builds and maintains a real-time network graph using LLDP discovery.
- **Device Manager:** Tracks connected devices (MAC addresses, IP addresses, attachment points).
- **Forwarding Module:** Implements Layer-2 MAC learning and flood-and-forward behavior as a reference switching module.
- **Switch Manager:** Manages OpenFlow switch connections and maintains switch-specific state.

**Extensible Module System:**
Custom modules implement the `IFloodlightModule` interface. Modules register event handlers and services in Floodlight's dependency injection framework, enabling third-party extensions without modifying the core controller code.

```
    FLOODLIGHT CONTROLLER ARCHITECTURE

    +------------------------------------------+
    |            Floodlight Core               |
    |  (Module Loader, Dependency Injection,   |
    |   Event Bus, Serialization)              |
    +------------------+-----------------------+
                       |
          +------------+------------+
          |                         |
    +-----v-----+           +-------v-------+
    | Mandatory |           |  Optional     |
    | Modules   |           |  Modules      |
    |           |           |               |
    | - REST    |           | - Static Flow |
    |   API     |           |   Pusher      |
    | - OpenFlow|           | - Firewall    |
    |   Protocol|           | - VTN (Virtual|
    | - Topology|           |   Tenant Net) |
    |   Manager |           | - Web UI      |
    | - Device  |           | - QoS         |
    |   Manager |           | - Packet      |
    | - Forward |           |   Debugger    |
    |   Module  |           |               |
    +-----------+           +---------------+
```

**Figure 8.1:** Floodlight modular service-oriented architecture showing core mandatory modules and optional extensions.

### 3. Key Features

**Virtual Tenant Network (VTN):** Enables creation of isolated virtual networks with defined topology, MAC space, and connectivity. Each VTN maintains its own MAC-to-port mapping table, providing strict isolation between tenants.

**Static Flow Pusher:** Allows persistent installation of OpenFlow flow rules. Rules survive switch disconnections and are reinstalled automatically upon reconnection.

**Firewall Module:** Demonstrates in-controller security enforcement by maintaining a permissive/deny rule database of allowed and blocked flows.

### 4. Floodlight REST API

Floodlight exposes a comprehensive REST API:

```
GET  /wm/topology/links/json           → Network links
GET  /wm/device/                      → Connected devices
GET  /wm/stats/switch/{dpid}/          → Switch statistics
POST /wm/staticflowentry/json          → Install static flow rule
GET  /wm/staticflowentry/json          → List static flow rules
DELETE /wm/staticflowentry/json        → Delete static flow rules
```

Example flow installation:
```bash
curl -X POST -d '{"switch": "00:00:00:00:00:00:00:01",
  "name": "flow-mod-1",
  "priority": "32768",
  "ingress-port": "1",
  "actions": "output=2"}' \
  http://controller-ip:8080/wm/staticflowentry/json
```

### 5. Conclusion

Floodlight Controller represents an important milestone in open-source SDN development. Its modular design, VTN feature, comprehensive REST API, and active community made it the platform of choice for early SDN research and education projects. While newer controllers have superseded it in enterprise deployments, Floodlight's architectural patterns and educational accessibility continue to influence SDN controller design.

"""

with open(out, "a") as f:
    f.write(content)

print("Q8a appended:", len(content), "chars")
