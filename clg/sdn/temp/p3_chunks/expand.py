import re

with open('/home/aj-000/aj/Learning_docs/clg/sdn/answers/answer3.md') as f:
    content = f.read()

# Short section expansions appended immediately after section heading
expansions = {
    'Q1c': """
### 3.4 Traffic Classes and Engineering Requirements

Data center traffic is classified into distinct classes:

1. **Latency-Sensitive Mouse Flows**: Database queries, API calls, interactive requests. Sub-millisecond to low-millisecond latency required. TE applies priority queuing and path selection to minimize delay.

2. **Elephant Flows**: Big data shuffle, ML training sync, backup replication. 10-100+ Gbps sustained, latency-tolerant but requiring guaranteed bandwidth. SDN TE detects these through flow size/duration heuristics and steers to dedicated paths.

3. **Burst/Scratch Traffic**: Dev environment transfers, CI/CD artifacts. Rate-limited and scheduled to prevent interference with latency-sensitive flows.

4. **Storage Traffic**: Distributed replication, erasure coding rebuilds. Consistent bandwidth with bounded latency, isolated from compute-facing East-West traffic.

SDN-based TE distinguishes these classes through real-time flow statistics and DSCP marking, applying differentiated treatment per class.""",

    'Q2c': """
### 6.3 Quantitative Demand Metrics

**Throughput**: Hyperscale data centers support aggregate East-West fabric throughput measured in petabits per second. Individual servers require 100-400 GbE; AI/ML clusters use NVLink at 900 GB/s or InfiniBand HDR at 200 Gbps. Bisection bandwidth must scale linearly with server count via leaf-spine topologies.

**Power Density**: 20-40 kW per rack for conventional compute; 40-100+ kW per rack for GPU/AI clusters. Network port density of 48-96 ports per ToR must match server NIC count (2-4 conventional, 8+ AI cluster).

**Growth Rates**: Cloud providers report 30-50% annual server count growth, 40-60% aggregate traffic growth. Infrastructure must scale over 10-15 year facility lifespans through modular design enabling incremental power, cooling, and network expansion.""",

    'Q3b': """
### 8.4 Programming Models: Imperative vs Declarative vs Intent-Based

**Imperative**: Application specifies exact flow rules per switch. Maximum control but requires device-level knowledge.

**Declarative**: Application specifies desired state without how-to. Example: ONOS Intent Framework expressing "Host A can reach Host B" translated into optimized flow rules across the fabric.

**Intent-Based Networking (IBN)**: Operator declares outcomes or constraints such as "guarantee 10 Gbps between app and DB tiers." Controller continuously monitors and autonomously corrects deviations. Closed-loop automation with feedback control.

### 8.5 Flow Table Pipeline Programming

Modern OpenFlow switches process packets through sequential flow table pipelines where Table 0 classifies traffic, Table 1 applies ACLs and security policies, and Table 2 determines forwarding actions. Key programming considerations: table-miss triggering packet-in to controller, priority-based rule matching with higher priority overriding lower, idle/hard timeouts for rule lifecycle management, per-rule counters for monitoring, and cookie fields for application tracking.""",

    'Q4a': """
### 10.5 Network Functions in the Data Plane

Data plane elements implement functions beyond forwarding: L4/L7 load balancing, firewall ACL enforcement at line rate, NAT/CGN IP address translation in hardware, QoS marking with DSCP/CoS classification and queuing, and VXLAN/NVGRE/MPLS encapsulation/decapsulation. The SDN controller programs these through flow rules configuring ACL actions, QoS queue outputs, tunnel header push/pop, and NAT action sets, enabling dynamic network function application without physical reconfiguration.

### 10.6 Management and Monitoring Plane

The management plane provides operational visibility through gNMI streaming telemetry, NETCONF/RESTCONF for device configuration, fault detection via event notifications, alarm management for anomalies including CPU overload and port flapping, and software lifecycle management through gNOI firmware image loading. Integration with northbound applications through event subscription APIs enables real-time dashboards and automated incident response.""",

    'Q4b': """
### 11.5 NBI Implementations Across Controllers

**OpenDaylight RESTCONF**: Data accessed through /restconf/data for configuration and /restconf/operational for state. RPCs via /restconf/operations/{module}:{operation}. Schema discovery at /restconf/operations/yanglib:yanglib. Supports POST/PUT/PATCH/DELETE with JSON and XML. Atomic transactions roll back on partial failure.

**ONOS Intent APIs**: Intent submission via POST /onos/v1/intents with JSON specifying source, destination, bandwidth requirements. ONOS compiler translates intents into optimized flow rules automatically.

**Ryu WSGI REST**: Applications register custom REST endpoints through the WSGI application context for management interfaces alongside event-driven packet processing.

**Floodlight REST /wm namespace**: Topology at /wm/topology/links/json, flow installation at /wm/staticflowentry/json, statistics at /wm/statistics/flow/{dpid}/json, SSE alarms at /wm/events/alarm.

### 11.6 NBI Security Mechanisms

Production NBIs enforce authentication through OAuth2 bearer tokens, JWT assertions, and mTLS client certificates. Authorization uses RBAC with granular permissions for read-only, operator, and admin roles. Multi-tenant data segregation filters API responses by tenant and rejects cross-tenant queries. Rate limiting protects the controller from API abuse and denial-of-service attacks.""",

    'Q4c': """
### 12.3 P4: Programmable Packet Processors

P4 is a domain-specific language for programming packet forwarding pipelines where developers define custom header types, parsers, match-action tables, and reusable components beyond OpenFlow's fixed action set. The toolchain includes p4c as the reference compiler, BMv2 as the software switch for research, Tofino ASIC SDK for production hardware, and PTF for testing. P4 enables in-network telemetry, custom load-balancing hashes, and DDoS detection at line rate within the switch.

### 12.4 eBPF: Extended Berkeley Packet Filter

eBPF extends the Linux kernel with sandboxed programs at XDP executing before the kernel network stack at NIC driver level, at tc-bpf attached to traffic control, and at cgroup/sock for socket-level control. SDN applications use eBPF for high-performance packet processing without kernel bypass, dynamic forwarding logic updated atomically without controller interaction, and integrated telemetry collection with minimal overhead.

### 12.5 Data Serialization Standards in SDN

YANG defines configuration schemas for ietf-interfaces, ietf-routing, and openconfig models used by NETCONF, RESTCONF, and gNMI. JSON serves as the universal REST API payload. Protobuf provides efficient binary serialization for gRPC streaming interfaces. XML remains for legacy NETCONF in telecommunications environments.""",

    'Q5a': """
### 13.7 SBI Selection Criteria

OpenFlow is best for dynamic flow programming and research but limited to flow table control. NETCONF/RESTCONF are best for vendor-neutral device configuration but verbose for streaming. gNMI/gNOI are best for modern streaming telemetry on OpenConfig-capable devices. OVSDB is specific to OVS management. BGP-LS is best for topology collection across domains. P4Runtime enables custom packet processing on programmable hardware. Production controllers routinely use multiple SBIs simultaneously with the SBI abstraction layer presenting unified service interfaces regardless of underlying protocol.

### 13.8 Multi-Protocol SBI Integration

Production SDN controllers use OpenFlow for OVS flow programming, NETCONF for legacy router configuration, BGP-LS for MPLS domain topology, gNMI for modern switch telemetry, and OVSDB for virtual switch management simultaneously. The controller's SBI abstraction layer enables transparent protocol translation.

### 13.9 SBI Security

SBIs require authentication through mutual TLS for NETCONF and gNMI and certificate-based authentication for OpenFlow. Authorization ensures mutual trust between controller and switches. Encryption through TLS for NETCONF and DTLS for OpenFlow. Modern deployments use automated certificate enrollment.""",

    'Q5b': """
### 14.5 Forwarding Stack Comparison

In SDN architectures: Application programs controller through NBI; controller programs switches through SBI; packets follow installed flow rules with global state in controller. In NVF architectures: traffic flows through hypervisor vSwitch to VNF VM then through vSwitch to next hop; each VNF independently implements forwarding logic with no centralized controller.

### 14.6 Operational Velocity Comparison

Adding firewall rules: SDN via controller API in seconds; NVF through VNFM reconfiguration per instance. Scaling bandwidth: SDN adjusts flow rules/ECMP weights; NVF adds instances via VNFM. New service deployment: SDN designs path programs flow rules; NVF packages VNF, creates NSD, instantiates via MANO. Failure recovery: SDN reroutes in under 100ms; NVF replaces instances in 30 seconds to 5 minutes.

### 14.7 Compounded Business Value

SDN delivers efficiency, operational velocity, and management simplicity. NVF delivers service agility, cost reduction, and vendor independence. Organizations implementing both achieve compounded benefits from programmatic control over virtualized services on elastic infrastructure.""",

    'Q5c': """
### 15.9 VNF Packaging and Distribution Standards

VNFs are packaged per ETSI specifications as VNF Packages containing software images, VNFD in YAML or TOSCA format defining VDUs, connection points, lifecycle scripts, monitoring requirements, scaling rules, and availability models, plus ancillary artifacts. The VNFD drives all MANO operations enabling fully automated VNF lifecycle management.

### 15.10 vCPE Service Chain Example

A residential broadband vCPE chain: customer requests service via portal triggering OSS service order to NFVO; NFVO locates vCPE NSD defining vCPE VNF, firewall VNF, NAT VNF, and IGMP VNF sequence; VNFMs instantiate on NFVI; SDN controller programs forwarding path from customer port through firewall to NAT to Internet; VNFM monitors and scales NAT on subscriber growth; on cancellation NFVO orchestrates teardown and resource reclamation.""",

    'Q6a': """
### 16.8 Containerized Network Functions Evolution

CNFs represent the cloud-native NFV trajectory. Containers share host kernels achieving lower overhead with instantiation in seconds. CNFs use OCI container images and Helm charts orchestrated by Kubernetes operators as VNFM equivalents. ETSI NFV Release 3 formally recognizes CNFs with container-VDU types. CNFs achieve 2 to 5 times better resource density than VM-based VNFs with lifecycle management aligned to CI/CD pipelines. Service mesh integration provides traffic management, mTLS, and observability.""",

    'Q6b': """
### 17.9 Integration and Testing Complexity

Production NFV requires testing across NFVO-VNFM at Ve-Vnfm, VNFM-VIM at Vi-Vnfm, VIM-NFVI API validation, SDN-MANO integration, and OSS/BSS-MANO integration for service order fulfillment. ETSI I-Test and OSM provide reference implementations but multi-vendor integration remains the dominant time-to-production challenge.

### 17.10 NVF Security Challenges

Security challenges include hypervisor VM escape vulnerabilities, Spectre/Meltdown side-channel attacks on shared CPU, MANO as high-value attack target, VNF supply chain integrity across multiple vendors, and isolation assurance verifying VXLAN/VLAN in multi-tenant NFVI.

### 17.11 Legacy OSS/BSS Integration

Legacy telecommunications OSS/BSS systems with decades-old architectures require comprehensive adapter layers, data model transformations, and workflow re-engineering to integrate with modern NFV-MANO APIs—a significantly underestimated adoption challenge.""",

    'Q6c': """
### 18.2.1 Performance Requirements

In-line functions require four critical properties: wire-rate forwarding at full line speed without packet loss, deterministic latency bounded within SLA limits, burst absorption capability without tail latency spikes, and zero packet corruption through all processing stages.

### 18.2.2 Acceleration Technologies

Wire-rate in-line forwarding uses DPDK poll-mode drivers at 50-100+ Gbps on x86, SR-IOV VFs bypassing hypervisor at 10-20 microsecond latency, SmartNIC cryptographic and flow processing offload, CPU pinning eliminating scheduler interference, huge pages reducing TLB misses, and NUMA-local allocation matching memory and NIC to vCPU NUMA nodes.

### 18.2.3 Bypass TAPs

Hardware bypass TAPs provide electrical fail-open for in-line security appliances maintaining traffic flow during appliance failure. SDN-based approaches trigger flow rule rerouting to healthy instances within milliseconds through health monitoring.""",

    'Q7a': """
### Orchestration Stack Layers

The complete data center orchestration stack operates across: Business UX layer with self-service portals and billing; Service orchestration layer using NFVO, OpenStack Heat, Kubernetes, and Terraform Enterprise; Infrastructure automation layer with Ansible, NAPALM, and Helm; and Physical infrastructure of servers, switches, storage, load balancers, and firewalls. Modern orchestration spans Day-1 deployment and Day-2 operations including automated patching, certificate lifecycle management, backup orchestration, and continuous security posture compliance.""",

    'Q7c': """
### Contrail Deep Dive

Contrail Control Nodes implement full MP-BGP routing logic including route target import/export filtering, route aggregation, and policy-based routing to compute forwarding paths distributed to vRouter agents through XMPP. The vRouter in DPDK mode achieves near-line-rate VXLAN encapsulation at 100 Gbps on x86 hardware. Multi-site DCI uses EVPN-based multi-homing, VXLAN stretched subnets across geographically distributed data centers, and BGP route distribution across Contrail domains enabling live VM migration across sites.

### Apstra Intent-Based Networking

Apstra enables declarative intent specification automatically translating to multi-vendor device configurations across Arista, Cisco, Juniper, and NVIDIA switches with continuous validation and autonomous remediation through agentless architecture.""",

    'Q8b': """
### ODL Technical Deep Dive

MD-SAL implements three inter-related data stores: Config Datastore for operator-declared desired state, Operational Datastore for actual current state as reported by southbound plugins, and the Binding-Aware layer providing typed programmatic interfaces with transaction semantics and change notification. The ODL plugin ecosystem supports OpenFlow for flow management, NETCONF for configuration, OVSDB for Open vSwitch, BGP-LS for topology, PCEP for path computation, P4Runtime for programmable switches, gNMI for streaming telemetry, and SNMP for legacy devices. ODL clustering uses the Distributed Manager framework with Raft consensus for data consistency in scale-out deployments.""",

    'Q8c': """
### BWC Operational Model

BWC maintains a bandwidth inventory catalog of all available network paths with capacities and current commitments. Reservation requests specifying source, destination, bandwidth, start time, and duration are evaluated by an admission control engine against existing calendar reservations. Committed reservations are stored in a calendar database with efficient time-range queries. At reservation start, the SDN controller enforces the commitment through OpenFlow meter tables, HTB queues, or DiffServ marking. At reservation end, capacity is released back to the available pool. The ODL BWC project implements this through YANG models, RESTCONF endpoints, and integration with topology and meter management services."""
}

# Check current word count
sections = {}
current_section = None
current_text = []
for line in content.split('\n'):
    if line.startswith('## Q') and ')' in line:
        if current_section:
            sections[current_section] = '\n'.join(current_text)
        current_section = line.strip()
        current_text = [line]
    else:
        current_text.append(line)
if current_section:
    sections[current_section] = '\n'.join(current_text)

# Find short sections
short_sections = {k: v for k, v in sections.items() if len(v.split()) < 700}
print(f"Total sections: {len(sections)}")
print(f"Short sections (<700 words): {len(short_sections)}")
for k in short_sections:
    print(f"  {k[:60]}: {len(short_sections[k].split())} words")
