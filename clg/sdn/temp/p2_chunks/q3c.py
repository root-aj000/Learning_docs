section = """---

## Q3c) Mininet: Explain its Basic Commands

### 7.1 What is Mininet?

Mininet is a network emulation platform that creates realistic virtual networks on a single machine (typically a Linux host) by instantiating lightweight virtual Ethernet network namespaces as host nodes, Open vSwitch instances as network switches, and TCP/UDP connections with configurable bandwidth, delay, jitter, and packet loss as network links. Developed primarily by researchers at Stanford University and released as open-source software under the BSD license, Mininet has become the most widely adopted tool for SDN research, teaching, and development, enabling network engineers and researchers to prototype, test, and validate network applications, protocols, and topologies without requiring physical network hardware.

Mininet's fundamental design principle is lightweight virtualization: rather than requiring a cluster of physical machines to emulate a network, Mininet creates virtual network nodes as Linux network namespaces—which provide isolated, full Linux TCP/IP stacks running as processes on the host system—and connects them through virtual Ethernet (veth) pairs or through Open vSwitch virtual bridges. This approach enables a single laptop or workstation to emulate a complete multi-switch, multi-host network topology—including hundreds of nodes—with realistic network behavior that faithfully represents the behavior of physical network hardware. Because Mininet's virtual nodes run real, unmodified Linux network stacks and real network applications, experiments conducted in Mininet faithfully replicate the behavior of the same applications running over physical network infrastructure.

```
+---------------------------------------------------------------+
|              MININET VIRTUAL NETWORK ARCHITECTURE               |
+---------------------------------------------------------------+
|                                                               |
|  PHYSICAL LINUX HOST MACHINE                                  |
|  +--------------------------------------------------------+   |
|  |                                                        |   |
|  |  +-----------+  +-----------+  +-----------+           |   |
|  |  | Network   |  | Network   |  | Network   |           |   |
|  |  | NS: host1 |  | NS: host2 |  | NS: host3 |  ...     |   |
|  |  | (Full     |  | (Full     |  | (Full     |           |   |
|  |  |  Linux    |  |  Linux    |  |  Linux    |           |   |
|  |  |  TCP/IP)  |  |  TCP/IP)  |  |  TCP/IP)  |           |   |
|  |  +-----+-----+  +-----+-----+  +-----+-----+           |   |
|  |        |              |              |                   |   |
|  |  +-----v--------------v--------------v------+            |   |
|  |  |        Open vSwitch (virtual)            |            |   |
|  |  |        s1 (OVS bridge)                   |            |   |
|  |  +-----+--------------+--------------+------+            |   |
|  |        |              |              |                   |   |
|  |  +-----v-----+  +-----v-----+  +-----v-----+            |   |
|  |  | veth pair |  | veth pair |  | veth pair |            |   |
|  |  +-----------+  +-----------+  +-----------+            |   |
|  |                                                        |   |
|  +--------------------------------------------------------+   |
|                                                               |
|  KEY COMPONENTS:                                             |
|  - Linux Network Namespaces: Isolated network stacks          |
|  - veth pairs: Virtual Ethernet cables                       |
|  - Open vSwitch: Virtual switch with OpenFlow support        |
|  - TC (traffic control): Bandwidth, delay, jitter, loss      |
|                                                               |
+---------------------------------------------------------------+
```

### 7.2 Mininet Architecture: Core Components

**Network Namespaces:** Mininet leverages Linux kernel network namespaces as the virtualization mechanism for host nodes. Each network namespace is an isolated copy of the Linux network stack with its own routing tables, ARP tables, firewall rules (iptables/nftables), network interfaces, and process space. Network namespaces provide process-level isolation—processes running within one namespace cannot see or interact with network interfaces in another namespace, and each namespace has its own loopback interface and can have its own virtual Ethernet interfaces. This isolation is precisely equivalent to the isolation provided by physically separate hosts on a network, making Mininet's virtual hosts functionally indistinguishable from real hosts for experimentation purposes.

**Virtual Ethernet Pairs (veth):** A Linux veth pair is a pair of interconnected virtual Ethernet network interfaces implemented in the Linux kernel. When a packet is transmitted through one end of the veth pair, it is received by the other end. Mininet uses veth pairs to connect host network namespaces to OVS virtual switches, effectively creating virtual network cables between virtual nodes. Each veth interface is assigned to a specific network namespace (the host's namespace) on one end, while the other end is connected to an OVS bridge port within the root network namespace.

**Open vSwitch (OVS):** Mininet uses Open vSwitch as its virtual switching substrate, providing the Layer 2 and Layer 3 forwarding functionality, the OpenFlow protocol support that enables SDN controller integration, the Spanning Tree Protocol (RSTP) support for loop prevention in bridged topologies, and the QoS and traffic shaping capability for emulating link bandwidth and delay characteristics. OVS in Mininet operates as a userspace switch daemon (ovs-vswitchd) that processes packets through a flow table pipeline, applying OpenFlow rules and standard switching behavior.

**Traffic Control (TC):** For emulating realistic network link characteristics, Mininet uses the Linux kernel's traffic control (tc) subsystem, which permits network administrators to impose queuing disciplines (qdiscs) that simulate specific bandwidth limits, propagation delays, packet jitter, and packet loss on virtual links. By configuring HTB (Hierarchical Token Bucket) qdiscs with appropriate rate and burst parameters on veth interfaces, Mininet can precisely simulate the behavior of physical network links ranging from slow 56 kbps serial connections to 400 Gbps data center interconnects. The tc netem (network emulator) qdisc provides additional simulation capabilities for random packet loss, packet duplication, packet reordering, and correlated packet loss patterns that simulate real-world network impairments.

### 7.3 Installing and Running Mininet

**Installation:** Mininet is primarily distributed as a Debian/Ubuntu package and can be installed on Ubuntu 18.04, 20.04, 22.04, or newer LTS releases through the standard package manager. Alternatively, Mininet can be installed from source by cloning the Mininet git repository and running the installation script. For demonstration, development, and teaching purposes, Mininet provides an optimized installation that installs the Open vSwitch kernel module, OVS userspace utilities, the Mininet Python API, and example applications in a single operation. The Mininet VM—a pre-built Ubuntu virtual machine appliance—offers the simplest deployment path for Windows and macOS users, who can download a pre-configured VM image, import it into VirtualBox or VMware, and run Mininet within the guest VM.

**Verification:** After installation, the `mn --version` command should display the installed Mininet version, the Open vSwitch version, and the Python version. The `ovs-vsctl --version` and `ovs-ofctl --version` commands provide version information for Open vSwitch components.

### 7.4 Mininet CLI Commands

Once a Mininet topology is running, the Mininet Command-Line Interface (CLI) provides an interactive shell through which the user can execute commands to interact with the virtual network, generate traffic, modify link parameters, install OpenFlow flow rules, and diagnose network behavior.

**Node and Link Inspection:**

`nodes`: Lists all nodes in the current Mininet topology, including switches, hosts, and the controller.
`net`: Displays the topology in ASCII art format, showing all links between nodes.
`dump`: Prints information about all nodes including their interfaces, IP addresses, MAC addresses, and DPIDs (for switches).
`intfList <node>`: Lists the interfaces of a specific node along with their associated virtual Ethernet pair and peer interface information.
`links`: Displays all links in the topology with their current status and parameters.

**Link Control:**

`link <node1> <node2>`: Toggles the state of the specified link, bringing it down if it was up and bringing it up if it was down. This command is useful for simulating link failures in SDN failover experiments.
`link <node1> <node2> up`: Explicitly brings a link up.
`link <node1> <node2> down`: Explicitly brings a link down.
`py net.configLinkStatus('<node1>', '<node2>', 'down')`: From the Mininet Python API, programmatically configures link status (useful in automated test scripts).

**Traffic Generation and Testing:**

`pingall`: Pings all hosts against all other hosts in the topology. This is the canonical Mininet command for verifying basic network connectivity across the entire topology and is frequently used as the first test after topology creation.
`ping <host1> <host2>`: Pings one host from another, generating ICMP echo request/reply traffic. Useful for testing specific connectivity paths and verifying routing behavior.
`iperf <host1> <host2>`: Runs iperf performance testing between two hosts, measuring achievable TCP throughput and UDP performance between the specified endpoints.
`iperfudp <host1> <host2> <bw> <time>`: Runs iperf in UDP mode with a specified bandwidth and duration.
`iperfserver <host>`: Starts an iperf server daemon on the specified host, enabling multiple sequential or concurrent performance tests.
`hping3 <target> <options>`: Uses hping3 to generate custom TCP, UDP, or ICMP packet streams with configurable source addresses, port numbers, packet sizes, and rates. Useful for testing firewall rules, rate limiters, and DoS protection behaviors.

**OpenFlow Flow Rule Management:**

`sh ovs-ofctl dump-flows <switch>`: Executes the Open vSwitch OpenFlow control tool to display all flow rules currently installed in the specified switch's flow tables. This command is essential for verifying that flow rules installed by the SDN controller or through static flow pushers are correctly installed and matching the expected traffic patterns.
`sh ovs-ofctl add-flow <switch> <flow_spec>`: Manually adds an OpenFlow flow rule to a specific switch. The flow specification follows standard OpenFlow flow syntax: `in_port=<port>,actions=output:<out_port>`, `dl_type=0x0800,nw_src=10.0.0.1,actions=drop`, `tcp,tp_dst=80,actions=CONTROLLER`. This command enables rapid experimentation with flow-based forwarding without requiring controller application code.

```
Example Mininet CLI Session:

$ sudo mn --topo single,3 --mac --controller remote
*** Creating network
*** Adding controller
*** Adding hosts:
h1 h2 h3
*** Adding switches:
s1
*** Adding links:
(h1, s1) (h2, s1) (h3, s1)
*** Configuring hosts
h1 h2 h3
*** Starting network
*** Starting CLI:
mininet> nodes
available nodes are:
c0 h1 h2 h3 s1
mininet> net
h1 -> s1 -> h2
h2 -> s1 -> h1
h2 -> s1 -> h3
h3 -> s1 -> h2
h3 -> s1 -> h1
h1 -> s1 -> h3
mininet> h1 ping -c 3 h3
PING 10.0.0.3 (10.0.0.3) 56(84) bytes of data.
64 bytes from 10.0.0.3: icmp_seq=1 ttl=64 time=0.024ms
64 bytes from 10.0.0.3: icmp_seq=2 ttl=24 time=0.032ms
64 bytes from 10.0.0.3: icmp_seq=3 ttl=64 time=0.019ms
--- 10.0.0.3 ping statistics ---
3 packets transmitted, 3 received, 0% loss
mininet> sh ovs-ofctl dump-flows s1
NXST_FLOW reply (xid=0x4):
 cookie=0x0, duration=3.42s, table=0, n_packets=3, n_bytes=258,
   ip,nw_src=10.0.0.1,nw_dst=10.0.0.3 actions=output:3
 cookie=0x0, duration=3.40s, table=0, n_packets=3, n_bytes=258,
   ip,nw_src=10.0.0.3,nw_dst=10.0.0.1 actions=output:1
mininet> link s1 h1 down
mininet> pingall
*** Ping: testing ping reachability
h2 -> h3 X
h3 -> h2 X
*** Results: 50% dropped
mininet> link s1 h1 up
mininet>
```

### 7.5 Mininet Python API: Topology Definition

Beyond the CLI, Mininet's most powerful feature is its Python API, which permits programmatic definition of network topologies, custom node types, link characteristics, and experiment automation. A Mininet topology is defined by subclassing the `Topo` class and implementing a `build()` method that calls `addSwitch()`, `addHost()`, and `addLink()` methods to specify the topology structure.

```
Mininet Python API - Custom Topology:

#!/usr/bin/python
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSController, OVSSwitch, Host
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

class LinearTopology(Topo):
    def build(self, n=4):
        # Add switches in a linear chain
        switches = []
        for i in range(n):
            switch = self.addSwitch(f's{i+1}')
            switches.append(switch)

        # Connect switches in a line
        for i in range(n-1):
            self.addLink(switches[i], switches[i+1],
                         cls=TCLink, bw=100, delay='2ms')

        # Add one host per switch
        for i, switch in enumerate(switches):
            host = self.addHost(f'h{i+1}')
            self.addLink(host, switch, cls=TCLink, bw=1000)

def run():
    topo = LinearTopology(n=5)
    net = Mininet(topo=topo, controller=OVSController,
                  link=TCLink, switch=OVSSwitch)
    net.start()
    print("=== Network started ===")
    print(f"Switches: {[s.name for s in net.switches]}")
    print(f"Hosts: {[h.name for h in net.hosts]}")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
```

### 7.6 Advanced Mininet Features

**Remote Controller Integration:** Mininet supports connecting virtual switches to external SDN controllers (RYU, Floodlight, ONOS, OpenDaylight) running on the host machine or on a separate physical or virtual machine. The `--controller=remote` command-line option configures all Mininet switches to connect to an external controller at a specified IP and port (the default OpenFlow port is 6633 or 6653).

**Custom Topology Plugins:** Mininet's `topo` module provides predefined topology generators: `SingleSwitchTopo` (one switch, n hosts), `LinearTopo` (linear chain of n switches with one host per switch), `TreeTopo` (k-ary tree topology), `TunnelTopo` (tunnels between hosts for VxLAN/GRE emulation), and `NanoTopo` (nanosecond-resolution timing for hardware testbed integration). Custom topologies can be built by subclassing `Topo` and implementing the `build()` method.

**CLI Extensions and Custom Commands:** The Mininet CLI supports custom command extensions through Python: developers can register custom CLI commands using the `CLI` class's extension mechanism, enabling experiment-specific diagnostic commands to be integrated directly into the interactive Mininet session.

### 7.7 Conclusion

Mininet's combination of lightweight virtualization, realistic network emulation, comprehensive Python API, OpenFlow integration, and open-source licensing has made it the standard tool for SDN research, education, and development worldwide. Understanding Mininet's basic commands and Python API—the `nodes`, `net`, `dump`, `pingall`, `iperf`, `links`, `link`, and `sh ovs-ofctl` commands, alongside the core `Topo` class methods and the `TCLink` link emulation mechanism—provides the essential skill foundation for conducting network experiments, testing SDN applications, validating protocol implementations, and demonstrating networking concepts in reproducible, scriptable, and shareable experimental environments.

"""

out_path = "/home/aj-000/aj/Learning_docs/clg/sdn/answers/answer2.md"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(section)
print(f"Appended Q3c to {out_path}")
