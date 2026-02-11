# Remote Teleoperation

Split bimanual SO101 teleoperation across two machines: **leader arms + display** on one machine (the operator station) and **follower arms + cameras** on another (the robot station). This enables demonstrations where the operator is not physically next to the robot.

## Problem Statement

The current `data_taking/teleop.py` runs everything on a single machine: it reads leader arm positions over serial, writes goal positions to follower arms over serial, reads camera frames, and optionally displays via Rerun — all in a tight 30 Hz loop. For remote demos, we need to physically separate the leader arms from the followers while maintaining responsive control and live video feedback.

**Data flow to split across the network:**

```
OPERATOR STATION                          ROBOT STATION
(leader arms, display)                    (follower arms, cameras)

BiSOLeader.get_action()                   BiSOFollower.send_action(action)
  ↓                                         ↑
12 joint positions (48 bytes)  ──────→   receive & apply
  ↓                                         ↓
receive & display   ←──────   observation + camera frames
  ↓
Rerun visualization
```

---

## Data Rate Analysis

### Joint positions (bidirectional)

Each arm has 6 joints. Bimanual = 12 joints total.

| Direction | Payload | Rate | Bandwidth |
|-----------|---------|------|-----------|
| Commands (operator → robot) | 12 × float64 = 96 bytes + framing | 30 Hz | **~5 KB/s** |
| Observations (robot → operator) | 12 × float64 = 96 bytes + framing | 30 Hz | **~5 KB/s** |

Joint data is negligible. Even with JSON overhead and timestamps, it stays under 20 KB/s each direction.

### Camera video (robot → operator)

| Cameras | Resolution | Codec | Per-frame | At 30 FPS | Total |
|---------|-----------|-------|-----------|-----------|-------|
| 1 | 640×480 | MJPEG | ~30-50 KB | ~1.0-1.5 MB/s | 1.5 MB/s |
| 2 | 640×480 | MJPEG | ~30-50 KB | ~1.0-1.5 MB/s | 3.0 MB/s |
| 3 | 640×480 | MJPEG | ~30-50 KB | ~1.0-1.5 MB/s | 4.5 MB/s |

### Total bandwidth requirement

With 3 cameras: **~5 MB/s sustained** (40 Mbps). This is well within both WiFi and Ethernet capacity.

### Latency budget

The 30 Hz loop has a 33 ms budget per cycle, but the servo motors themselves take 50-200 ms to physically respond to a position command. This means **network latency under ~15 ms is indistinguishable from local operation** — the motors are the bottleneck, not the network.

---

## Network: Tailscale over WiFi vs Direct Ethernet

### Tailscale over WiFi — Recommended for demos

| Factor | Value |
|--------|-------|
| Added latency (same LAN) | 2-5 ms (WireGuard overhead) |
| WiFi latency (802.11ac/ax) | 1-5 ms |
| **Total added latency** | **3-10 ms** |
| Throughput (WiFi 5) | 100-400 Mbps |
| Throughput needed | 40 Mbps |
| Setup complexity | Zero-config (Tailscale handles NAT, auth, encryption) |

**Verdict: Perfectly fine.** 3-10 ms of added latency is invisible given 50-200 ms servo response. The 40 Mbps bandwidth need is well within WiFi 5/6 capacity. Tailscale gives you:
- Encrypted transport for free (WireGuard)
- Stable IPs that survive WiFi reconnects (100.x.y.z)
- Works across subnets, NATs, and even different networks
- No port forwarding or firewall configuration

### Direct Ethernet — Only if WiFi is unreliable

| Factor | Value |
|--------|-------|
| Added latency | <1 ms |
| Throughput | 1 Gbps |
| Setup complexity | Need a cable, static IPs or DHCP |

Direct Ethernet eliminates WiFi jitter but requires physical proximity and a cable. Only worth it if:
- The demo venue has unreliable WiFi (congested conference WiFi, etc.)
- You need absolute minimum latency for some reason
- You're streaming higher resolution video (1080p+)

### Recommendation

**Use Tailscale over WiFi for all demos.** Fall back to direct Ethernet only if you encounter WiFi reliability issues at a specific venue. The code should be transport-agnostic (just uses TCP) so switching is just a matter of which IP address you connect to.

---

## Architecture Comparison

### Option A: Direct ZMQ pub/sub (Recommended)

Two processes communicating over ZMQ sockets. The robot station publishes observations and subscribes to commands; the operator station does the reverse.

```
OPERATOR STATION                              ROBOT STATION
┌─────────────────────┐                       ┌─────────────────────┐
│  leader_client.py   │                       │  follower_server.py │
│                     │   ZMQ PUB (commands)  │                     │
│  BiSOLeader ────────┼──────────────────────→│──→ BiSOFollower     │
│                     │                       │                     │
│  Rerun display  ←───┼──────────────────────←│←── Cameras          │
│                     │   ZMQ PUB (obs+video) │                     │
└─────────────────────┘                       └─────────────────────┘
```

**Pros:**
- ZMQ is already in the codebase (`lerobot/cameras/zmq/`) — proven pattern
- Minimal dependencies (just `pyzmq`, already installed)
- Extremely low overhead — ZMQ PUB/SUB adds <1 ms latency
- `ZMQ_CONFLATE=True` automatically drops stale messages (critical: operator always gets latest frame, not a queue backlog)
- Simple to reason about: two scripts, two sockets, done
- No service discovery needed — you know the IPs via Tailscale
- No daemon processes, no message broker, no infrastructure

**Cons:**
- Manual reconnection handling (ZMQ does auto-reconnect, but you need to handle the gap)
- No built-in QoS beyond CONFLATE
- Custom wire protocol (but it's just JSON, same as existing ZMQ camera code)

**Implementation effort: Small.** Follow the exact pattern from `camera_zmq.py` but extend it to carry joint positions alongside video.

### Option B: ROS 2

Each arm and camera becomes a ROS node. Joint commands sent as `JointState` messages, camera frames as `Image` messages, all over DDS (the ROS 2 middleware).

```
OPERATOR STATION                              ROBOT STATION
┌─────────────────────┐                       ┌─────────────────────┐
│  leader_node        │                       │  follower_node      │
│  display_node       │←──── DDS/UDP ────────→│  camera_node(s)     │
│  (ROS 2)            │                       │  (ROS 2)            │
└─────────────────────┘                       └─────────────────────┘
```

**Pros:**
- Standard robotics middleware with battle-tested transport
- Built-in QoS profiles (reliable vs best-effort, keep-last vs keep-all)
- Service discovery via DDS (automatic, no hardcoded IPs)
- Rich ecosystem of tools (rviz2, ros2 topic echo, rosbag, etc.)
- Standard message types (`sensor_msgs/JointState`, `sensor_msgs/Image`)

**Cons:**
- **Heavy dependency** — ROS 2 installation is large and complex (1+ GB, specific Ubuntu versions)
- **DDS tuning over Tailscale is painful** — DDS uses UDP multicast for discovery, which doesn't work over Tailscale. Requires `ROS_DOMAIN_ID` workarounds, Cyclone DDS XML config, or FastDDS discovery servers. This alone can eat hours of debugging.
- **Latency overhead** — DDS serialization + deserialization adds 1-5 ms vs raw ZMQ
- **Doesn't match the existing codebase** — the project uses LeRobot's own robot/camera abstractions, not ROS. Adding ROS would mean wrapping everything or maintaining a parallel interface.
- **Overkill** — we're sending 12 floats and 3 JPEG streams between exactly 2 machines. We don't need service discovery, QoS policies, or a ROS graph.

**Implementation effort: Large.** Need to install ROS 2 on both machines, write node wrappers around existing LeRobot code, configure DDS for point-to-point over Tailscale, and deal with the impedance mismatch between ROS message types and LeRobot's dict-based interface.

### Option C: gRPC bidirectional streaming

A gRPC service on the robot station with bidirectional streams for commands and observations.

**Pros:**
- Strong typing via protobuf
- Built-in bidirectional streaming
- Good Python support, works well over any TCP connection including Tailscale
- HTTP/2 multiplexing

**Cons:**
- Protobuf compilation step adds build complexity
- More complex than ZMQ for this use case (service definitions, stubs, etc.)
- No equivalent of `ZMQ_CONFLATE` — would need custom logic to drop stale frames
- Not already in the codebase
- gRPC streaming has higher overhead than raw ZMQ PUB/SUB for fire-and-forget patterns

**Implementation effort: Medium.** More boilerplate than ZMQ (proto files, generated stubs, service implementation), and the streaming semantics need careful handling to avoid frame buffering.

### Option D: WebRTC (video) + ZMQ (control)

Use WebRTC for adaptive video streaming and ZMQ for the tiny joint position channel.

**Pros:**
- WebRTC is optimized for real-time video: adaptive bitrate, jitter buffers, FEC
- Would give the best video quality under variable network conditions
- Could potentially expose video to a web browser for additional viewers

**Cons:**
- **Massive complexity increase** — WebRTC signaling, STUN/TURN, codec negotiation
- Two different transport protocols to manage
- Python WebRTC libraries (aiortc) are less mature than ZMQ
- MJPEG at 640x480 is only 1.5 MB/s per camera — there's no need for adaptive bitrate on a LAN

**Implementation effort: Large.** Dramatically overengineered for LAN-based MJPEG streams.

### Decision: Option A — Direct ZMQ

ZMQ is the clear winner:
1. Already proven in the codebase for exactly this kind of thing (camera streaming)
2. Minimal new dependencies
3. Transport-agnostic (TCP works identically over Tailscale, Ethernet, or anything else)
4. CONFLATE mode solves the "stale frame" problem automatically
5. Simplest implementation by far

---

## Detailed Design

### Wire Protocol

Extend the existing ZMQ camera JSON protocol to carry joint data alongside images. Two message types on two separate ZMQ PUB/SUB socket pairs:

**Command channel (operator → robot, port 5555):**
```json
{
    "timestamp": 1706000000.123,
    "action": {
        "left_shoulder_pan.pos": 45.2,
        "left_shoulder_lift.pos": -12.0,
        "left_elbow_flex.pos": 90.1,
        "left_wrist_flex.pos": 5.0,
        "left_wrist_roll.pos": 0.0,
        "left_gripper.pos": 50.0,
        "right_shoulder_pan.pos": 44.8,
        "right_shoulder_lift.pos": -11.5,
        "right_elbow_flex.pos": 89.7,
        "right_wrist_flex.pos": 4.8,
        "right_wrist_roll.pos": 0.1,
        "right_gripper.pos": 49.5
    }
}
```

**Observation channel (robot → operator, port 5556):**
```json
{
    "timestamp": 1706000000.123,
    "observation": {
        "left_shoulder_pan.pos": 44.8,
        "...": "..."
    },
    "images": {
        "left": "<base64-jpeg>",
        "right": "<base64-jpeg>",
        "top": "<base64-jpeg>"
    }
}
```

This is deliberately close to the existing `camera_zmq.py` format so the patterns are familiar.

### Components

#### `follower_server.py` — runs on robot station

```
1. Load config (config.toml — follower ports, camera config)
2. Create BiSOFollower with cameras
3. Connect
4. Start ZMQ:
   - SUB socket on port 5555 (receive commands)
   - PUB socket on port 5556 (publish observations)
5. Main loop (30 Hz):
   a. Non-blocking recv from SUB → if command available, send_action()
   b. get_observation() → read joints + cameras
   c. Encode camera frames as JPEG, base64
   d. Publish observation JSON on PUB
6. Graceful shutdown on Ctrl+C
```

#### `leader_client.py` — runs on operator station

```
1. Load config (config.toml — leader ports, URDF config)
2. Create BiSOLeader (no cameras needed locally)
3. Connect
4. Start ZMQ:
   - PUB socket connecting to robot:5555 (send commands)
   - SUB socket connecting to robot:5556 (receive observations)
5. Init Rerun with URDF + camera panels
6. Main loop (30 Hz):
   a. teleop.get_action() → read leader arms
   b. Publish action JSON on PUB
   c. Non-blocking recv from SUB → latest observation
   d. Display in Rerun (URDF joint viz + camera images)
7. Graceful shutdown on Ctrl+C
```

### Configuration

Uses the existing `config.toml` system via `lib/config.py`. Each machine only needs the sections relevant to its role:

**Robot station config.toml:**
```toml
[follower.left]
port = "/dev/serial/by-id/usb-..."

[follower.right]
port = "/dev/serial/by-id/usb-..."

[camera.left]
path = 6
width = 640
height = 480
fps = 30
fourcc = "MJPG"

[camera.right]
path = 4
width = 640
height = 480
fps = 30
fourcc = "MJPG"

[camera.top]
path = 0
width = 640
height = 480
fps = 30
```

**Operator station config.toml:**
```toml
[leader.left]
port = "/dev/serial/by-id/usb-..."

[leader.right]
port = "/dev/serial/by-id/usb-..."

[urdf]
path = "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
left_offset = [0.0, 0.2, 0.0]
right_offset = [0.0, -0.2, 0.0]
left_rotation = 0.0
right_rotation = 0.0
```

**New remote-specific config (`remote_teleop/config.toml`):**
```toml
[remote]
robot_ip = "100.64.0.2"      # Tailscale IP of robot station
command_port = 5555           # Operator → Robot
observation_port = 5556       # Robot → Operator
jpeg_quality = 85             # JPEG compression quality (1-100)
```

### ZMQ Socket Pattern

Use **PUB/SUB with CONFLATE** on both channels:

- `ZMQ_CONFLATE=True`: subscriber only keeps the latest message, dropping anything older. This is critical — if the network hiccups, we don't want the operator to see a queue of stale frames playing back. We want to skip straight to the current state.
- `ZMQ_SNDHWM=1`: publisher high-water mark of 1, same idea — don't buffer unsent messages.
- Separate sockets for commands vs observations (not REQ/REP) so neither side blocks waiting for the other.

### Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| Network drops | Follower holds last position (no new commands), operator shows stale frame | ZMQ auto-reconnects when network returns |
| Operator quits | Follower holds last commanded position | Follower detects command timeout → optional safety stop |
| Robot quits | Operator sees stale/no observation | Reconnect automatically on restart |
| High latency spike | CONFLATE drops intermediate frames, latest state always shown | Self-recovering |

A command timeout watchdog on the follower server is important for safety: if no command arrives for N seconds (e.g., 2s), the follower should stop accepting stale commands and optionally disable torque.

### File Structure

```
remote_teleop/
├── README.md                 # This file
├── config.toml               # Remote-specific config (robot_ip, ports)
├── follower_server.py        # Runs on robot station
├── leader_client.py          # Runs on operator station
└── protocol.py               # Shared encode/decode for wire messages
```

### Usage

**On the robot station:**
```bash
uv run remote_teleop/follower_server.py
```

**On the operator station:**
```bash
uv run remote_teleop/leader_client.py --robot 100.64.0.2 --display
```

---

## Future Considerations

- **Recording integration**: The follower server could optionally record datasets locally using the existing `data_taking/record.py` infrastructure, since it has access to both the commanded actions and the observations.
- **Multiple viewers**: ZMQ PUB naturally supports multiple subscribers. Additional operator stations could subscribe to the observation feed for spectating.
- **Higher resolution**: If 1080p cameras are needed, consider H.264 streaming instead of MJPEG-over-JSON. The ZMQ transport stays the same, but the encoding changes.
- **Latency monitoring**: Log round-trip timestamps to measure actual end-to-end latency during demos. The timestamp fields in the protocol make this straightforward.
