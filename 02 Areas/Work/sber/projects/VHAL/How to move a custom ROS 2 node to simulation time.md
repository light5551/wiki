Step-by-step guide for making your own ROS 2 node use **simulation time** (`/clock`) instead of wall-clock time when running against Virtual HAL.

Use this when your node subscribes to simulator topics (camera images, observer, TF, odometry, …) and you need its timestamps, timers, and time-sync logic to follow the simulation instead of the machine's real clock.
## Background: how sim time works here

- The simulator publishes `/clock` (sim time) **every physics step**. It is on by

default — the Makefile forces `--enable_clock_sync`, and `clock_sync.py` publishes
`/clock` via `rclpy` (see `source/ros2_connector/ros2_connector/ros2/core/clock_sync.py`).

This is the **producer** side and is already handled for you.
- A node only *consumes* sim time when its parameter **`use_sim_time=True`**. With that
set, `rclpy`'s `node.get_clock().now()` returns sim time and ROS timers follow it.
- The Kit path pushes `use_sim_time true` only to a **hardcoded node list**
(`kit/run.py`): `adapter_tui`, `control_node`, `robot_state_publisher_xmeta`, `rviz2`,
`robot_state_publisher`. **Your custom node is not in that list**, so you must set `use_sim_time` yourself.
## Prerequisites
- Virtual HAL running with clock sync enabled (the default: `make vhal` / `./vhal`).
- ROS 2 environment sourced (`/opt/ros/jazzy/setup.bash` + your workspace).
- A custom node that subscribes to simulator topics.
## Step 1 — Set `use_sim_time=True` on your node
This is the robust, node-local way (independent of the hardcoded push list). Pick one:
**Launch file:**
```python

Node(

package="my_pkg",

executable="my_node",

parameters=[{"use_sim_time": True}],

)

```
**CLI:**
```bash

ros2 run my_pkg my_node --ros-args -p use_sim_time:=true

```
  
  ### Step 2 — Use the node clock for all timestamps
In the node code, take time from the node's clock — never from wall-clock APIs.

```python

# Correct — follows sim time when use_sim_time=True

now = self.get_clock().now()

msg.header.stamp = now.to_msg()

```
Avoid:
- `time.time()`, `datetime.now()` — wall clock.
- `rclpy.clock.Clock()` created by hand — defaults to a system clock; use
`self.get_clock()` (the node's clock) instead.
ROS timers (`self.create_timer(...)`) automatically follow sim time once `use_sim_time` is set, so no change is needed there.
## Step 3 — Fix time-dependent logic (image + observer sync)
Publishers already stamp `header.stamp` in sim time, so time synchronization keeps
working. Just make sure any comparison uses the sim clock:
```python

from message_filters import ApproximateTimeSynchronizer, Subscriber

img_sub = Subscriber(self, Image, "/left_eye/image")
obs_sub = Subscriber(self, Observer, "/observer")
sync = ApproximateTimeSynchronizer([img_sub, obs_sub], queue_size=10, slop=0.05)

sync.registerCallback(self.on_pair)

```
- **"Message age" / staleness checks:** compute age as
`self.get_clock().now() - Time.from_msg(msg.header.stamp)`. Using wall clock here
makes every message look stale when the sim runs slower than real time.
- **TF lookups:** pass a sim-time stamp (or `Time()` for "latest"); the buffer is populated in sim time.
  
## Step 4 — (Optional) Add the node to the simulator's push list

Belt-and-suspenders: have the simulator also push the param. Add your node name to the list in `source/ros2_connector/ros2_connector/kit/run.py`:

```python
clock_sync = ClockSynchronizer(
	env,
	node_names=[
			"adapter_tui",
			"control_node",
			"robot_state_publisher_xmeta",
			"rviz2",
			"robot_state_publisher",
			"my_node", # <-- your custom node
	],
	node=sim_node,
)

```

Note: this uses `ros2 param set` in a subprocess with a 5 s timeout and **silently** fails if the node is not up yet** — so it is best-effort. Prefer Step 1; use this only as a fallback.

## Step 5 — Verify
Run the bundled checker:
```bash
./scripts/check-sim-time.sh
```
It confirms `/clock` is live and prints a per-node `use_sim_time` table — your node should show **`True` / OK**.
Manual spot-checks:
```bash
# /clock is being published
ros2 topic echo /clock --once # your node reports sim time
ros2 param get /my_node use_sim_time # expect: Boolean value is: True
```
Behavioural check: pause the simulation. A correctly configured node's timers and `get_clock().now()` stop advancing; a wall-clock node keeps ticking.

## Switch a running node at runtime (no restart)
You can move an **already-running** node from wall time to sim time without restarting

it — just set the parameter live:
```bash
ros2 param set /my_node use_sim_time true # switch to sim time
ros2 param set /my_node use_sim_time false # switch back to wall time
```

The moment the parameter flips, `rclpy`/`rclcpp`'s internal `TimeSource` reacts, subscribes to `/clock`, and every `self.get_clock().now()` and ROS timer in that node starts reporting sim time. This is exactly how the simulator's `enable_sim_time()` (`clock_sync.py`) drives the external nodes — it runs `ros2 param set` against nodes that are already up.

Caveats for the live switch:
- **Only works if the node reads time via the node clock** (`self.get_clock().now()`). Code using `time.time()`, `datetime.now()`, or a hand-made `rclpy.clock.Clock()` stays on wall clock — see Step 2.
- **Backward time jump.** Wall clock is a large epoch (~1.7e9 s); sim time starts near 0. At the instant of the switch the node's clock jumps *backward*, which can fire/skip a timer for one cycle, make TF throw "extrapolation into the past" or flush its buffer, and make any cached "last timestamp" go negative. Nodes usually recover within a step or two, but prefer setting `use_sim_time` **before** the node starts processing data (Step 1) when you can.
- **Custom parameter callbacks.** If the node registered its own `add_on_set_parameters_callback`, make sure it does not reject `use_sim_time`, or the live `param set` silently fails.

