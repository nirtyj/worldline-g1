"""GR00T N1.7 arms for the G1 on SONIC, runtime side (py3.11; numpy + pyzmq + msgpack; never imports `gr00t`).

Owner decision (b) (PLAN §0.7, docs/groot_arms_design.md): GR00T outputs arm + Dex3 hand joint targets, the body
`arm` op streams them into SONIC's planner upper-body override, SONIC keeps the legs. The policy is the off-the-shelf
checkpoint `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` (N1.7, label **experimental**), served by the stock
Isaac-GR00T PolicyServer (scripts/groot_server.sh; the wire and the contract are in docs/groot_serving.md).

Modules:
    joint_order     every joint order involved (GR00T/Arena keys, LeRobot 43, MuJoCo 29, SONIC wire 17, mj17, Dex3),
                    URDF limits and the SONIC stand pose; all conversions go BY NAME
    wire            msgpack + numpy codec, byte-compatible with Isaac-GR00T's MsgSerializer
    policy_client   PolicyClient: ZMQ REQ client of the PolicyServer, robust to timeouts
    obs             build_observation(): ego frame + g1_debug state + prompt -> the server's observation dict
    actions         ArmChunk + to_arm_chunk(): a GR00T action chunk -> SONIC upper body (wire order) + Dex3 hands
    urdf            joint limits read from a URDF (for the box-side checks)
    open_loop       open-loop check against the CC-BY Arena dataset (runs on the dev box)
    bench           latency sampler (dev box local, or through an SSH tunnel)
"""

LABEL = "experimental"              # every GR00T skill result carries this label (owner decision, PLAN §0.8)
CHECKPOINT = "nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace"
CHECKPOINT_REVISION = "7f78bebf1a90131e7304beacfcd47eb27bad16ab"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5550"
