// How it works: one static picture of Worldline on a Unitree G1, shown full screen from the header.
//
// Top to bottom, the way a request travels: you and the page; System 1 (Jev labels every
// message, Gemini Live watches the frames the gate lets through); the runtime, the only part
// that moves the robot, with the planner (System 2) and the persona beside it; the api/
// contract every call and result goes through (tools, envelopes, executions, events, tool
// state); the services behind the tools and the robot facade; the world model, the only
// reader of the simulator's ground truth; the SONIC body server and the unmodified SONIC
// deploy; and Isaac Sim at the bottom with the G1 in a MolmoSpaces ProcTHOR house.
// The model names, the profile and its fallbacks come from the running session when it has them.
(() => {
  "use strict";
  const W = 1500, H = 900;

  window.RobotViews = window.RobotViews || {};
  window.RobotViews.diagram = (api) => {
    const { esc } = api;

    // "**word** rest" -> the word bold
    const rich = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<tspan class="k">$1</tspan>');

    function group(x, y, w, h, c, label) {
      return `<g style="--c: var(--${c})"><rect class="d-group" x="${x}" y="${y}" width="${w}" height="${h}" rx="14"/>
        <text class="d-glabel" x="${x + 14}" y="${y + 20}">${esc(label)}</text></g>`;
    }
    function box(x, y, w, h, c, title, { tag = "", sub = "", lines = [], step = 19, top = 64, dashed = false } = {}) {
      let s = `<g style="--c: var(--${c})"><rect class="d-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="10"${dashed ? ' stroke-dasharray="6 4"' : ""}/>
        <text class="d-title" x="${x + 14}" y="${y + 26}">${esc(title)}</text>`;
      if (tag) s += `<text class="d-tag" x="${x + w - 12}" y="${y + 24}" text-anchor="end">${esc(tag)}</text>`;
      if (sub) s += `<text class="d-sub" x="${x + 14}" y="${y + 46}">${esc(sub)}</text>`;
      lines.forEach((l, i) => { s += `<text class="d-line" x="${x + 14}" y="${y + top + i * step}">${rich(l)}</text>`; });
      return s + `</g>`;
    }
    function edge(pts, { label = "", at = null, anchor = "middle", both = false } = {}) {
      let s = `<polyline class="d-edge" points="${pts.map((p) => p.join(",")).join(" ")}" marker-end="url(#d-head)"${both ? ' marker-start="url(#d-head)"' : ""}/>`;
      if (label) s += `<text class="d-elabel" x="${at[0]}" y="${at[1]}" text-anchor="${anchor}">${esc(label)}</text>`;
      return s;
    }

    function render(el, info = {}) {
      const s1 = info.system1 || {}, m = /labels (\S+?):.*observations (\S+?):/.exec(s1.detail || "");
      const jev = m ? m[1] : "jev", live = m ? m[2] : "gemini-live";
      const planner = info.planner ? info.planner[1] : "Gemini 3.8 Flash";
      const profile = info.profile || "sonic";
      const stones = info.stones || [];
      const s1note = s1.status === "off" ? " · OFF NOW: PLANNER LABELS" : s1.status === "error" ? " · ERROR NOW" : "";
      let h = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="How Worldline drives a Unitree G1: you and the page, System 1, the runtime and planner, the api contract, the services and robot facade, the world model (the only ground-truth reader), the SONIC body server and deploy, and Isaac Sim with the G1 in a MolmoSpaces house">
        <defs><marker id="d-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="#6a7894"/></marker></defs>`;

      // groups first, so boxes and arrows sit on top
      h += group(290, 14, 520, 246, "s1", "SYSTEM 1 · FAST, ALWAYS ON" + s1note);
      h += group(20, 336, 1460, 96, "runtime", "api/ · THE CONTRACT · STDLIB ONLY · SAME IN EVERY PROFILE");
      h += group(290, 452, 900, 186, "robot", `SERVICES + ROBOT FACADE (robot/, services/) · PROFILE ${profile.toUpperCase()}`);
      h += group(290, 660, 1190, 110, "robot", "BODY · SONIC (P3 wl-body, P2 wl-sonic, P4 wl-policy)");

      // arrows
      h += edge([[190, 70], [306, 70]], { label: "each message", at: [248, 62] });
      h += edge([[794, 70], [860, 70]], { label: "label + P", at: [827, 62] });
      h += edge([[794, 196], [860, 196]], { label: "noticed", at: [827, 188] });
      h += edge([[1190, 60], [1270, 60]], { label: "context", at: [1230, 52] });
      h += edge([[1270, 110], [1190, 110]], { label: "1 tool call", at: [1230, 102] });
      h += edge([[1270, 222], [1190, 222]], { label: "own goal", at: [1230, 214] });
      h += edge([[1025, 320], [1025, 350]], { label: "validate · start · envelope", at: [1035, 344], anchor: "start", both: true });
      h += edge([[740, 432], [740, 468]], { label: "ToolResult, events", at: [750, 456], anchor: "start", both: true });
      h += edge([[290, 545], [240, 545]], { label: "GT", at: [265, 537] });
      h += edge([[1090, 638], [1090, 676]], { label: "DEALER 5610 · SUB 5611 · PUSH halt 5612", at: [1100, 660], anchor: "start", both: true });
      h += edge([[740, 724], [800, 724]], { label: "PUB 5556", at: [770, 716] });
      h += edge([[1040, 770], [1040, 800]], { label: "DDS rt/lowcmd", at: [1050, 792], anchor: "start" });
      h += edge([[1160, 800], [1160, 770]], { label: "rt/lowstate", at: [1170, 792], anchor: "start" });
      h += edge([[480, 800], [480, 770]], { label: "gt.pose 50 Hz (5601)", at: [490, 792], anchor: "start" });
      h += edge([[135, 800], [135, 638]], { label: "REQ 5600 · SUB 5601", at: [145, 720], anchor: "start", both: true });
      h += edge([[262, 800], [262, 300], [190, 300]], { label: "frames 5565 · 5602", at: [256, 664], anchor: "end" });
      h += edge([[135, 452], [135, 300]], { label: "truth (UI + eval only)", at: [145, 324], anchor: "start" });
      h += edge([[1370, 432], [1370, 468]], { label: "write · recall · guidance", at: [1380, 456], anchor: "start", both: true });

      // you and the page
      h += box(20, 24, 170, 110, "user", "You", { sub: "type in the chat", lines: ["Stop halts (keyword lane)", "Kill = hardware e-stop"] });
      h += box(20, 170, 170, 130, "world", "This page", { sub: "an observer", lines: ["cameras: viz FrameTap", "reality: sim ground truth", "**never** fed to the planner"] });

      // System 1
      h += box(306, 42, 488, 100, "s1", "Labels · Jev", { tag: `TypeSafe ${jev}`, sub: "every message, one stateless call, ~0.1 s",
        lines: ["**kind** request, stop, correction, question … + P(kind)"] });
      h += box(306, 152, 330, 98, "s1", "Observer · Gemini Live", { tag: live, sub: "sees frames the gate lets through",
        lines: ["**never told the goal**"] });
      h += box(646, 152, 148, 98, "s1", "Frame gate", { sub: "new view or", lines: ["scene change", "≤ 1 frame/s"], top: 64 });

      // the runtime
      h += box(860, 14, 330, 306, "runtime", "Runtime (agent/)", { sub: "the only part that moves the robot", step: 24, top: 70, lines: [
        "**belief** · never ground truth",
        "**validate** · schema → enum → state → capability",
        "**executions** · nav-000012 · g/e fences",
        "**stop** · halt() first, no model call",
        "**correction** · generation + 1, cancel",
        "**late** results · world info, not progress",
        "**looks** · arrival scan, verify, reconcile",
        "**layout** + goal check · memory · narrator",
        "**step mode** · waits before each call",
      ] });

      // planner, persona
      h += box(1270, 14, 210, 150, "brain", "Planner", { tag: "System 2", sub: planner, lines: ["one step at a time", "one forced tool call a turn", "prompt: belief only"] });
      h += box(1270, 182, 210, 138, "persona", "Persona", { sub: "own goals when idle", lines: ["off · quiet · medium · optimize", "asks before it explores", "never manipulates"] });

      // the contract
      h += `<text class="d-line" x="36" y="382">${rich("**tools** speak · list_locations · navigate(location | reach_stance) · check_reachability · manipulate(pick | place) · wait_and_observe · recall [WL]")}</text>`;
      h += `<text class="d-line" x="36" y="408">${rich("**ToolResult** succeeded | failed | cancelled | timed_out | rejected · executor · skill · observation_id · late   **events** InteractionEvent   **tool state** IDLE · NAVIGATING · MANIPULATING · OBSERVING · WAITING · STOPPED · FAULT")}</text>`;

      // world model
      h += box(20, 452, 220, 186, "world", "World model (world/)", { sub: "the ONLY ground-truth reader", step: 20, lines: [
        "static map: rooms, surfaces,", "keypoints, occupancy / NavGrid", "objects: where, visible, hands", "truth() → the page + eval only", "swappable for perception later"] });

      // services + facade
      const fb = (name) => stones.includes(name) ? " (fallback)" : "";
      h += box(306, 486, 200, 140, "robot", "Navigation", { sub: "NavGrid A* · 0.4 m/s", step: 19, lines: [
        `**sonic_walk**${fb("sonic_walk")}`, `kinematic_nav${fb("kinematic_nav") || " (bring-up)"}`, "reach_stance reposition"] });
      h += box(516, 486, 216, 140, "robot", "Manipulation", { sub: "skill registry · reachability", step: 19, lines: [
        "**groot_sonic** (full)", `sonic_arm_script${fb("sonic_arm_script") || " (fallback)"}`, "  = SONIC reach + GT attach"] });
      h += box(742, 486, 200, 140, "robot", "Observation", { sub: "scan: waist ±35°, 2 rows", step: 19, lines: ["glance on every result", "Speech: queue, cut, drop", "Locations: list_locations"] });
      h += box(952, 486, 226, 140, "robot", "Robot facade", { tag: "G1Robot", sub: "robot.start(execution)", step: 19, lines: ["halt() ≤ 30 ms · estop()", "telemetry: body mode, RTF", "BodyClient → P3"] });

      // memory
      h += box(1270, 468, 210, 170, "memory", "Memory", { sub: "on disk, per house", step: 20, lines: ["spatial · where things are", "episodic · what happened", "procedural · what works next", "notes · what you said"] });

      // body
      h += box(306, 680, 434, 80, "robot", "BodyServer (P3)", { tag: "sole binder of 5556", sub: "lease · halt latch · PathFollower 50 Hz · waist scan · CarryLock", top: 64, lines: [] });
      h += box(800, 680, 390, 80, "robot", "gear_sonic_deploy (P2)", { tag: "unmodified C++", sub: "SONIC: planner 10 Hz · control 50 Hz · TensorRT", lines: [] });
      h += box(1250, 680, 216, 80, "brain", "GR00T N1.7 (P4)", { sub: "full profile, later", dashed: true, lines: [] });

      // the simulator
      h += box(20, 800, 1460, 86, "world", "Isaac Sim 5.1 (P1) · Unitree G1 in a MolmoSpaces ProcTHOR house", {
        sub: "PhysX 200 Hz · Unitree DDS bridge · head camera 5565 · VizCams chase/top 5602 · GT server REP 5600 / PUB 5601 · the truth the planner never sees" });

      el.innerHTML = h + `</svg>`;
    }
    return { render };
  };
})();
