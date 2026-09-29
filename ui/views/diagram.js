// How it works: one static picture of the whole system, shown full screen from the header.
//
// Left to right: you and the robot's sensors; System 1 (Jev labels every message, Gemini
// Live watches the frames the gate lets through); the fused state; the runtime, the only
// part that moves the robot; the planner, the persona and the skills. Memory sits under
// the runtime and the simulator under everything. The model names come from the running
// session when it has them.
(() => {
  "use strict";
  const W = 1500, H = 856;

  window.RobotViews = window.RobotViews || {};
  window.RobotViews.diagram = (api) => {
    const { esc } = api;

    // "**word** rest" -> the word bold
    const rich = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<tspan class="k">$1</tspan>');

    function group(x, y, w, h, c, label) {
      return `<g style="--c: var(--${c})"><rect class="d-group" x="${x}" y="${y}" width="${w}" height="${h}" rx="14"/>
        <text class="d-glabel" x="${x + 14}" y="${y + 20}">${esc(label)}</text></g>`;
    }
    function box(x, y, w, h, c, title, { tag = "", sub = "", lines = [], step = 20, wide = false } = {}) {
      let s = `<g style="--c: var(--${c})"><rect class="d-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>
        <text class="d-title" x="${x + 14}" y="${y + (wide ? 34 : 26)}">${esc(title)}</text>`;
      if (tag) s += `<text class="d-tag" x="${x + w - 12}" y="${y + 24}" text-anchor="end">${esc(tag)}</text>`;
      if (sub) s += `<text class="d-sub" x="${wide ? x + 200 : x + 14}" y="${y + (wide ? 34 : 47)}">${esc(sub)}</text>`;
      lines.forEach((l, i) => { s += `<text class="d-line" x="${x + 14}" y="${y + 70 + i * step}">${rich(l)}</text>`; });
      return s + `</g>`;
    }
    function edge(pts, { label = "", at = null, anchor = "middle", both = false } = {}) {
      let s = `<polyline class="d-edge" points="${pts.map((p) => p.join(",")).join(" ")}" marker-end="url(#d-head)"${both ? ' marker-start="url(#d-head)"' : ""}/>`;
      if (label) s += `<text class="d-elabel" x="${at[0]}" y="${at[1]}" text-anchor="${anchor}">${esc(label)}</text>`;
      return s;
    }

    function render(el, info = {}) {
      const s1 = info.system1 || {}, m = /labels (\S+?):.*observations (\S+?):/.exec(s1.detail || "");
      const jev = m ? m[1] : "jev-1.13.0", live = m ? m[2] : "gemini-3.8-live";
      const planner = info.planner ? info.planner[1] : "Gemini 3.8 Flash";
      const s1note = s1.status === "off" ? " · OFF IN THIS SESSION: THE PLANNER LABELS" : s1.status === "error" ? " · ERROR IN THIS SESSION" : "";
      let h = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="How the system fits together: you and the sensors, System 1 (Jev labels, Gemini Live observations), the fused state, the runtime, the planner, the persona, the skills, memory and the simulated house">
        <defs><marker id="d-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="#6a7894"/></marker></defs>`;

      // groups first, so boxes and arrows sit on top
      h += group(290, 24, 500, 286, "s1", "SYSTEM 1 · FAST, ALWAYS ON" + s1note);
      h += group(1268, 340, 224, 250, "robot", "SKILLS");
      h += group(290, 620, 900, 132, "memory", "MEMORY · ON DISK, KEPT ACROSS SESSIONS");

      // arrows
      h += edge([[190, 106], [306, 106]], { label: "each message", at: [248, 98] });
      h += edge([[774, 106], [850, 106]], { label: "label + P", at: [812, 98] });
      h += edge([[774, 240], [850, 240]], { label: "noticed", at: [812, 232] });
      h += edge([[190, 388], [290, 388]], { label: "frames", at: [240, 380] });
      h += edge([[190, 500], [290, 500]], { label: "frames", at: [240, 492] });
      h += edge([[405, 340], [405, 302]], { label: "new frames only", at: [414, 326], anchor: "start" });
      h += edge([[520, 490], [550, 490]]);
      h += edge([[670, 400], [670, 312]], { label: "robot state", at: [680, 360], anchor: "start" });
      h += edge([[790, 430], [850, 430]], { label: "state", at: [820, 422] });
      h += edge([[1190, 70], [1280, 70]], { label: "context", at: [1235, 62] });
      h += edge([[1280, 124], [1190, 124]], { label: "tool call", at: [1235, 116] });
      h += edge([[1280, 252], [1190, 252]], { label: "own goal", at: [1235, 244] });
      h += edge([[1190, 400], [1268, 400]], { label: "goals", at: [1229, 392] });
      h += edge([[1020, 456], [1020, 618]], { label: "write · recall · guidance", at: [1032, 542], anchor: "start", both: true });
      h += edge([[1380, 590], [1380, 788]], { label: "moves · grasps", at: [1390, 700], anchor: "start" });
      h += edge([[105, 788], [105, 542]], { label: "frames · pose", at: [115, 680], anchor: "start" });

      // you and the robot's sensors
      h += box(20, 60, 170, 104, "user", "You", { sub: "type in the chat", lines: ["replies show up there"] });
      h += box(20, 340, 170, 200, "robot", "Sensors", { sub: "on the robot", lines: ["head camera 640×480", "odometry", "gripper state"], step: 24 });

      // System 1
      h += box(306, 50, 468, 112, "s1", "Labels · Jev", { tag: `TypeSafe ${jev}`, sub: "every message, one stateless call, ~0.1 s",
        lines: ["**kind** request, stop, correction, question, answer … + P(kind)", "**also** a yes/no answer · replaces the task? · which object"] });
      h += box(306, 178, 468, 122, "s1", "Observations · Gemini 3.8 Live", { tag: live, sub: "one live session; sees only what the gate lets through",
        lines: ["notices what the object list can't hold:", "a door left open, a spill, a pet bed", "**never told the goal**, so it can't imagine the target"] });
      h += box(290, 340, 230, 96, "s1", "Frame gate", { sub: "a new view or a scene change", lines: ["32×24 diff · at most 1 a second"] });
      h += box(290, 462, 230, 78, "robot", "Perception", { sub: "objects in view (detector)" });
      h += box(550, 400, 240, 100, "runtime", "Fused state", { sub: "10 Hz: pose, objects, task", lines: ["what the runtime and System 1 read"] });

      // the runtime
      h += box(850, 24, 340, 432, "runtime", "Runtime", { sub: "the only part that moves the robot", step: 31, lines: [
        "**belief** · what it thinks is where",
        "**labels** · System 1's when P ≥ 0.5",
        "      ↳ otherwise the planner labels",
        "**stop** · halts at once, no model call",
        "**version** · drops stale planner answers",
        "**checks** · every tool call before it runs",
        "**looks** · after every pick and place",
        "**speech** · drops lines that went stale",
        "**recall** · answers from memory, no model",
        "**step mode** · waits before each call",
      ] });

      // planner, persona, skills
      h += box(1280, 24, 200, 144, "brain", "Planner", { tag: "System 2", sub: planner, lines: ["one step at a time, 1–4 s", "labels when System 1 is unsure", "one forced tool call a turn"] });
      h += box(1280, 196, 200, 112, "persona", "Persona", { sub: "own goals when idle", lines: ["off · quiet · medium · optimize", "asks before it explores"] });
      h += box(1280, 366, 200, 62, "robot", "Nav", { sub: "grid paths · 0.6 m/s" });
      h += box(1280, 438, 200, 62, "robot", "Arms", { sub: "pick, place: timed chunks" });
      h += box(1280, 510, 200, 62, "robot", "Say", { sub: "timed lines → your chat" });

      // memory
      h += box(306, 648, 206, 92, "memory", "Spatial", { sub: "where things are", lines: ["from verified looks only"] });
      h += box(526, 648, 206, 92, "memory", "Episodic", { sub: "what happened", lines: ["requests, steps, deliveries"] });
      h += box(746, 648, 206, 92, "memory", "Procedural", { sub: "what usually works next", lines: ["→ guidance in the prompt"] });
      h += box(966, 648, 206, 92, "memory", "Notes", { sub: "what you told it", lines: ["+ what System 1 noticed"] });

      // the simulator
      h += box(20, 788, 1460, 58, "world", "ProcTHOR house", { wide: true,
        sub: "AI2-THOR simulator: runs the moves, renders the camera, and holds the truth the runtime never sees" });

      el.innerHTML = h + `</svg>`;
    }
    return { render };
  };
})();
