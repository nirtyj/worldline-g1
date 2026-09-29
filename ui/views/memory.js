// Memory & procedures: what the robot has stored, and what it has learned works.
//
// The server sends a "memory" message when memory changes (spatial memory, notes,
// earlier sessions, the procedural graph). Recall queries come from the trace.
// Entries that changed since the last message flash.
(() => {
  "use strict";
  const TABS = [["spatial", "Spatial"], ["noticed", "Noticed"], ["notes", "Notes"], ["recalls", "Recalls"], ["sessions", "Sessions"], ["rules", "Rules"]];
  const MIN_SUPPORT = 2;                          // agent/procedures.py: an edge needs this many tasks

  window.RobotViews = window.RobotViews || {};
  window.RobotViews.memory = (api) => {
    const { esc, nice } = api;
    let el = null, graphEl = null, guideEl = null, tab = "spatial", mem = null, prev = {}, fresh = new Set(), recalls = [];
    let lastGraphKey = "";

    const ago = (s) => (s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : s < 172800 ? `${Math.round(s / 3600)} h` : `${Math.round(s / 86400)} d`);

    function mount(inspector, graph, guide) {
      el = inspector; graphEl = graph; guideEl = guide;
      el.innerHTML = `<div class="subtabs" role="tablist">${TABS.map(([k, l]) => `<button class="tab" role="tab" data-tab="${k}" aria-selected="${k === tab}">${l} <span class="cnt" id="mcnt-${k}"></span></button>`).join("")}</div>
        <div class="state" id="mem-body"></div>`;
      el.querySelector(".subtabs").addEventListener("click", (e) => {
        const b = e.target.closest("[data-tab]"); if (!b) return;
        tab = b.dataset.tab;
        for (const x of el.querySelectorAll("[data-tab]")) x.setAttribute("aria-selected", String(x.dataset.tab === tab));
        renderBody();
      });
      el.addEventListener("click", (e) => { if (e.target.id === "btn-relearn") api.send({ type: "relearn" }); });
    }

    function memory(msg) {
      if (!msg.available) { mem = msg; renderBody(); return; }
      const keyed = {};
      for (const o of msg.spatial) keyed["o:" + o.id] = JSON.stringify([o.surface, o.status, o.history.length]);
      for (const n of msg.notes) keyed["n:" + n.text] = "1";
      for (const o of msg.observations || []) keyed["b:" + o.where + o.text] = String(o.seen_wall);
      for (const r of (msg.procedures || {}).rules || []) keyed["r:" + r.id] = r.status;
      fresh = new Set(Object.keys(keyed).filter((k) => mem && mem.available && prev[k] !== keyed[k]));
      prev = keyed; mem = msg;
      renderBody();
      lastGraphKey = "";
    }
    function rows(trs) {
      for (const r of trs) if (r.type === "recall") recalls.unshift(r);
      if (trs.some((r) => r.type === "recall") && tab === "recalls") renderBody();
    }
    function reset() { recalls = []; fresh = new Set(); prev = {}; mem = null; lastGraphKey = ""; renderBody(); }

    // ------------------------------------------------------------------ inspector
    function renderBody() {
      if (!el) return;
      const body = el.querySelector("#mem-body");
      const counts = mem && mem.available ? { spatial: mem.spatial.length, noticed: (mem.observations || []).length, notes: mem.notes.length, recalls: recalls.length,
        sessions: mem.sessions.length, rules: ((mem.procedures || {}).rules || []).length } : { recalls: recalls.length };
      for (const [k] of TABS) { const c = el.querySelector(`#mcnt-${k}`); if (c) c.textContent = counts[k] != null ? counts[k] : ""; }
      if (!mem) { body.innerHTML = `<div class="empty">Waiting for the server…</div>`; return; }
      if (!mem.available) { body.innerHTML = `<div class="empty">This runtime has no memory.</div>`; return; }
      body.innerHTML = { spatial, noticed, notes, recalls: recallList, sessions, rules }[tab]();
    }
    const fr = (k) => (fresh.has(k) ? " flash" : "");

    function spatial() {
      const byRoom = {}, away = [];
      for (const o of mem.spatial) {
        if (o.status === "seen" || o.status === "missed") {
          const room = o.room || "here";
          ((byRoom[room] = byRoom[room] || {})[o.surface] = byRoom[room][o.surface] || []).push(o);
        } else away.push(o);
      }
      const obj = (o) => {
        const trail = o.history.slice(0, 4).map((h) => `${esc(nice(h.surface))} <span class="muted">${ago(h.ago_s)}</span>`).join(" ← ");
        return `<div class="mrow${fr("o:" + o.id)}"><div><b>${esc(o.id)}</b> <span class="tag st-${o.status}">${o.status}</span> <span class="tag">${o.volatility}</span>
          ${o.usual && o.usual !== o.surface ? `<span class="muted">usually</span> ${esc(nice(o.usual))}` : ""}</div>
          ${trail ? `<div class="trail">${trail}</div>` : ""}</div>`;
      };
      let h = `<p class="hint">What spatial memory holds (<code>runs/memory/${esc(mem.scene)}.json</code>). The next session starts from it as unverified hints. It is only written from verified looks.</p>`;
      for (const [room, spots] of Object.entries(byRoom).sort()) {
        h += `<div class="block"><h3>${esc(nice(room))}</h3>`;
        for (const [spot, list] of Object.entries(spots).sort()) h += `<div class="spot"><div class="spotname">${esc(nice(spot))}</div>${list.map(obj).join("")}</div>`;
        h += `</div>`;
      }
      if (away.length) h += `<div class="block"><h3>Moved or carried <span class="muted" style="text-transform:none;letter-spacing:0">no current place; history kept</span></h3>${away.map(obj).join("")}</div>`;
      if (mem.landmarks.length) h += `<div class="block"><h3>Landmarks</h3><div class="list">${mem.landmarks.map((l) => `<div>${esc(l.label)} <span class="muted">near</span> ${esc(nice(l.near))}${l.room ? ` <span class="muted">(${esc(nice(l.room))})</span>` : ""}</div>`).join("")}</div></div>`;
      h += `<div class="block"><h3>Spots looked at</h3><div class="muted" style="font-size:12.5px">${mem.looked.map((s) => esc(nice(s))).join(", ") || "none"}</div></div>`;
      if (!mem.spatial.length) h += `<div class="empty">Nothing remembered in this house yet.</div>`;
      return h;
    }
    function noticed() {
      const list = mem.observations || [];
      if (!list.length) return `<div class="empty">Nothing noticed yet. System 1 watches the head camera and reports what the object list can't hold: a door left open, a spill, what a room looks like.</div>`;
      return `<p class="hint">What System 1 noticed, newest first. Unverified: the planner sees these as hints under NOTICED and confirms with a look before relying on them.</p><div class="list">${list.map((o) =>
        `<div class="late${fr("b:" + o.where + o.text)}">${esc(o.text)} <span class="muted">· ${o.where ? esc(nice(o.where)) + " · " : ""}${Math.round((o.confidence || 0) * 100)}% · ${ago(o.ago_s)} ago</span></div>`).join("")}</div>`;
    }
    function notes() {
      if (!mem.notes.length) return `<div class="empty">No notes. Tell it something about the home (“my keys are usually on the counter”) and it keeps the words.</div>`;
      return `<p class="hint">Kept word for word and shown to the planner as NOTES.</p><div class="list">${mem.notes.slice().reverse().map((n) =>
        `<div class="run${fr("n:" + n.text)}">“${esc(n.text)}”${n.about ? ` <span class="muted">about the ${esc(n.about)}</span>` : ""} <span class="muted">· ${ago(n.ago_s)} ago</span></div>`).join("")}</div>`;
    }
    function recallList() {
      if (!recalls.length) return `<div class="empty">No recall yet this session. The planner calls <code>recall</code> to ask memory instead of carrying it all in its prompt.</div>`;
      return `<p class="hint">Answered locally from belief, spatial memory, notes and the episode log. No model call.</p>` +
        recalls.map((r) => `<div class="block"><h3 style="text-transform:none;letter-spacing:0;color:var(--text)">t=${Number(r.t).toFixed(1)} · recall("${esc(r.query)}")</h3><pre class="ans">${esc(r.answer)}</pre></div>`).join("");
    }
    function sessions() {
      return `<p class="hint">Earlier sessions in this house, from the episode log (<code>runs/episodes/${esc(mem.scene)}/</code>).</p>` +
        mem.sessions.map((s) => `<div class="block"><h3>${s.current ? "this session" : s.wall ? ago(mem.wall - s.wall) + " ago" : "earlier"} <span class="muted" style="text-transform:none;letter-spacing:0">${s.decisions} decisions · ${s.recalls} recalls</span></h3><div class="list">
          ${s.requests.length ? s.requests.map((t) => `<div>asked: “${esc(t)}”</div>`).join("") : `<div class="muted">no requests</div>`}
          ${s.delivered.map((d) => `<div class="run">delivered ${esc(nice(d))}</div>`).join("")}</div></div>`).join("");
    }
    function rules() {
      const p = mem.procedures;
      if (!p) return `<div class="empty">No procedural graph.</div>`;
      const edges = Object.values(p.edges).filter((c) => c.ok + c.fail >= MIN_SUPPORT).length;
      let h = `<p class="hint">Learned from ${p.tasks.ok + p.tasks.fail} tasks (${p.tasks.ok} went well, ${p.tasks.fail} did not)${p.learned_wall ? `, last learned ${ago(mem.wall - p.learned_wall)} ago` : ""}. ${edges} transitions have enough support to be suggested.
        <button class="mini" type="button" id="btn-relearn" title="Recount the graph from every episode on disk">Relearn now</button></p>`;
      if (!p.rules.length) h += `<div class="empty">No rules proposed yet (<code>python eval/evolve.py</code>).</div>`;
      h += `<div class="list">${p.rules.map((r) => `<div class="${r.status === "kept" ? "run" : r.status === "rejected" ? "cxl" : "late"}${fr("r:" + r.id)}"><span class="tag st-${r.status}">${r.status}</span> after <b>${esc(r.after)}</b>: ${esc(r.then)}<div class="muted">${esc(r.why)}</div></div>`).join("")}</div>`;
      return h;
    }

    // ------------------------------------------------------------------ procedure graph
    function renderGraph(msg) {
      if (!graphEl || !mem || !mem.available || !mem.procedures) {
        if (graphEl && (!mem || !mem.procedures)) graphEl.innerHTML = `<div class="empty" style="padding:20px">No procedural graph yet.</div>`;
        return;
      }
      const proc = (msg.runtime && msg.runtime.procedure) || { steps: [], guidance: "" };
      const key = JSON.stringify([mem.wall, proc.steps]);
      if (guideEl) {
        const g = proc.guidance ? `<b>Guidance the planner gets now</b><pre class="ans">${esc(proc.guidance)}</pre>` : `<span class="muted">No guidance right now: ${proc.steps.length ? "nothing learned after " + esc(proc.steps[proc.steps.length - 1]) : "no request in progress"}.</span>`;
        const html = `${g}<div class="muted" style="margin-top:6px">This task so far: ${proc.steps.length ? ["start", ...proc.steps].map(esc).join(" → ") : "—"}</div>`;
        if (guideEl.dataset.h !== html) { guideEl.innerHTML = html; guideEl.dataset.h = html; }
      }
      if (key === lastGraphKey) return;
      lastGraphKey = key;
      const p = mem.procedures;
      const E = Object.entries(p.edges).map(([k, c]) => { const [a, b] = k.split(" -> "); return { a, b, ok: c.ok, fail: c.fail, n: c.ok + c.fail }; })
        .filter((e) => e.n >= MIN_SUPPORT);
      // columns by BFS depth from "start"
      const depth = { start: 0 }, q = ["start"];
      while (q.length) { const a = q.shift(); for (const e of E) if (e.a === a && depth[e.b] == null) { depth[e.b] = depth[a] + 1; q.push(e.b); } }
      const nodes = [...new Set(E.flatMap((e) => [e.a, e.b]))];
      const maxD = Math.max(0, ...Object.values(depth));
      for (const n of nodes) if (depth[n] == null) depth[n] = maxD + 1;
      const cols = {};
      for (const n of nodes) (cols[depth[n]] = cols[depth[n]] || []).push(n);
      const W = 150, H = 54, X0 = 20, Y0 = 30;
      const P = {};
      for (const [d, list] of Object.entries(cols)) {
        list.sort((a, b) => (p.nodes[b] || 0) - (p.nodes[a] || 0));
        list.forEach((n, i) => { P[n] = [X0 + Number(d) * (W + 50), Y0 + i * (H + 26)]; });
      }
      const width = X0 * 2 + (Math.max(...Object.keys(cols).map(Number)) + 1) * (W + 50);
      const height = Y0 + Math.max(...Object.values(cols).map((l) => l.length)) * (H + 26) + 20;
      const path = ["start", ...proc.steps];
      const onPath = new Set(path.slice(1).map((b, i) => `${path[i]}|${b}`));
      const cur = path[path.length - 1];
      let h = `<svg viewBox="0 0 ${width} ${height}" style="min-width:${Math.min(width, 900)}px" role="img" aria-label="Procedural graph: which step usually follows which, and how often it went well">`;
      for (const e of E) {
        const [ax, ay] = P[e.a], [bx, by] = P[e.b];
        const r = e.ok / e.n, col = r >= 0.8 ? "var(--robot)" : r >= 0.5 ? "var(--warn)" : "var(--bad)";
        const sw = Math.min(1.2 + e.n * 0.5, 9), hot = onPath.has(`${e.a}|${e.b}`);
        let d;
        if (e.a === e.b) d = `M${ax + W - 20} ${ay} C${ax + W + 20} ${ay - 40} ${ax + W + 30} ${ay + 20} ${ax + W} ${ay + H / 2}`;
        else if (bx > ax) d = `M${ax + W} ${ay + H / 2} C${ax + W + 30} ${ay + H / 2} ${bx - 30} ${by + H / 2} ${bx} ${by + H / 2}`;
        else d = `M${ax + W / 2} ${ay + H} C${ax + W / 2} ${ay + H + 60} ${bx + W / 2} ${by + H + 60} ${bx + W / 2} ${by + H}`;
        h += `<path class="g-edge${hot ? " hot" : ""}" d="${d}" stroke="${col}" stroke-width="${sw}"><title>${esc(e.a)} → ${esc(e.b)}: ${e.ok} of ${e.n} tasks went well</title></path>`;
      }
      for (const n of nodes) {
        const [x, y] = P[n];
        h += `<g class="g-node${n === cur ? " cur" : ""}${path.includes(n) ? " done" : ""}"><rect x="${x}" y="${y}" width="${W}" height="${H}" rx="8"/>
          <text x="${x + W / 2}" y="${y + 23}" text-anchor="middle" class="gn">${esc(n)}</text>
          <text x="${x + W / 2}" y="${y + 40}" text-anchor="middle" class="gc">${n === "start" ? "a request" : `${p.nodes[n] || 0}× seen`}</text></g>`;
      }
      h += `</svg>`;
      graphEl.innerHTML = h +
        `<p class="hint" style="padding:0 14px">Steps as columns by how soon they come after a request. Edge width: how many tasks took it; colour: how often those tasks went well (green ≥ 80 %, amber ≥ 50 %, red below). The current task's path is bright; its last step pulses. Only transitions seen in ${MIN_SUPPORT}+ tasks are drawn.</p>`;
    }
    return { mount, memory, rows, reset, renderGraph, renderBody };
  };
})();
