// The robot's map, the way a robot vacuum shows it.
//
// Layers the robot itself knows: its nav stack's free-space grid (the occupancy grid's
// inflated layer, shown at 0.25 m), the floor its head camera has covered, the path it
// walked, the route the path follower is on (with its lookahead point, stuck points and
// blocked edges), the camera's view fan (both scan rows during a scan, each with its own
// near edge), where it stood when it found things, its named spots (the map service), and
// belief: a chip per object on the spot the runtime thinks it is on, coloured by where
// that belief came from. The robot is drawn at its own pose estimate, with the executor
// that is moving it (amber when it is a labelled sim shortcut).
// The house outline comes from the world model (sim ground truth) and is only a visual
// aid; "truth" adds ghost chips where the simulator disagrees with belief.
(() => {
  "use strict";
  const FRESH_MS = 3000, MAX_CHIPS = 6, CHIP_H = 16;
  const KIND = {
    seen: { cls: "seen", label: "seen" },                          // verified: a look or the camera
    mem: { cls: "mem", label: "remembered from an earlier session" },
    skill: { cls: "skill", label: "from an action result, not seen yet" },
    gone: { cls: "gone", label: "not where expected" },
  };
  const kindOf = (o) => (o.where === "UNKNOWN" ? KIND.gone : o.verified ? KIND.seen : o.source === "memory" ? KIND.mem : KIND.skill);

  window.RobotViews = window.RobotViews || {};
  window.RobotViews.plan = (api) => {
    const { esc, nice } = api;
    let root = null, svg = null, legend = null, lay = null, VB = null, FULL = null;
    let showTruth = false, prevWhere = null, fresh = {};
    let trail = [], sightings = [], sightKeys = new Set(), explored = new Set(), cell = 4;
    const LAYERS = [["house", "house outline", "From the world model (sim ground truth): a visual aid only. The robot has no walls or room shapes."],
                    ["grid", "nav grid", "The free cells its nav stack plans paths on: the occupancy grid's inflated layer, shown at 0.25 m."],
                    ["explored", "explored", "Floor its head camera has covered this session (in range, in view, past the near edge, clear line of sight)."],
                    ["path", "path", "Where it walked this session (its own pose), the route it is on now (dashed), the follower's lookahead, stuck points and blocked edges."],
                    ["view", "view", "The head camera's view fan now: 90 deg, 2.5 m; both scan rows (waist +-35 deg) while it scans."],
                    ["found", "found", "Where it stood when it verified where something is."],
                    ["belief", "belief", "Chips: what the runtime believes is on each spot."]];
    const layers = { house: true, grid: true, explored: true, path: true, view: true, found: true, belief: true };

    function mount(el) {
      root = el;
      el.innerHTML = `<div class="planwrap">
          <svg id="plan-svg" role="img" aria-label="The robot's internal belief map: its nav grid, explored floor, path, finds and belief. Scroll to zoom, drag to pan, double-click to fit."></svg>
          <span class="viewlabel">Robot's internal belief map <span class="muted" id="plan-stats"></span></span>
          <div class="maplegend" id="plan-legend"></div>
          <span class="vtools">
            <button class="mini" type="button" data-zoom="in" title="Zoom in (or scroll)">+</button>
            <button class="mini" type="button" data-zoom="out" title="Zoom out">−</button>
            <button class="mini" type="button" data-zoom="fit" title="Show the whole house (or double-click)">Fit</button>
            <label title="Ghost chips where the simulator disagrees with belief"><input type="checkbox" id="plan-truth"> truth</label>
            <button class="mini" type="button" id="plan-layers-btn" aria-expanded="false" aria-controls="plan-layers" title="Choose what the map draws">Layers</button>
          </span>
          <div class="layerpop" id="plan-layers" hidden>
            <h4>Layers</h4>
            ${LAYERS.map(([k, label, tip]) => `<label title="${esc(tip)}"><input type="checkbox" data-layer="${k}" checked> ${label}${k === "house" ? ' <span class="muted">visual aid</span>' : ""}</label>`).join("")}
          </div>
        </div>`;
      svg = el.querySelector("#plan-svg"); legend = el.querySelector("#plan-legend");
      const pop = el.querySelector("#plan-layers"), btn = el.querySelector("#plan-layers-btn");
      const showPop = (on) => { pop.hidden = !on; btn.setAttribute("aria-expanded", String(on)); };
      el.addEventListener("change", (e) => {
        if (e.target.dataset.layer) { layers[e.target.dataset.layer] = e.target.checked; applyLayers(); }
        if (e.target.id === "plan-truth") { showTruth = e.target.checked; api.rerender(); }
      });
      el.addEventListener("click", (e) => {
        const z = e.target.closest("[data-zoom]");
        if (z && VB) {
          if (z.dataset.zoom === "fit") VB = { ...FULL }; else zoomAt(VB.x + VB.w / 2, VB.y + VB.h / 2, z.dataset.zoom === "in" ? 0.7 : 1 / 0.7);
          applyVB();
        }
        if (e.target === btn) showPop(pop.hidden);
      });
      document.addEventListener("click", (e) => { if (!pop.hidden && !pop.contains(e.target) && e.target !== btn) showPop(false); });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") showPop(false); });
      svg.addEventListener("wheel", (e) => { e.preventDefault(); const p = toSvg(e.clientX, e.clientY); zoomAt(p.x, p.y, Math.exp(e.deltaY * 0.0015)); }, { passive: false });
      let drag = null;
      svg.addEventListener("pointerdown", (e) => { if (e.target.closest("a,button")) return; drag = { cx: e.clientX, cy: e.clientY, vb: { ...VB }, a: svg.getScreenCTM().a }; svg.setPointerCapture(e.pointerId); });
      svg.addEventListener("pointermove", (e) => { if (!drag) return; VB.x = drag.vb.x - (e.clientX - drag.cx) / drag.a; VB.y = drag.vb.y - (e.clientY - drag.cy) / drag.a; applyVB(); });
      const end = () => { drag = null; };
      svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
      svg.addEventListener("dblclick", () => { VB = { ...FULL }; applyVB(); });
    }
    function applyLayers() {
      if (!svg) return;
      const show = { "plan-house": layers.house, "plan-houselabels": layers.house, "plan-grid": layers.grid, "plan-explored": layers.explored,
                     "plan-trail": layers.path, "plan-route": layers.path, "plan-nav": layers.path, "plan-view": layers.view, "plan-sight": layers.found,
                     "plan-chips": layers.belief, "plan-lines": layers.belief };
      for (const [id, on] of Object.entries(show)) { const g = svg.querySelector("#" + id); if (g) g.style.display = on ? "" : "none"; }
    }
    function applyVB() { if (svg && VB) svg.setAttribute("viewBox", `${VB.x} ${VB.y} ${VB.w} ${VB.h}`); }
    function toSvg(cx, cy) { const pt = svg.createSVGPoint(); pt.x = cx; pt.y = cy; return pt.matrixTransform(svg.getScreenCTM().inverse()); }
    function zoomAt(x, y, f) {
      const w = Math.min(Math.max(VB.w * f, 60), FULL.w * 1.5), k = w / VB.w;
      VB = { x: x - (x - VB.x) * k, y: y - (y - VB.y) * k, w, h: VB.h * k }; applyVB();
    }

    const short = (spot) => {
      const room = lay && lay.keypoints && Object.keys(lay.rooms || {}).find((r) => spot.startsWith(r + "_"));
      return nice(room ? spot.slice(room.length + 1) : spot);
    };

    function reset(layout) {
      lay = layout; prevWhere = null; fresh = {};
      trail = []; sightings = []; sightKeys = new Set(); explored = new Set();
      const step = lay.grid_step || 0.25;
      const a = api.uv(0, 0), b2 = api.uv(step, 0);
      cell = Math.abs(b2[0] - a[0]);                      // one grid cell in drawing units
      const pts = [];
      for (const [ix, iz] of lay.grid || []) pts.push(api.uv(ix * step, iz * step));
      for (const r of Object.values(lay.rooms || {})) for (const [x, z] of r.polygon || []) pts.push(api.uv(x, z));
      for (const k of Object.values(lay.keypoints || {})) pts.push(api.uv(k.x, k.z));
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const pad = 40, x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad;
      FULL = { x: x0, y: y0, w: Math.max(...xs) + pad - x0 + 90, h: Math.max(...ys) + pad - y0 };
      VB = { ...FULL };
      let h = `<defs><pattern id="fog" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="8" height="8" fill="#0d1420"/><line x1="0" y1="0" x2="0" y2="8" stroke="#2a3650" stroke-width="3"/></pattern></defs>`;
      h += `<g id="plan-house">`;
      const rooms = Object.entries(lay.rooms || {});
      if (rooms.length) {
        for (const [, r] of rooms) {
          const poly = (r.polygon || []).map(([x, z]) => api.uv(x, z).join(",")).join(" ");
          if (poly) h += `<polygon class="p-room" points="${poly}"/>`;
        }
      } else {                                            // an iTHOR room: the box around its spots
        h += `<rect class="p-room" x="${Math.min(...xs) - 20}" y="${Math.min(...ys) - 20}" width="${Math.max(...xs) - Math.min(...xs) + 40}" height="${Math.max(...ys) - Math.min(...ys) + 40}" rx="6"/>`;
      }
      h += `</g><g id="plan-grid">`;
      for (const [ix, iz] of lay.grid || []) { const [u, v] = api.uv(ix * step, iz * step); h += `<rect x="${u - cell / 2}" y="${v - cell / 2}" width="${cell}" height="${cell}"/>`; }
      h += `</g><g id="plan-explored"></g><g id="plan-houselabels">`;
      for (const [, r] of rooms) { const [u, v] = api.uv(r.x, r.z); h += `<text class="p-roomlabel" x="${u}" y="${v}" text-anchor="middle">${esc(r.label.toUpperCase())}</text>`; }
      h += `</g><g id="plan-view"></g><polyline id="plan-trail" points=""/><polyline id="plan-route" points=""/><g id="plan-nav"></g><g id="plan-sight"></g>
            <g id="plan-spots"></g><g id="plan-lines"></g><g id="plan-chips"></g><g id="plan-agents"></g>`;
      svg.innerHTML = h;
      applyVB(); applyLayers();
    }

    // The robot's own map, from the server: everything so far on connect, then what is new each frame.
    function robotMap(m, full) {
      if (!svg || !lay || !m) return;
      if (full) { trail = []; sightings = []; sightKeys = new Set(); explored = new Set(); svg.querySelector("#plan-explored").innerHTML = ""; }
      const step = lay.grid_step || 0.25;
      let cells = "";
      for (const [ix, iz] of m.explored || []) {
        const k = ix + "," + iz; if (explored.has(k)) continue;
        explored.add(k);
        const [u, v] = api.uv(ix * step, iz * step);
        cells += `<rect x="${u - cell / 2}" y="${v - cell / 2}" width="${cell}" height="${cell}"/>`;
      }
      if (cells) svg.querySelector("#plan-explored").insertAdjacentHTML("beforeend", cells);
      for (const p of m.trail || []) if (!trail.length || p[0] > trail[trail.length - 1][0]) trail.push(p);
      for (const s2 of m.sightings || []) { const k = s2.t + s2.object; if (!sightKeys.has(k)) { sightKeys.add(k); sightings.push(s2); } }
    }

    function chip(x, y, text, cls, title, extra = "") {
      const w = Math.max(28, text.length * 5.6 + 10);
      return `<g class="p-chip ${cls}${extra}" transform="translate(${x} ${y})"><rect x="0" y="-7" width="${w}" height="14" rx="7"/><text x="${w / 2}" y="3.5" text-anchor="middle">${esc(text)}</text><title>${esc(title)}</title></g>`;
    }

    function render(msg) {
      if (!svg || !lay || !msg.runtime) return;
      const rt = msg.runtime, b = rt.belief || {}, tr = msg.truth || {};
      const looked = b.looked || {}, objs = b.objects || {}, now = performance.now();
      // what changed since the last frame glows for a few seconds
      const where = Object.fromEntries(Object.entries(objs).map(([id, o]) => [id, o.where]));
      if (prevWhere) for (const [id, w] of Object.entries(where)) if (prevWhere[id] !== w) fresh[id] = now;
      prevWhere = where;

      // spots, with fog where it never looked
      let spots = "";
      const pos = {}, labels = [];
      const hit = (a, list) => list.some((b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h);
      for (const [name, s] of Object.entries(lay.surfaces || {})) {
        const [u, v] = api.uv(s.x, s.z); pos[name] = [u, v];
        const seen = !!looked[name], user = lay.user_surface === name;
        const text = short(name), w = text.length * 5.2;
        let ly = null;                                    // above the pad, else below, else only in the tooltip
        for (const y of [v - 13, v + 21]) { const box = { x: u - w / 2, y: y - 8, w, h: 10 }; if (!hit(box, labels)) { labels.push(box); ly = y; break; } }
        spots += `<g class="p-spot${seen ? "" : " fog"}${user ? " user" : ""}"><rect x="${u - 9}" y="${v - 9}" width="18" height="18" rx="4"/>
          ${ly != null ? `<text x="${u}" y="${ly}" text-anchor="middle">${esc(text)}</text>` : ""}
          <title>${esc(name)}${s.desc ? " · " + esc(s.desc) : ""}\n${seen ? `looked at t=${looked[name].t}${looked[name].source === "memory" ? " (remembered)" : ""}` : "never looked at"}${user ? "\nyour spot: deliveries go here" : ""}</title></g>`;
      }
      for (const [id, lm] of Object.entries(b.landmarks || {})) {
        const p = pos[lm.near]; if (!p) continue;
        spots += `<g class="p-landmark"><rect x="${p[0] - 5}" y="${p[1] + 11}" width="10" height="10" rx="2"/><text x="${p[0] + 8}" y="${p[1] + 20}">${esc(lm.label)}</text><title>${esc(id)} near ${esc(lm.near)} (${esc(lm.source)})</title></g>`;
      }
      svg.querySelector("#plan-spots").innerHTML = spots;

      // chips: what belief says is on each spot
      const bySpot = {}, held = [];
      for (const [id, o] of Object.entries(objs)) {
        const w = o.where || "";
        if (w.startsWith("hand")) { held.push([id, o]); continue; }
        const spot = pos[w] ? w : (o.usual && pos[o.usual] ? o.usual : ((o.history || []).slice(-1)[0] || {}).surface);
        if (!spot || !pos[spot]) continue;
        (bySpot[spot] = bySpot[spot] || []).push([id, o, !pos[w]]);
      }
      let chips = "", lines = "";
      const chipAt = {}, placed = [];
      const spotsSorted = Object.entries(bySpot).sort((a, b2) => pos[a[0]][1] - pos[b2[0]][1] || pos[a[0]][0] - pos[b2[0]][0]);
      for (const [spot, list] of spotsSorted) {
        const [u, v] = pos[spot];
        list.sort((a, b2) => a[0].localeCompare(b2[0]));
        let y0 = v + 2;                                    // first chip beside the pad, pushed down past earlier chips
        const w0 = Math.max(...list.slice(0, MAX_CHIPS).map(([id, o]) => nice(o.type || id).length * 5.6 + 12), 30);
        const n = Math.min(list.length, MAX_CHIPS) + (list.length > MAX_CHIPS ? 1 : 0);
        while (hit({ x: u + 13, y: y0 - 7, w: w0, h: n * CHIP_H }, placed)) y0 += CHIP_H;
        placed.push({ x: u + 13, y: y0 - 7, w: w0, h: n * CHIP_H });
        if (y0 > v + 2) lines += `<line class="p-lead" x1="${u + 9}" y1="${v}" x2="${u + 13}" y2="${y0}"/>`;
        list.slice(0, MAX_CHIPS).forEach(([id, o, away], i) => {
          const src = away ? KIND.gone : kindOf(o);
          const hist = (o.history || []).slice(-3).reverse().map((h) => h.surface).join(" ← ");
          const title = `${id} (${o.label || o.type})\n${away ? "not where expected; drawn at its usual place" : "on " + o.where}\nsource: ${src.label}${o.verified ? ", verified" : ""} · t=${o.t}` +
            `${o.usual ? "\nusually: " + o.usual : ""}${hist ? "\nlast seen: " + hist : ""}${o.mem_status ? "\nmemory status: " + o.mem_status : ""}`;
          const x = u + 13, y = y0 + i * CHIP_H;
          chipAt[id] = [x, y];
          chips += chip(x, y, (away ? "? " : "") + nice(o.type || id), src.cls, title, now - (fresh[id] || -1e9) < FRESH_MS ? " fresh" : "");
        });
        if (list.length > MAX_CHIPS) chips += `<text class="p-more" x="${u + 13}" y="${y0 + 4 + MAX_CHIPS * CHIP_H}">+${list.length - MAX_CHIPS} more</text>`;
      }
      // truth ghosts: where the simulator disagrees with belief
      let wrong = 0, unseen = 0;
      if (showTruth) {
        const ghostCount = {};
        for (const [id, t] of Object.entries(tr.objects || {})) {
          const o = objs[id];
          if (!o) { unseen += 1; continue; }
          if (!t.where || t.where === o.where || String(t.where).startsWith("hand") || !pos[t.where]) continue;
          wrong += 1;
          const [u, v] = pos[t.where], i = (ghostCount[t.where] = (ghostCount[t.where] || 0) + 1);
          const gx = u - 13 - 60, gy = v + 2 + (i - 1) * 16;
          chips += chip(gx, gy, nice(o.type || id), "ghost", `truth: ${id} is on ${t.where}`);
          const a = chipAt[id];
          if (a) lines += `<line class="p-diff" x1="${a[0]}" y1="${a[1]}" x2="${gx + 30}" y2="${gy}"/>`;
        }
      }
      svg.querySelector("#plan-chips").innerHTML = chips;
      svg.querySelector("#plan-lines").innerHTML = lines;

      // the path it walked, the route it is on, the follower's lookahead, stuck points, blocked edges
      const me = msg.view || tr.robot || null;                // its own pose estimate (odometry), else truth
      svg.querySelector("#plan-trail").setAttribute("points", trail.map((p) => api.uv(p[1], p[2]).join(",")).join(" "));
      const nav = msg.nav || {};
      const route = nav.route || null;
      const rpts = route && me ? [[me.x, me.z], ...route] : [];
      svg.querySelector("#plan-route").setAttribute("points", rpts.map(([x, z]) => api.uv(x, z).join(",")).join(" "));
      let navh = "";
      if (nav.lookahead) { const [lu, lv] = api.uv(nav.lookahead[0], nav.lookahead[1]); navh += `<circle class="p-look" cx="${lu}" cy="${lv}" r="3.5"><title>path follower lookahead (0.6 m)</title></circle>`; }
      for (const p of nav.stuck || []) { const [su, sv] = api.uv(p[0], p[1]); navh += `<g class="p-stuck" transform="translate(${su} ${sv})"><path d="M-5 -5L5 5M5 -5L-5 5"/><title>stuck here: a virtual obstacle was added and the path replanned</title></g>`; }
      for (const e of nav.blocked_edges || []) {
        const a = lay.keypoints[e[0]], b = lay.keypoints[e[1]]; if (!a || !b) continue;
        const [au, av] = api.uv(a.x, a.z), [bu, bv] = api.uv(b.x, b.z);
        navh += `<line class="p-blocked" x1="${au}" y1="${av}" x2="${bu}" y2="${bv}"><title>blocked: ${esc(e[0])} → ${esc(e[1])}</title></line>`;
      }
      if (nav.reach_stance) { const [ru, rv] = api.uv(nav.reach_stance.x, nav.reach_stance.z); navh += `<g class="p-stance" transform="translate(${ru} ${rv})"><rect x="-4" y="-4" width="8" height="8" transform="rotate(45)"/><title>reach_stance: the short step check_reachability asked for</title></g>`; }
      svg.querySelector("#plan-nav").innerHTML = navh;
      // the head camera's view fan: one wedge per row, from its near edge on the floor out to its range
      let vh = "";
      if (msg.view && msg.view.rows) {
        const v = msg.view, px = (x, z) => api.uv(x, z).join(",");
        for (const row of v.rows) {
          const a0 = (v.yaw - row.half) * Math.PI / 180, a1 = (v.yaw + row.half) * Math.PI / 180, n = 16, pts = [];
          for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; pts.push(px(v.x + Math.sin(a) * v.range, v.z + Math.cos(a) * v.range)); }
          for (let i = n; i >= 0; i--) { const a = a0 + (a1 - a0) * i / n; pts.push(px(v.x + Math.sin(a) * row.near, v.z + Math.cos(a) * row.near)); }
          vh += `<polygon class="p-fan" points="${pts.join(" ")}"><title>head camera view, ${row.tilt} deg down: from ${row.near} m to ${v.range} m, ${2 * row.half} deg across</title></polygon>`;
        }
      }
      svg.querySelector("#plan-view").innerHTML = vh;
      let sight = "";
      for (const f of sightings) {
        const p = pos[f.place]; if (!p) continue;
        const [u, v] = api.uv(f.x, f.z);
        sight += `<g class="p-found"><line x1="${u}" y1="${v}" x2="${p[0]}" y2="${p[1]}"/><path d="M${u} ${v - 5}L${u + 5} ${v}L${u} ${v + 5}L${u - 5} ${v}Z"/>
          <title>found ${esc(f.object)} on ${esc(f.place)} at t=${f.t}, seen from here</title></g>`;
      }
      svg.querySelector("#plan-sight").innerHTML = sight;

      // the robot (its own pose) and you
      let ag = "";
      if (lay.human) { const [u, v] = api.uv(lay.human.x, lay.human.z); ag += `<g class="p-human" transform="translate(${u} ${v})"><circle r="11"/><text y="4" text-anchor="middle">you</text></g>`; }
      if (me) {
        const [u, v] = api.uv(me.x, me.z);
        const walking = (rt.active || []).find((e) => e.tool === "navigate");
        const stones = new Set(api.fallbacks ? api.fallbacks() : ["kinematic_nav", "kinematic_attach", "sonic_arm_script"]);
        const fb = walking && stones.has(walking.executor);
        ag += `<g class="p-robot${fb ? " fb" : ""}" transform="translate(${u} ${v})"><g transform="rotate(${me.yaw})"><circle r="11"/><path d="M-5 -9L0 -17L5 -9"/></g>`;
        if (walking && walking.executor) ag += `<text class="p-exec${fb ? " fb" : ""}" x="0" y="24" text-anchor="middle">${esc(walking.executor)}${fb ? " (fallback)" : ""}</text>`;
        held.forEach(([id, o], i) => { ag += chip(14, -8 + i * CHIP_H, "holds " + nice(o.type || id), kindOf(o).cls, `${id} in the ${o.where.replace("hand:", "")} hand (${o.source}${o.verified ? ", verified" : ""})`); });
        ag += `</g>`;
      }
      svg.querySelector("#plan-agents").innerHTML = ag;

      const html = `<span title="Verified by a look or the camera"><i class="sw seen"></i>seen</span>
        <span title="From an earlier session, not checked yet"><i class="sw mem"></i>remembered</span>
        <span title="From an action's result, not seen yet"><i class="sw skill"></i>unverified</span>
        <span title="It looked and the object wasn't there"><i class="sw gone"></i>not there</span>
        <span title="A spot it has never looked at"><i class="sw fogsw"></i>never looked</span>
        <span title="Floor its head camera has covered this session"><i class="sw exsw"></i>explored</span>
        <span title="The head camera's view fan (both scan rows while scanning)"><i class="sw fansw"></i>view</span>
        ${showTruth ? `<span class="${wrong ? "bad" : ""}" title="Where the simulator disagrees with belief (ghost chips)">truth: ${wrong} wrong · ${unseen} never seen</span>` : ""}`;
      if (legend.dataset.h !== html) { legend.innerHTML = html; legend.dataset.h = html; }
      const stats = root.querySelector("#plan-stats"), st = `· ${explored.size} cells · ${trail.length} poses · ${sightings.length} finds`;
      if (stats.textContent !== st) stats.textContent = st;
    }
    return { mount, reset, render, robotMap };
  };
})();
