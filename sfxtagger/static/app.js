"use strict";

// ---------------------------------------------------------------- estado
const S = {
  records: [],          // lista leve (sem forma de onda)
  byId: new Map(),
  view: [],             // ids filtrados, na ordem mostrada
  active: null,         // id do som aberto no painel
  activeFull: null,     // registo completo (com forma de onda)
  selected: new Set(),
  filter: "all",
  search: "",
  ucs: [],
  ucsById: new Map(),
  defaults: {},
  settings: {},
  autoplay: true,
  polling: null,
};

const FIELD_LABELS = {
  catid: "CatID", fxname: "FXName", description: "Description", keywords: "Keywords",
  notes: "Notes", library: "Library", designer: "Designer", microphone: "Microphone",
  rec_medium: "RecMedium", location: "Location", creator_id: "CreatorID", source_id: "SourceID",
};
const MOD = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "Cmd" : "Ctrl";
const ICON_PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';
const DEFAULT_KEYS = ["library", "designer", "creator_id", "source_id", "microphone", "rec_medium", "location"];
const FXNAME_MAX = 30;

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDur = (s) => {
  if (s == null) return "";
  if (s < 60) return s.toFixed(1) + "s";
  return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
};
const basename = (p) => p.split("/").pop();

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 6000 : 3000);
}

// ---------------------------------------------------------------- arranque
// ---------------------------------------------------------------- instalação automática
function showSetup(s) {
  const busy = s.state === "a instalar" || s.state === "pendente";
  const err = s.state === "erro";
  $("#setupModal").classList.toggle("hidden", !(busy || err));
  if (!(busy || err)) return false;
  $("#setupTitle").textContent = err ? "A instalação não terminou" : "A preparar o SFX Tagger";
  $("#setupStep").textContent = err ? "Ocorreu um problema." : s.step || "A preparar…";
  $("#setupSpin").classList.toggle("hidden", err);
  $("#setupElapsed").textContent = s.elapsed ? `Tempo decorrido: ${Math.floor(s.elapsed / 60)} min ${s.elapsed % 60} s` : "";
  $("#setupLog").textContent = (s.log || []).join("\n");
  $("#setupErrFoot").classList.toggle("hidden", !err);
  $("#setupErr").textContent = err ? `${s.error}. Confirma a ligação à internet e tenta de novo.` : "";
  return true;
}

async function waitForSetup(st) {
  if (st.setup?.state === "pendente") await api("/api/setup/retry", { method: "POST" }).catch(() => {});
  while (showSetup(st.setup || {})) {
    await new Promise((r) => setTimeout(r, 2000));
    try { st = await api("/api/status"); } catch { /* servidor ocupado */ }
  }
  if (st.setup?.state === "ok" && $("#setupModal").dataset.shown) toast("Tudo pronto! Já podes carregar os teus sons.");
}
$("#setupRetry").onclick = async () => {
  await api("/api/setup/retry", { method: "POST" });
  waitForSetup(await api("/api/status"));
};

// app de Mac (sem consola): botão Sair e sinal de vida para a app se fechar sozinha
function setupStandalone(st) {
  if (!st.standalone) return;
  $("#btnQuit").classList.remove("hidden");
  const ping = () => fetch("/api/ping", { method: "POST" }).catch(() => {});
  ping();
  setInterval(ping, 30000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) ping(); });
  $("#btnQuit").onclick = async () => {
    await flushSave();
    try {
      await api("/api/quit", { method: "POST" });
      $("#quitModal").classList.remove("hidden");
    } catch (e) { toast(e.message, true); }
  };
}

async function init() {
  const st = await api("/api/status");
  S.settings = st.settings;
  setupStandalone(st);
  if (st.setup && st.setup.state !== "ok") {
    $("#setupModal").dataset.shown = "1";
    waitForSetup(st);
  }
  renderRecent();
  refreshTrainCount();
  S.ucs = await api("/api/ucs");
  S.ucs.forEach((c) => S.ucsById.set(c.catid, c));
  if (st.project) {
    loadProject(await api("/api/project"));
    if (st.job && st.job.running) startPolling();
  }
}

function renderRecent() {
  const box = $("#recentBox");
  const rec = S.settings.recent || [];
  if (!rec.length) { box.innerHTML = ""; return; }
  box.innerHTML = "<h3>Pastas recentes</h3>" +
    rec.map((p) => `<button data-path="${esc(p)}">${esc(p)}</button>`).join("");
  $$("button", box).forEach((b) => b.onclick = () => openFolder(b.dataset.path));
}

async function openFolder(path) {
  try {
    const res = await api("/api/open", { method: "POST", body: { path } });
    loadProject(res);
    const s = res.scan;
    toast(`${s.total} ficheiros WAV encontrados` + (s.added ? ` (${s.added} novos)` : ""));
    if (!s.total) toast("Não há ficheiros WAV nesta pasta nem nas subpastas.", true);
  } catch (e) { toast(e.message, true); }
}

function loadProject(p) {
  S.root = p.root;
  S.defaults = p.defaults || {};
  setRecords(p.records);
  $("#folderLabel").textContent = p.root;
  $("#empty").classList.add("hidden");
  $("#mainView").classList.remove("hidden");
  S.isUpload = !!p.is_upload;
  ["#btnAnalyze", "#btnWrite", "#btnDefaults", "#btnExport"].forEach((s) => $(s).disabled = false);
  if (p.job && p.job.running) startPolling();
  if (p.warning) toast(p.warning, true);
  if (!S.active && S.view.length) select(S.view[0]);
}

function setRecords(list) {
  S.records = list;
  S.byId = new Map(list.map((r) => [r.id, r]));
  for (const id of [...S.selected]) if (!S.byId.has(id)) S.selected.delete(id);
  renderList();
}

function mergeRecords(list) {
  for (const r of list) {
    S.byId.set(r.id, r);
    const i = S.records.findIndex((x) => x.id === r.id);
    if (i >= 0) S.records[i] = r; else S.records.push(r);
  }
  renderList();
}

// ---------------------------------------------------------------- lista
const statusOf = (r) => (r.error ? "erro" : r.status);

function matches(r) {
  if (S.filter !== "all" && statusOf(r) !== S.filter) return false;
  if (!S.search) return true;
  const f = r.fields;
  const hay = [r.path, f.catid, f.fxname, f.description, f.keywords, f.category, f.subcategory]
    .join(" ").toLowerCase();
  return S.search.toLowerCase().split(/\s+/).every((w) => hay.includes(w));
}

function renderList() {
  const counts = { all: S.records.length, novo: 0, sugerido: 0, revisto: 0, gravado: 0, erro: 0 };
  S.records.forEach((r) => counts[statusOf(r)]++);
  $$("#filters button").forEach((b) => {
    b.classList.toggle("on", b.dataset.f === S.filter);
    $("span", b).textContent = counts[b.dataset.f] || 0;
  });

  const rows = S.records.filter(matches).sort((a, b) => a.path.localeCompare(b.path));
  S.view = rows.map((r) => r.id);
  $("#rows").innerHTML = rows.map((r) => {
    const f = r.fields;
    const conf = r.suggestion?.categories?.[0];
    const confTxt = conf && conf.catid === f.catid ? Math.round(conf.score * 100) + "%" : "";
    const st = statusOf(r);
    return `<tr data-id="${r.id}" class="${r.id === S.active ? "active" : ""}">
      <td class="c-chk"><input type="checkbox" ${S.selected.has(r.id) ? "checked" : ""}></td>
      <td class="c-st"><span class="dot ${st}" title="${esc(st)}"></span></td>
      <td title="${esc(r.path)}">${r.dirty && r.status !== "gravado" ? '<span class="dirty-mark" title="Alterações por gravar"></span>' : ""}${esc(basename(r.path))}</td>
      <td class="c-dur">${fmtDur(r.tech?.duration)}</td>
      <td class="c-cat">${esc(f.catid)}</td>
      <td>${esc(f.fxname)}</td>
      <td class="muted">${esc(f.description)}</td>
      <td class="c-conf">${confTxt}</td></tr>`;
  }).join("");
  updateSelBar();
  updateAccuracy();
}

function updateAccuracy() {
  const reviewed = S.records.filter((r) => r.review?.verdict);
  const el = $("#accuracy");
  if (!reviewed.length) { el.classList.add("hidden"); return; }
  const right = reviewed.filter((r) => r.review.verdict === "certo").length;
  const top5 = reviewed.filter((r) => r.review.verdict === "top5").length;
  el.classList.remove("hidden");
  el.innerHTML = `IA acertou <b>${right}/${reviewed.length}</b> (${Math.round((right / reviewed.length) * 100)}%)` +
    (top5 ? ` · +${top5} nas 5 sugestões` : "");
}

function updateSelBar() {
  const n = S.selected.size;
  $("#selCount").textContent = `${n} selecionado${n === 1 ? "" : "s"}`;
  $("#btnBulk").disabled = !n;
  $("#btnApproveSel").disabled = !n;
  $("#selAll").checked = n > 0 && S.view.every((id) => S.selected.has(id));
}

$("#rows").addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (!tr) return;
  const id = tr.dataset.id;
  if (e.target.matches("input[type=checkbox]")) {
    e.target.checked ? S.selected.add(id) : S.selected.delete(id);
    updateSelBar();
    return;
  }
  if (e.shiftKey && S.active) {           // shift+clique: seleciona intervalo
    const a = S.view.indexOf(S.active), b = S.view.indexOf(id);
    const [lo, hi] = a < b ? [a, b] : [b, a];
    S.view.slice(lo, hi + 1).forEach((x) => S.selected.add(x));
    renderList();
    return;
  }
  if (e.ctrlKey || e.metaKey) {
    S.selected.has(id) ? S.selected.delete(id) : S.selected.add(id);
    renderList();
    return;
  }
  select(id);
});

$("#selAll").onchange = (e) => {
  S.view.forEach((id) => e.target.checked ? S.selected.add(id) : S.selected.delete(id));
  renderList();
};
$("#search").oninput = (e) => { S.search = e.target.value.trim(); renderList(); };
$$("#filters button").forEach((b) => b.onclick = () => { S.filter = b.dataset.f; renderList(); });

// ---------------------------------------------------------------- painel de detalhe
async function select(id, { scroll = true } = {}) {
  await flushSave();
  S.active = id;
  $$("#rows tr").forEach((tr) => tr.classList.toggle("active", tr.dataset.id === id));
  if (scroll) $(`#rows tr[data-id="${id}"]`)?.scrollIntoView({ block: "nearest" });
  try { S.activeFull = await api(`/api/file/${id}`); } catch { S.activeFull = S.byId.get(id); }
  if (S.active !== id) return;
  renderDetail();
  loadAudio(id);
}

function techTags(t) {
  if (!t) return '<span class="tag">ainda não analisado</span>';
  const tags = [
    fmtDur(t.duration),
    `${(t.samplerate / 1000).toFixed(t.samplerate % 1000 ? 1 : 0)} kHz`,
    t.bit_depth ? `${t.bit_depth}-bit` : t.subtype,
    t.channels === 1 ? "mono" : t.channels === 2 ? "stereo" : `${t.channels} canais`,
  ].map((x) => `<span class="tag">${esc(x)}</span>`);
  if (t.lufs != null) tags.push(`<span class="tag">${t.lufs} LUFS</span>`);
  tags.push(`<span class="tag ${t.peak_dbfs > -0.1 ? "warn" : ""}">pico ${t.peak_dbfs} dBFS</span>`);
  const kinds = { "one-shot": "one-shot", takes: `${t.events} takes`, ambiente: "ambiente", evento: "evento" };
  if (kinds[t.kind]) tags.push(`<span class="tag">${kinds[t.kind]}</span>`);
  if (t.clipped_samples > 10) tags.push(`<span class="tag err" title="Amostras no máximo digital">clipping</span>`);
  if (Math.abs(t.dc_offset) > 0.01) tags.push(`<span class="tag warn">DC offset</span>`);
  return tags.join("");
}

function renderDetail() {
  const r = S.activeFull;
  const el = $("#detail");
  if (!r) { el.innerHTML = '<div class="placeholder">Escolhe um som da lista.</div>'; return; }
  const f = r.fields;
  const sug = r.suggestion;
  const cat = S.ucsById.get(f.catid);

  const sugHtml = sug ? sug.categories.map((c, i) => `
      <button class="sug ${c.catid === f.catid ? "on" : ""}" data-catid="${esc(c.catid)}" title="Tecla ${i + 1}">
        <span class="k">${i + 1}</span><span class="id">${esc(c.catid)}</span>
        <span class="nm">${esc(c.category)} · ${esc(c.subcategory)}</span>
        <span class="pc">${Math.round(c.score * 100)}%</span>
        <span class="pbar"><div style="width:${Math.round(c.score * 100)}%"></div></span>
      </button>`).join("") : "";
  const kwChips = (sug?.keywords || []).concat(cat ? cat.synonyms.slice(0, 6) : [])
    .filter((k, i, a) => a.indexOf(k) === i)
    .map((k) => `<button class="chip" data-kw="${esc(k)}">+ ${esc(k)}</button>`).join("");

  el.innerHTML = `
    <div class="d-head">
      <div class="d-name">${esc(basename(r.path))}</div>
      <div class="d-path">${esc(r.path)}</div>
      <div class="tech">${techTags(r.tech)}</div>
    </div>
    ${r.error ? `<div class="err-box">Erro: ${esc(r.error)}</div>` : ""}
    <div class="wave-box">
      <div class="wave-wrap" id="waveWrap"><canvas id="wave"></canvas></div>
      <div class="transport">
        <button class="small icon" id="btnPlay" title="Tocar / parar">${ICON_PLAY}</button>
        <span class="time" id="ptime">0:00</span>
        <div class="spacer"></div>
        <label><input type="checkbox" id="autoplay" ${S.autoplay ? "checked" : ""}> tocar ao escolher</label>
        <span class="kbd">espaço</span>
      </div>
    </div>

    ${sug ? `<div class="section"><h3>Sugestões da IA</h3><div class="sugs">${sugHtml}</div>
      ${sug.memory?.length ? `<div class="note">Aprendido com sons já confirmados: ${sug.memory.slice(0, 3).map((m) =>
        `${m.n} som${m.n > 1 ? "s" : ""} parecido${m.n > 1 ? "s" : ""} confirmado${m.n > 1 ? "s" : ""} como <b>${esc(m.catid)}</b> (${Math.round(m.best * 100)}% semelhante)`).join("; ")}</div>` : ""}
      ${sug.used_name ? `<div class="note">Combinado com pistas do nome: <i>${esc([...(sug.name_words || []), ...(sug.folder_words || []).map((w) => w + " (pasta)")].join(", "))}</i></div>` : ""}
    </div>` : ""}

    <div class="section">
      <h3>Metadados ${verdictBadge(r)}</h3>
      ${sug ? `<div class="field heard">
        <label for="fHeard">O que ouves? <span class="muted">(opcional)</span></label>
        <div class="with-mic"><input id="fHeard" value="${esc(r.review?.heard)}" placeholder="ex.: iron gate slam, keys jingle, paper crumple…">${micButton("heard")}</div>
        <div class="hint">Descreve o som em poucas palavras, de preferência em inglês. A IA volta a ordenar as sugestões com essa pista (Enter).</div>
      </div>` : ""}
      <div class="field">
        <label>CatID (UCS) <span>${cat ? "" : f.catid ? '<span style="color:var(--err)">CatID desconhecido</span>' : ""}</span></label>
        <div class="cat-combo">
          <input id="fCatid" value="${esc(f.catid)}" placeholder="Escreve para procurar: door, metal, whoosh…" autocomplete="off">
          <div class="cat-results hidden" id="catResults"></div>
        </div>
        <div class="cat-info">${cat ? `${esc(cat.category)} › ${esc(cat.subcategory)} — ${esc(cat.explanation)}` : ""}</div>
      </div>
      <div class="field">
        <label>FXName <span class="count" id="fxCount"></span></label>
        <input data-f="fxname" value="${esc(f.fxname)}" placeholder="Nome curto, ex.: Door Wood Creak Slow">
      </div>
      <div class="field">
        <label>Description <span class="muted">ditar: descreve o som, o ambiente e o local</span></label>
        <div class="with-mic"><textarea data-f="description" rows="3">${esc(f.description)}</textarea>${micButton("description")}</div>
      </div>
      <div class="field">
        <label>Location <span class="muted">onde foi gravado (ex.: Lisboa, cozinha de uma casa antiga)</span></label>
        <div class="with-mic"><input data-f="location" value="${esc(f.location)}">${micButton("location")}</div>
      </div>
      <div class="field">
        <label>Keywords <span>separadas por vírgulas</span></label>
        <textarea data-f="keywords" rows="2">${esc(f.keywords)}</textarea>
        <div class="chips" id="kwChips">${kwChips}</div>
      </div>
      <details>
        <summary>Mais campos (Library, Designer, Microphone…)</summary>
        <div class="field"><label>Notes</label><div class="with-mic"><textarea data-f="notes" rows="2">${esc(f.notes)}</textarea>${micButton("notes")}</div></div>
        <div class="grid2">
          ${DEFAULT_KEYS.filter((k) => k !== "location").map((k) => `<div class="field"><label>${FIELD_LABELS[k]}</label>
            <input data-f="${k}" value="${esc(f[k])}"></div>`).join("")}
        </div>
      </details>
      <div class="field"><label>Nome UCS (se renomeares ao gravar)</label>
        <div class="ucsname" id="ucsName"></div></div>
    </div>

    <div class="actions">
      <button class="primary" id="btnApprove" title="Confirma a categoria e os campos deste som">Confirmar e seguinte <span class="kbd">${MOD}+Enter</span></button>
      ${r.group ? '<button id="btnGroup" title="Seleciona os outros takes da mesma série">Selecionar série</button>' : ""}
      ${sug ? '<button id="btnSimilar" title="Seleciona sons parecidos (pela IA)">Selecionar semelhantes</button>' : ""}
      <button id="btnReanalyze" title="Voltar a analisar este som">Reanalisar</button>
    </div>`;

  // eventos
  $$(".sug", el).forEach((b) => b.onclick = () => setCatid(b.dataset.catid));
  $$("[data-f]", el).forEach((inp) => inp.addEventListener("input", () => {
    queueSave({ [inp.dataset.f]: inp.value });
    updateDerived();
  }));
  $$("#kwChips .chip", el).forEach((b) => b.onclick = () => addKeyword(b.dataset.kw, b));
  setupCatCombo();
  const heard = $("#fHeard");
  if (heard) {
    const send = async () => {
      const text = heard.value.trim();
      if (text === (S.activeFull.review?.heard || "")) return;
      const id = S.activeFull.id;
      heard.disabled = true;
      try {
        const rec = await api("/api/heard", { method: "POST", body: { id, text } });
        if (S.active !== id) return;
        S.activeFull = rec;
        mergeRecords([rec]);
        renderDetail();
        if (text) toast("Sugestões atualizadas com a tua descrição. Escolhe com as teclas 1-5.");
      } catch (e) { toast(e.message, true); heard.disabled = false; }
    };
    heard.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.ctrlKey && !e.metaKey) { e.preventDefault(); send(); } });
    heard.addEventListener("change", send);
  }
  $("#btnApprove").onclick = approveAndNext;
  $("#btnPlay").onclick = togglePlay;
  $("#autoplay").onchange = (e) => (S.autoplay = e.target.checked);
  $("#btnGroup") && ($("#btnGroup").onclick = () => selectGroup(r));
  $("#btnSimilar") && ($("#btnSimilar").onclick = () => selectSimilar(r));
  $("#btnReanalyze").onclick = () => runAnalysis([r.id], true);
  $("#waveWrap").onclick = (e) => {
    const p = $("#player");
    if (!p.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    p.currentTime = ((e.clientX - rect.left) / rect.width) * p.duration;
    if (p.paused) p.play();
  };
  updateDerived();
  drawWave();
}

function verdictBadge(r) {
  const v = r.review?.verdict;
  if (!v) return "";
  const txt = { certo: "IA acertou", top5: "IA quase: estava nas sugestões", errado: "IA errou" }[v];
  const tip = v === "certo" ? "" : ` title="A IA sugeriu ${esc(r.review.ai_catid)}"`;
  return `<span class="verdict ${v}"${tip}>${txt}</span>`;
}

function updateDerived() {
  const r = S.activeFull;
  if (!r) return;
  const fx = $('[data-f="fxname"]')?.value || "";
  const c = $("#fxCount");
  if (c) { c.textContent = `${fx.length}/${FXNAME_MAX}`; c.classList.toggle("over", fx.length > FXNAME_MAX); }
  const get = (k) => $(`[data-f="${k}"]`)?.value ?? r.fields[k] ?? "";
  const clean = (s) => s.replace(/[<>:"/\\|?*_]/g, " ").replace(/\s+/g, " ").trim();
  const name = [r.fields.catid || "UNKNOWN", clean(fx) || "Untitled",
    clean(get("creator_id")) || "XX", clean(get("source_id")) || "XX"].join("_") + ".wav";
  $("#ucsName").textContent = name;
}

function addKeyword(kw, btn) {
  const ta = $('[data-f="keywords"]');
  const list = ta.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (!list.map((s) => s.toLowerCase()).includes(kw.toLowerCase())) list.push(kw);
  ta.value = list.join(", ");
  queueSave({ keywords: ta.value });
  btn?.remove();
}

// --- pesquisa de CatID
function searchUcs(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  const words = q.split(/\s+/);
  const scored = [];
  for (const c of S.ucs) {
    const id = c.catid.toLowerCase();
    const name = `${c.category} ${c.subcategory}`.toLowerCase();
    const syn = c.synonyms.join(" ").toLowerCase();
    const expl = c.explanation.toLowerCase();
    // palavras que não existem no UCS (ex.: "gravel") não eliminam o resultado,
    // mas quem coincide com mais palavras aparece primeiro
    let score = 0, hits = 0;
    for (const w of words) {
      let s = 0;
      if (id === w) s += 100;
      else if (id.startsWith(w)) s += 40;
      if (name.includes(w)) s += 20;
      if (syn.includes(w)) s += 8;
      if (expl.includes(w)) s += 3;
      if (s) { hits++; score += s; }
    }
    if (hits) scored.push([hits * 1000 + score, c]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, 40).map((x) => x[1]);
}

function setupCatCombo() {
  const inp = $("#fCatid"), box = $("#catResults");
  let hl = 0, results = [];
  const show = () => {
    results = searchUcs(inp.value);
    hl = 0;
    if (!results.length) { box.classList.add("hidden"); return; }
    box.innerHTML = results.map((c, i) => `<div data-catid="${esc(c.catid)}" class="${i === hl ? "hl" : ""}">
      <b>${esc(c.catid)}</b>${esc(c.category)} › ${esc(c.subcategory)}<small>${esc(c.explanation)}</small></div>`).join("");
    box.classList.remove("hidden");
  };
  inp.addEventListener("input", show);
  inp.addEventListener("focus", () => inp.select());
  inp.addEventListener("keydown", (e) => {
    const items = $$("div", box);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!items.length) return;
      hl = (hl + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((d, i) => d.classList.toggle("hl", i === hl));
      items[hl].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      if (results[hl]) setCatid(results[hl].catid);
      else if (S.ucsById.has(inp.value.trim())) setCatid(inp.value.trim());
    } else if (e.key === "Escape") {
      box.classList.add("hidden");
      inp.value = S.activeFull.fields.catid;
      inp.blur();
    }
  });
  inp.addEventListener("blur", () => setTimeout(() => {
    box.classList.add("hidden");
    if (S.activeFull && inp.value !== S.activeFull.fields.catid) inp.value = S.activeFull.fields.catid;
  }, 150));
  box.addEventListener("mousedown", (e) => {
    const d = e.target.closest("[data-catid]");
    if (d) { e.preventDefault(); setCatid(d.dataset.catid); }
  });
}

async function setCatid(catid) {
  const r = S.activeFull;
  if (!r) return;
  await flushSave();
  const res = await api("/api/update", { method: "POST", body: { ids: [r.id], fields: { catid } } });
  mergeRecords(res.records);
  Object.assign(r.fields, res.records[0].fields);
  r.status = res.records[0].status;
  renderDetail();
}

// --- gravação automática das edições (no estado do projeto, não no WAV)
let pending = null, pendingId = null, pendingTimer = null;
function queueSave(fields) {
  const id = S.active;
  if (pendingId && pendingId !== id) flushSave();
  pendingId = id;
  pending = { ...(pending || {}), ...fields };
  Object.assign(S.activeFull.fields, fields);
  clearTimeout(pendingTimer);
  pendingTimer = setTimeout(flushSave, 500);
}
async function flushSave(status) {
  clearTimeout(pendingTimer);
  const fields = pending, id = pendingId;
  pending = null; pendingId = null;
  if (!fields && !status) return;
  const target = id || S.active;
  const res = await api("/api/update", {
    method: "POST", body: { ids: [target], fields: fields || {}, status: status || null },
  });
  mergeRecords(res.records);
}

async function refreshTrainCount() {
  try {
    const before = S.trained || 0;
    const st = (await api("/api/knowledge")).stats;
    updateTrainBadge(st.total);
    if (st.total > before && st.total % 25 === 0)
      toast(`Já ensinaste ${st.total} sons à IA. Lembra-te de exportar o que ela aprendeu (Treinar a IA > Exportar).`);
  } catch { /* não é crítico */ }
}

async function approveAndNext() {
  const id = S.active;
  if (!id) return;
  pendingId = pendingId || id;
  await flushSave("revisto");
  refreshTrainCount();
  const i = S.view.indexOf(id);
  const next = S.view[i + 1] || S.view[i];
  if (next) select(next);
}

function selectGroup(r) {
  S.records.filter((x) => x.group && x.group === r.group).forEach((x) => S.selected.add(x.id));
  renderList();
  toast(`${S.selected.size} sons selecionados. Usa "Editar selecionados…" para copiar campos.`);
}
async function selectSimilar(r) {
  const sims = await api(`/api/similar/${r.id}`);
  S.selected.add(r.id);
  sims.forEach((s) => S.selected.add(s.id));
  renderList();
  toast(sims.length ? `${sims.length} sons semelhantes selecionados.` : "Não encontrei sons muito parecidos.");
}

// ---------------------------------------------------------------- áudio
const player = $("#player");
let peaks = [];

function loadAudio(id) {
  peaks = S.activeFull?.tech?.waveform || [];
  player.src = `/api/audio/${id}`;
  if (S.autoplay) player.play().catch(() => {});
  drawWave();
}
function togglePlay() { player.paused ? player.play().catch(() => {}) : player.pause(); }
player.addEventListener("play", () => $("#btnPlay") && ($("#btnPlay").innerHTML = ICON_PAUSE));
player.addEventListener("pause", () => $("#btnPlay") && ($("#btnPlay").innerHTML = ICON_PLAY));
player.addEventListener("ended", () => $("#btnPlay") && ($("#btnPlay").innerHTML = ICON_PLAY));
player.addEventListener("timeupdate", () => {
  const t = $("#ptime");
  if (t) t.textContent = `${fmtDur(player.currentTime)} / ${fmtDur(player.duration || 0)}`;
});
(function loop() {
  if (!player.paused) drawWave();
  requestAnimationFrame(loop);
})();

function drawWave() {
  const c = $("#wave");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const col = css.getPropertyValue("--wave").trim();
  const played = css.getPropertyValue("--wave-played").trim();
  const prog = player.duration ? player.currentTime / player.duration : 0;
  if (!peaks.length) {
    g.fillStyle = col; g.fillRect(0, h / 2, w, 1);
  } else {
    const n = peaks.length;
    for (let x = 0; x < w; x++) {
      const v = peaks[Math.floor((x / w) * n)] || 0;
      const bh = Math.max(1, v * (h - 6));
      g.fillStyle = x / w < prog ? played : col;
      g.fillRect(x, (h - bh) / 2, 1, bh);
    }
  }
  if (prog) { g.fillStyle = played; g.fillRect(Math.floor(prog * w), 0, 1, h); }
}
window.addEventListener("resize", drawWave);

// ---------------------------------------------------------------- análise
async function runAnalysis(ids, force) {
  try {
    await api("/api/analyze", { method: "POST", body: { ids, force } });
    startPolling();
  } catch (e) { toast(e.message, true); }
}

function startPolling() {
  $("#jobBox").classList.remove("hidden");
  $("#btnAnalyze").disabled = true;
  clearInterval(S.polling);
  let lastDone = -1, lastRefresh = 0, busy = false;
  S.polling = setInterval(async () => {
    if (busy) return;                     // pedido anterior ainda a decorrer
    busy = true;
    try { await pollJob(); } finally { busy = false; }
  }, 1000);

  async function pollJob() {
    let res;
    try { res = await api("/api/job"); } catch { return; }
    const j = res.job;
    const pct = j.total ? (j.done / j.total) * 100 : 0;
    $("#jobBar").style.width = pct + "%";
    $("#jobText").textContent = j.done === 0 && j.phase !== "a analisar"
      ? (res.model !== "pronto" ? res.model : j.phase)
      : `${j.done}/${j.total} · ${basename(j.current || "")}`;
    // atualiza a lista à medida que avança, mas no máximo a cada 3 s (bibliotecas grandes)
    if (j.running && j.done !== lastDone && Date.now() - lastRefresh > 3000) {
      lastDone = j.done;
      lastRefresh = Date.now();
      const p = await api("/api/project");
      setRecords(p.records);
      if (S.active && S.byId.get(S.active)?.suggestion && !S.activeFull?.suggestion) select(S.active, { scroll: false });
    }
    if (!j.running) {
      clearInterval(S.polling);
      $("#jobBox").classList.add("hidden");
      $("#btnAnalyze").disabled = false;
      const p = await api("/api/project");
      setRecords(p.records);
      // não redesenha o painel se a pessoa estiver a escrever num campo
      const typing = document.activeElement?.matches("#detail input, #detail textarea");
      if (S.active && !typing) select(S.active, { scroll: false });
      if (j.phase === "erro") toast(j.message, true);
      else toast(`Análise concluída: ${j.done} sons` + (j.errors ? `, ${j.errors} com erro` : "") + ".");
    }
  }
}

$("#btnAnalyze").onclick = (e) => { e.stopPropagation(); $("#analyzeMenu").classList.toggle("hidden"); };
$$("#analyzeMenu button").forEach((b) => b.onclick = () => {
  $("#analyzeMenu").classList.add("hidden");
  const mode = b.dataset.an;
  if (mode === "sel") {
    if (!S.selected.size) return toast("Não há sons selecionados.", true);
    runAnalysis([...S.selected], true);
  } else if (mode === "all") {
    if (confirm("Reanalisar todos os sons? Os campos que já editaste não são apagados.")) runAnalysis(null, true);
  } else runAnalysis(null, false);
});
document.addEventListener("click", () => $("#analyzeMenu").classList.add("hidden"));

// ---------------------------------------------------------------- carregar sons
const AUDIO_RE = /\.(wav|wave|bwf|aif|aiff|flac|ogg|mp3)$/i;

async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) => AUDIO_RE.test(f.name));
  const ignored = fileList.length - files.length;
  if (!files.length) return toast("Nenhum ficheiro de som reconhecido (WAV, AIFF, FLAC, OGG, MP3).", true);
  if (S.polling && !$("#jobBox").classList.contains("hidden")) return toast("Espera que a análise termine.", true);
  try {
    // os sons carregados vão para um lote novo (a não ser que já estejas num lote carregado)
    if (!S.isUpload) {
      S.active = null;
      loadProject(await api("/api/upload/new", { method: "POST", body: {} }));
    }
    $("#jobBox").classList.remove("hidden");
    let ok = 0;
    const failed = [];
    for (const [i, f] of files.entries()) {
      $("#jobText").textContent = `A carregar ${i + 1}/${files.length} · ${f.name}`;
      $("#jobBar").style.width = (i / files.length) * 100 + "%";
      const r = await fetch("/api/upload/file?name=" + encodeURIComponent(f.name), { method: "PUT", body: f });
      if (r.ok) ok++;
      else {
        let msg = r.statusText;
        try { msg = (await r.json()).detail || msg; } catch {}
        failed.push(msg);
      }
    }
    $("#jobBox").classList.add("hidden");
    loadProject(await api("/api/rescan", { method: "POST" }));
    if (failed.length) toast(`${failed.length} ficheiro(s) falharam: ${failed[0]}`, true);
    else toast(`${ok} sons carregados${ignored ? ` (${ignored} ficheiros ignorados)` : ""}. A analisar…`);
    if (ok) runAnalysis(null, false);
  } catch (e) {
    $("#jobBox").classList.add("hidden");
    toast(e.message, true);
  }
}

$("#btnUpload").onclick = () => $("#fileInput").click();
$("#btnUpload2").onclick = () => $("#fileInput").click();
$("#fileInput").onchange = (e) => { uploadFiles(e.target.files); e.target.value = ""; };

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $("#dropOverlay").classList.remove("hidden");
  $("#dropzone").classList.add("over");
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    $("#dropOverlay").classList.add("hidden");
    $("#dropzone").classList.remove("over");
  }
});
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("#dropOverlay").classList.add("hidden");
  $("#dropzone").classList.remove("over");
  uploadFiles(e.dataTransfer.files);
});

// menu exportar
$("#btnExport").onclick = (e) => { e.stopPropagation(); $("#exportMenu").classList.toggle("hidden"); };
$$("#exportMenu a").forEach((a) => a.addEventListener("click", async (e) => {
  $("#exportMenu").classList.add("hidden");
  await flushSave();
  if (a.id === "exReport" && !S.records.some((r) => r.review?.verdict)) {
    e.preventDefault();
    toast(`Ainda não confirmaste nenhum som. Ouve um som e carrega em "Confirmar e seguinte" (${MOD}+Enter).`, true);
  } else if (a.id === "exZip") toast("A preparar o ZIP… o download começa daqui a pouco.");
}));
document.addEventListener("click", () => $("#exportMenu").classList.add("hidden"));

// ---------------------------------------------------------------- ditado por voz
// Grava o microfone, converte para WAV mono 16 kHz no browser (igual em Chrome e Safari)
// e envia ao servidor, onde o Whisper transcreve (e opcionalmente traduz para inglês).
const MIC_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5A2.5 2.5 0 0 0 5.5 4v4a2.5 2.5 0 0 0 5 0V4A2.5 2.5 0 0 0 8 1.5z"/><path d="M3.5 7.5a.75.75 0 0 1 1.5 0 3 3 0 0 0 6 0 .75.75 0 0 1 1.5 0 4.5 4.5 0 0 1-3.75 4.43v1.57h1.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1 0-1.5h1.5v-1.57A4.5 4.5 0 0 1 3.5 7.5z"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="4" width="8" height="8" rx="1.5"/></svg>';
const DICT = { rec: null, stream: null, chunks: [], btn: null, timer: null };
// campos em que, por omissão, o ditado é traduzido para inglês (norma das bibliotecas)
const DICT_TRANSLATE = { description: true, heard: true, fxname: true, keywords: true, location: false, notes: false };

function micButton(field) {
  return `<button type="button" class="mic" data-mic="${field}" title="Ditar por voz (clica, fala e clica de novo para parar)">${MIC_ICON}</button>`;
}

async function toggleDictation(btn) {
  if (DICT.rec) { DICT.rec.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia) return toast("Este browser não permite gravar o microfone.", true);
  try {
    DICT.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch {
    return toast("Sem acesso ao microfone. Autoriza o microfone nas definições do browser.", true);
  }
  player.pause();
  DICT.chunks = [];
  DICT.btn = btn;
  DICT.rec = new MediaRecorder(DICT.stream);
  DICT.rec.ondataavailable = (e) => e.data.size && DICT.chunks.push(e.data);
  DICT.rec.onstop = () => finishDictation(btn);
  DICT.rec.start();
  btn.classList.add("rec");
  btn.innerHTML = STOP_ICON;
  toast("A ouvir… fala e clica no botão vermelho para terminar.");
  DICT.timer = setTimeout(() => DICT.rec && DICT.rec.stop(), 90000);   // limite de 90 s
}

async function finishDictation(btn) {
  clearTimeout(DICT.timer);
  DICT.stream.getTracks().forEach((t) => t.stop());
  const blob = new Blob(DICT.chunks, { type: DICT.rec.mimeType });
  DICT.rec = null;
  btn.classList.remove("rec");
  btn.classList.add("busy");
  btn.innerHTML = '<span class="spinner"></span>';
  const field = btn.dataset.mic;
  const translate = S.settings.dictate_translate !== false && DICT_TRANSLATE[field];
  const slow = setTimeout(() => toast("Na primeira vez o modelo de voz é descarregado (cerca de 1 GB). Pode demorar alguns minutos…"), 5000);
  try {
    const wav = await toWav16k(blob);
    const r = await fetch(`/api/dictate?translate=${translate ? "true" : "false"}`, { method: "PUT", body: wav });
    const res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.statusText);
    if (!res.text) toast("Não percebi nada. Tenta falar mais perto do microfone.", true);
    else insertDictation(field, res.text);
  } catch (e) {
    toast(e.message, true);
  } finally {
    clearTimeout(slow);
    btn.classList.remove("busy");
    btn.innerHTML = MIC_ICON;
  }
}

async function toWav16k(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(buf);
  ctx.close();
  const len = Math.ceil(decoded.duration * 16000);
  const off = new OfflineAudioContext(1, Math.max(len, 1), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, 16000, true); out.setUint32(28, 32000, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  return new Blob([out], { type: "audio/wav" });
}

function insertDictation(field, text) {
  if (field === "heard") {
    const inp = $("#fHeard");
    if (!inp) return;
    inp.value = text;                     // "o que ouves?" é uma pista curta: substitui
    inp.dispatchEvent(new Event("change"));
    return;
  }
  const el = $(`[data-f="${field}"]`);
  if (!el) return;
  const cur = el.value.trim();
  if (field === "keywords") el.value = cur ? `${cur}, ${text.replace(/\.$/, "")}` : text.replace(/\.$/, "");
  else el.value = cur ? `${cur.replace(/[.\s]*$/, ".")} ${text}` : text;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  toast("Texto ditado acrescentado.");
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-mic]");
  if (b) { e.preventDefault(); toggleDictation(b); }
});

// ---------------------------------------------------------------- conhecimento da IA
function storeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function storeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* sem armazenamento */ } }

function updateTrainBadge(total) {
  S.trained = total;
  $("#trainCount").textContent = total;
}

function renderKnowledge(st) {
  updateTrainBadge(st.total);
  const goal = Math.max(25, st.categories * 5);
  $("#kbBar").style.width = Math.min(100, (st.total / goal) * 100) + "%";
  $("#kbCount").textContent = st.total
    ? `${st.total} sons ensinados · ${st.categories} tipos de som diferentes`
    : "Ainda nenhum som ensinado";
  $("#kbMod").textContent = MOD;
  const last = storeGet("sfxtagger.lastExport");
  $("#kbLastExport").textContent = last ? `Última exportação: ${last}` : "";
  const pct = st.measured ? Math.round((st.right / st.measured) * 100) + "%" : "–";
  const authors = st.authors.map((a) => `<tr><td>${esc(a.author)}</td><td>${a.count}</td>
      <td>${a.medidos ? Math.round((a.certo / a.medidos) * 100) + "%" : "–"}</td>
      <td><button class="link-btn" data-forget="${esc(a.author)}">remover</button></td></tr>`).join("");
  const conf = st.confusions.map((c) => `<tr><td><code>${esc(c.ai)}</code></td><td><code>${esc(c.real)}</code></td><td>${c.n}×</td></tr>`).join("");
  $("#kbStats").innerHTML = `
    <div class="kb-cards">
      <div class="kb-card"><b>${st.total}</b><span>sons confirmados (exemplos)</span></div>
      <div class="kb-card"><b>${st.categories}</b><span>categorias UCS diferentes</span></div>
      <div class="kb-card"><b>${pct}</b><span>acerto da IA (1.ª sugestão)</span></div>
    </div>
    ${st.total ? `<h3>Quem contribuiu</h3>
      <table class="kb-table"><tr><th>Pessoa (CreatorID)</th><th>Exemplos</th><th>Acerto da IA</th><th></th></tr>${authors}</table>` :
      '<p class="muted">Ainda não há exemplos. Ouve sons e carrega em "Confirmar e seguinte", ou importa o conhecimento de um colega.</p>'}
    ${conf ? `<h3>O que a IA mais confunde</h3>
      <table class="kb-table"><tr><th>A IA disse</th><th>Era afinal</th><th>Vezes</th></tr>${conf}</table>` : ""}`;
  $$("[data-forget]").forEach((b) => b.onclick = async () => {
    if (!confirm(`Remover os exemplos de "${b.dataset.forget}" da memória da IA?`)) return;
    const res = await api("/api/knowledge/forget", { method: "POST", body: { author: b.dataset.forget } });
    renderKnowledge(res.stats);
    toast(`${res.removed} exemplos removidos.`);
  });
}

$("#btnKnowledge").onclick = async () => {
  await flushSave();
  const res = await api("/api/knowledge");
  if (!res.author) toast('Dica: preenche o teu CreatorID em "Biblioteca" para os teus exemplos ficarem com o teu nome.');
  $("#kbResult").classList.add("hidden");
  renderKnowledge(res.stats);
  $("#knowledgeModal").classList.remove("hidden");
};
$("#kbStart").onclick = () => {
  $("#knowledgeModal").classList.add("hidden");
  if (!S.root) {
    toast("Primeiro carrega os teus sons: arrasta-os para a janela ou usa \"Abrir uma pasta\".");
    return;
  }
  const pending = S.records.filter((r) => r.status === "sugerido");
  if (!pending.length) {
    toast(S.records.some((r) => r.status === "novo")
      ? "Primeiro analisa os sons: botão \"Analisar\"." : "Não há sons por rever nesta biblioteca. Carrega mais sons.");
    return;
  }
  S.filter = "sugerido";
  S.autoplay = true;
  renderList();
  select(S.view[0]);
  toast(`Modo treino: ouve o som; se a categoria estiver certa carrega ${MOD}+Enter, se não escolhe a certa (teclas 1-5) e depois ${MOD}+Enter.`);
};
$("#kbExport").addEventListener("click", () => {
  const d = new Date();
  storeSet("sfxtagger.lastExport", d.toLocaleDateString() + " " + d.toLocaleTimeString().slice(0, 5));
  setTimeout(() => toast("Ficheiro de conhecimento descarregado. Envia-o a quem coordena o projeto."), 400);
});
$("#kbImport").onclick = () => $("#kbFile").click();
$("#kbFile").onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const r = await fetch("/api/knowledge/import", { method: "PUT", body: f });
    const res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.statusText);
    renderKnowledge(res.stats);
    const x = res.result;
    const box = $("#kbResult");
    box.classList.remove("hidden");
    box.innerHTML = `Importado de <b>${esc(x.from || "colega")}</b>: ${x.new} exemplos novos` +
      (x.updated ? `, ${x.updated} atualizados` : "") + (x.same ? `, ${x.same} já existiam` : "") + "." +
      (S.root && S.records.some((r) => r.status === "sugerido")
        ? `<br><button class="primary" id="kbResuggest">Atualizar as sugestões dos sons por rever</button>` : "");
    $("#kbResuggest") && ($("#kbResuggest").onclick = resuggestAll);
  } catch (err) { toast(err.message, true); }
};

async function resuggestAll() {
  const btn = $("#kbResuggest");
  if (btn) { btn.disabled = true; btn.textContent = "A atualizar…"; }
  try {
    const res = await api("/api/resuggest", { method: "POST" });
    setRecords(res.records);
    if (S.active) select(S.active, { scroll: false });
    $("#knowledgeModal").classList.add("hidden");
    toast(res.changed ? `${res.changed} sons mudaram de categoria sugerida com o novo conhecimento.`
      : "Sugestões atualizadas (nenhuma categoria mudou).");
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------- navegador de pastas
const fm = { path: "" };
async function fmLoad(path) {
  try {
    const res = await api("/api/browse?path=" + encodeURIComponent(path));
    fm.path = res.path; fm.parent = res.parent;
    $("#fmPath").value = res.path;
    $("#fmList").innerHTML = res.dirs.length
      ? res.dirs.map((d) => `<div data-path="${esc(d.path)}">${esc(d.name)}</div>`).join("")
      : '<div class="muted">Sem subpastas</div>';
    $$("#fmList [data-path]").forEach((d) => d.ondblclick = d.onclick = () => fmLoad(d.dataset.path));
    $("#fmInfo").textContent = res.path ? `${res.wavs} WAV nesta pasta (as subpastas também são incluídas)` : "Escolhe um local";
    $("#fmOpen").disabled = !res.path;
    $("#fmUp").disabled = res.parent == null;
  } catch (e) { toast(e.message, true); }
}
function openFolderModal() {
  $("#folderModal").classList.remove("hidden");
  fmLoad(S.root || "");
}
$("#btnFolder").onclick = openFolderModal;
$("#btnFolder2").onclick = openFolderModal;
$("#fmUp").onclick = () => fmLoad(fm.parent || "");
$("#fmGo").onclick = () => fmLoad($("#fmPath").value.trim());
$("#fmPath").onkeydown = (e) => { if (e.key === "Enter") fmLoad(e.target.value.trim()); };
$("#fmOpen").onclick = () => { $("#folderModal").classList.add("hidden"); S.active = null; openFolder(fm.path); };

// ---------------------------------------------------------------- modais
$$(".modal:not([data-locked])").forEach((m) => {
  m.addEventListener("mousedown", (e) => { if (e.target === m) m.classList.add("hidden"); });
  $$("[data-close]", m).forEach((b) => b.onclick = () => m.classList.add("hidden"));
});

$("#btnDefaults").onclick = () => {
  $("#defaultsForm").innerHTML = DEFAULT_KEYS.map((k) => `<div class="field"><label>${FIELD_LABELS[k]}</label>
    <input data-d="${k}" value="${esc(S.defaults[k])}" placeholder="${k === "creator_id" ? "ex.: as tuas iniciais" : ""}"></div>`).join("");
  $("#defTranslate").checked = S.settings.dictate_translate !== false;
  $("#defaultsModal").classList.remove("hidden");
};
$("#defSave").onclick = async () => {
  S.settings.dictate_translate = $("#defTranslate").checked;
  api("/api/settings", { method: "POST", body: { dictate_translate: S.settings.dictate_translate } }).catch(() => {});
  const defaults = {};
  $$("[data-d]").forEach((i) => (defaults[i.dataset.d] = i.value.trim()));
  try {
    const p = await api("/api/defaults", { method: "POST", body: { defaults, apply_to_all: $("#defApply").checked } });
    S.defaults = p.defaults;
    setRecords(p.records);
    if (S.active) select(S.active, { scroll: false });
    $("#defaultsModal").classList.add("hidden");
    toast("Dados da biblioteca guardados.");
  } catch (e) { toast(e.message, true); }
};

// edição em lote
$("#btnBulk").onclick = () => {
  const n = S.selected.size;
  const src = S.activeFull?.fields || {};
  $("#bulkCount").textContent = n;
  $("#bulkForm").innerHTML = Object.entries(FIELD_LABELS).map(([k, label]) => `
    <div class="brow"><input type="checkbox" data-bk="${k}">
      <span class="lbl">${label}</span>
      <input data-bv="${k}" value="${esc(src[k])}" ${k === "catid" ? 'list="ucsList"' : ""}></div>`).join("") +
    `<datalist id="ucsList">${S.ucs.map((c) => `<option value="${esc(c.catid)}">${esc(c.category)} › ${esc(c.subcategory)}</option>`).join("")}</datalist>
    <p class="muted">Os valores vêm do som aberto no painel. Marca os campos a copiar.</p>`;
  $$("#bulkForm [data-bv]").forEach((i) => i.addEventListener("input", () => ($(`[data-bk="${i.dataset.bv}"]`).checked = true)));
  $("#bulkModal").classList.remove("hidden");
};
$("#bulkSave").onclick = async () => {
  const fields = {};
  $$("#bulkForm [data-bk]:checked").forEach((c) => (fields[c.dataset.bk] = $(`[data-bv="${c.dataset.bk}"]`).value.trim()));
  if (!Object.keys(fields).length) return toast("Marca pelo menos um campo.", true);
  if (fields.catid && !S.ucsById.has(fields.catid)) return toast(`CatID "${fields.catid}" não existe no UCS.`, true);
  await flushSave();
  const res = await api("/api/update", { method: "POST", body: { ids: [...S.selected], fields } });
  mergeRecords(res.records);
  $("#bulkModal").classList.add("hidden");
  if (S.active) select(S.active, { scroll: false });
  toast(`${res.records.length} sons atualizados.`);
};
$("#btnApproveSel").onclick = async () => {
  await flushSave();
  const res = await api("/api/update", { method: "POST", body: { ids: [...S.selected], status: "revisto" } });
  mergeRecords(res.records);
  toast(`${res.records.length} sons confirmados.`);
  refreshTrainCount();
};

// gravar
function writeTargets(scope) {
  const analyzed = S.records.filter((r) => r.status !== "novo");
  if (scope === "revisto") return S.records.filter((r) => r.status === "revisto").map((r) => r.id);
  if (scope === "sel") return [...S.selected];
  return analyzed.map((r) => r.id);
}
$("#btnWrite").onclick = async () => {
  await flushSave();
  $("#wcRev").textContent = writeTargets("revisto").length;
  $("#wcSel").textContent = S.selected.size;
  $("#wcAll").textContent = writeTargets("all").length;
  $("#wBackup").checked = S.settings.backup !== false;
  $("#wRename").checked = !!S.settings.rename;
  const unrev = S.records.filter((r) => r.status === "sugerido").length;
  $("#wWarn").textContent = unrev ? `Atenção: ${unrev} sons ainda não foram revistos. As sugestões da IA podem ter erros.` : "";
  $("#writeModal").classList.remove("hidden");
};
$("#wGo").onclick = async () => {
  const scope = $('[name="wscope"]:checked').value;
  const ids = writeTargets(scope);
  if (!ids.length) return toast("Não há sons para gravar nesta opção.", true);
  const rename = $("#wRename").checked, backup = $("#wBackup").checked;
  $("#wGo").disabled = true;
  try {
    const res = await api("/api/write", { method: "POST", body: { ids, rename, backup } });
    S.settings.rename = rename; S.settings.backup = backup;
    setRecords(res.records);
    $("#writeModal").classList.add("hidden");
    const r = res.result;
    toast(`${r.ok} ficheiros gravados` + (r.errors.length ? `, ${r.errors.length} com erro` : "") + ".", !!r.errors.length);
    if (S.active) select(S.active, { scroll: false });
  } catch (e) { toast(e.message, true); }
  $("#wGo").disabled = false;
};

// ---------------------------------------------------------------- teclado
document.addEventListener("keydown", (e) => {
  const typing = e.target.matches("input, textarea, select");
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); approveAndNext(); return; }
  if (e.key === "Escape" && typing) { e.target.blur(); return; }
  if (typing || $(".modal:not(.hidden)")) return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const i = S.view.indexOf(S.active);
    const next = S.view[Math.max(0, Math.min(S.view.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
    if (next && next !== S.active) select(next);
  } else if (/^[1-5]$/.test(e.key)) {
    const c = S.activeFull?.suggestion?.categories?.[+e.key - 1];
    if (c) setCatid(c.catid);
  } else if (e.key === "/") { e.preventDefault(); $("#search").focus(); }
});

init().catch((e) => toast("Erro ao iniciar: " + e.message, true));
