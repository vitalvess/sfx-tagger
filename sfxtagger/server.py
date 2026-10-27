"""Servidor web local (só acessível neste computador)."""
from __future__ import annotations

import json
import os
import string
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, paths, setup_ai, ucs
from .ai import get_tagger
from .knowledge import get_knowledge
from .project import Project

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SFX Tagger", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_stale_ui(request, call_next):
    """Depois de uma atualização da app, o browser não pode ficar com a interface antiga."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

_project: Project | None = None


# --- definições globais ------------------------------------------------------

def load_settings() -> dict:
    defaults = {"recent": [], "backup": True, "rename": False,
                "designer": "", "creator_id": ""}
    f = paths.settings_file()
    if f.exists():
        try:
            defaults.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return defaults


def save_settings(s: dict) -> None:
    paths.settings_file().write_text(json.dumps(s, ensure_ascii=False, indent=1),
                                     encoding="utf-8")


def project() -> Project:
    if _project is None:
        raise HTTPException(400, "Nenhuma pasta aberta")
    return _project


def _light(rec: dict) -> dict:
    """Registo sem os dados pesados (forma de onda), para a lista."""
    r = dict(rec)
    if r.get("tech"):
        r["tech"] = {k: v for k, v in r["tech"].items() if k not in ("waveform", "event_times")}
    return r


# --- páginas -----------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    tagger = get_tagger()
    return {"version": __version__, "model": tagger.status,
            "project": str(_project.root) if _project else None,
            "job": dict(_project.job) if _project else None,
            "settings": load_settings(),
            "setup": setup_ai.status(),
            "standalone": STANDALONE}


# --- instalação automática da IA e ciclo de vida da app ------------------------

# Lançada pela app de Mac (sem janela de consola): o servidor fecha-se sozinho
# quando o browser deixa de o usar, e a interface mostra um botão "Sair".
STANDALONE = os.environ.get("SFXTAGGER_APP") == "1"
IDLE_LIMIT = 20 * 60
_last_ping = time.time()


@app.post("/api/setup/retry")
def setup_retry():
    setup_ai.ensure_started()
    return setup_ai.status()


@app.post("/api/ping")
def ping():
    global _last_ping
    _last_ping = time.time()
    return {"ok": True}


@app.post("/api/quit")
def quit_app():
    if _project and _project.job["running"]:
        raise HTTPException(409, "Há uma análise a decorrer")
    if _project:
        _project.save()
    get_knowledge().save()
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"ok": True}


def _idle_watchdog() -> None:
    while True:
        time.sleep(60)
        busy = (_project and _project.job["running"]) or setup_ai.STATE["state"] == "a instalar"
        if not busy and time.time() - _last_ping > IDLE_LIMIT:
            if _project:
                _project.save()
            get_knowledge().save()
            os._exit(0)


if STANDALONE:
    threading.Thread(target=_idle_watchdog, daemon=True).start()


@app.get("/api/ucs")
def ucs_list():
    return [c.to_dict() for c in ucs.load_ucs()]


# --- pastas ------------------------------------------------------------------

@app.get("/api/browse")
def browse(path: str = ""):
    if not path:
        places = []
        home = Path.home()
        for p in (home / "Desktop", home / "Music", home / "Documents", home / "Downloads", home):
            if p.exists():
                places.append({"name": p.name or str(p), "path": str(p)})
        if sys.platform == "win32":
            for d in string.ascii_uppercase:
                root = f"{d}:\\"
                if os.path.exists(root):
                    places.append({"name": f"Disco {d}:", "path": root})
        else:
            vols = Path("/Volumes")
            if vols.exists():
                for v in sorted(vols.iterdir()):
                    places.append({"name": v.name, "path": str(v)})
        return {"path": "", "parent": None, "dirs": places, "wavs": 0}
    p = Path(path).expanduser()
    if not p.is_dir():
        raise HTTPException(404, "Pasta não encontrada")
    dirs, wavs = [], 0
    try:
        for e in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                dirs.append({"name": e.name, "path": str(e)})
            elif e.suffix.lower() == ".wav":
                wavs += 1
    except PermissionError:
        raise HTTPException(403, "Sem permissão para abrir esta pasta")
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "wavs": wavs}


class OpenReq(BaseModel):
    path: str


@app.post("/api/open")
def open_folder(req: OpenReq):
    global _project
    p = Path(req.path).expanduser()
    if not p.is_dir():
        raise HTTPException(404, "Pasta não encontrada")
    if _project and _project.job["running"]:
        raise HTTPException(409, "Espera que a análise atual termine")
    _project = Project(p)
    s = load_settings()
    # preenche o designer/creator por omissão com as definições do utilizador
    for k in ("designer", "creator_id"):
        if s.get(k) and not _project.defaults.get(k):
            _project.defaults[k] = s[k]
    result = _project.scan()
    _project.learn_existing()  # confirmações antigas desta biblioteca também ensinam a IA
    recent =[str(p.resolve())] + [r for r in s["recent"] if r != str(p.resolve())]
    s["recent"] = recent[:8]
    save_settings(s)
    return {"scan": result, **_summary()}


@app.post("/api/rescan")
def rescan():
    return {"scan": project().scan(), **_summary()}


def _is_upload_batch(p: Project) -> bool:
    try:
        p.root.relative_to(paths.uploads_dir().resolve())
        return True
    except ValueError:
        return False


def _summary() -> dict:
    p = project()
    s = p.summary()
    s["records"] = [_light(r) for r in s["records"]]
    s["is_upload"] = _is_upload_batch(p)
    return s


# --- carregar sons pelo browser ------------------------------------------------

UPLOAD_EXT = {".wav", ".wave", ".bwf", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}


class NewBatchReq(BaseModel):
    name: str = ""


@app.post("/api/upload/new")
def upload_new(req: NewBatchReq):
    """Cria uma pasta nova em Documentos/SFX Tagger e abre-a."""
    global _project
    if _project and _project.job["running"]:
        raise HTTPException(409, "Espera que a análise atual termine")
    from datetime import datetime
    name = _safe_name(req.name) or "Sons " + datetime.now().strftime("%Y-%m-%d %Hh%M")
    folder = paths.uploads_dir() / name
    n = 2
    while folder.exists() and any(folder.iterdir()):
        folder = paths.uploads_dir() / f"{name} ({n})"
        n += 1
    folder.mkdir(parents=True, exist_ok=True)
    _project = Project(folder)
    _project.scan()
    return _summary()


def _safe_name(s: str) -> str:
    import re
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s or "").strip(" .")
    return s[:120]


@app.put("/api/upload/file")
async def upload_file(request: Request, name: str):
    """Recebe um ficheiro (corpo do pedido em bruto). Formatos não-WAV são convertidos."""
    p = project()
    if not _is_upload_batch(p):
        raise HTTPException(400, "Os sons só podem ser carregados para um lote novo")
    fname = _safe_name(Path(name).name)
    ext = Path(fname).suffix.lower()
    if not fname or ext not in UPLOAD_EXT:
        raise HTTPException(415, f"Formato não suportado: {name}")
    stem = Path(fname).stem
    target = p.root / (stem + ".wav")
    n = 2
    while target.exists():
        target = p.root / f"{stem} ({n}).wav"
        n += 1
    tmp = p.root / (".upload-" + target.stem + ext)
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        if ext in (".wav", ".wave", ".bwf"):
            sf.info(str(tmp))  # confirma que é mesmo áudio
            os.replace(tmp, target)
        else:
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
            sf.write(str(target), data, sr, subtype="PCM_24")
    except Exception:
        target.unlink(missing_ok=True)
        raise HTTPException(415, f"{name} não é um ficheiro de áudio válido ou está danificado")
    finally:
        tmp.unlink(missing_ok=True)
    return {"saved": target.name}


@app.get("/api/download.zip")
def download_zip():
    """Todos os WAV da biblioteca (com os metadados gravados) num zip."""
    import tempfile
    import zipfile
    from starlette.background import BackgroundTask
    p = project()
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # WAV quase não comprime
        for rec in sorted(p.records.values(), key=lambda r: r["path"].lower()):
            f = p.abspath(rec)
            if not rec.get("missing") and f.exists():
                z.write(f, rec["path"])
    name = (p.root.name or "sons") + ".zip"
    return FileResponse(tmp, media_type="application/zip", filename=name,
                        background=BackgroundTask(lambda: os.unlink(tmp)))


# --- conhecimento da IA (memória de exemplos partilhável) ---------------------

def _author() -> str:
    if _project and _project.author:
        return _project.author
    s = load_settings()
    return s.get("creator_id") or s.get("designer") or ""


@app.get("/api/knowledge")
def knowledge_stats():
    return {"stats": get_knowledge().stats(), "author": _author()}


@app.get("/api/knowledge/export")
def knowledge_export():
    import re
    from datetime import datetime
    author = _author()
    data = get_knowledge().export_bytes(author)
    tag = re.sub(r"[^A-Za-z0-9_-]+", "", author) or "sfxtagger"
    name = f"conhecimento_{tag}_{datetime.now().strftime('%Y-%m-%d')}.sfxk"
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.put("/api/knowledge/import")
async def knowledge_import(request: Request):
    data = await request.body()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "Ficheiro demasiado grande")
    try:
        result = get_knowledge().import_bytes(data)
    except ValueError as e:
        raise HTTPException(415, str(e))
    return {"result": result, "stats": get_knowledge().stats()}


class ForgetReq(BaseModel):
    author: str


@app.post("/api/knowledge/forget")
def knowledge_forget(req: ForgetReq):
    n = get_knowledge().forget_author(req.author)
    return {"removed": n, "stats": get_knowledge().stats()}


@app.post("/api/resuggest")
def resuggest():
    p = project()
    if p.job["running"]:
        raise HTTPException(409, "Espera que a análise termine")
    changed = p.resuggest()
    return {"changed": changed, **_summary()}


# --- ditado por voz -------------------------------------------------------------

@app.put("/api/dictate")
async def dictate(request: Request, translate: bool = False):
    """Recebe um WAV gravado no browser e devolve o texto (Whisper, local)."""
    from starlette.concurrency import run_in_threadpool
    from .voice import get_dictation
    if not setup_ai.ai_installed():
        raise HTTPException(409, "Os componentes de IA ainda estão a ser instalados")
    data = await request.body()
    if not data:
        raise HTTPException(400, "Gravação vazia")
    try:
        text = await run_in_threadpool(get_dictation().transcribe, data, translate)
    except ValueError as e:
        raise HTTPException(415, f"Não foi possível transcrever: {e}")
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(500, "Não foi possível transcrever. Tenta de novo; se continuar, reinicia a app.")
    return {"text": text}


@app.get("/api/dictate/status")
def dictate_status():
    from .voice import get_dictation
    return {"status": get_dictation().status}


# --- confirmação: "o que ouves?" e relatório de precisão ----------------------

class HeardReq(BaseModel):
    id: str
    text: str


@app.post("/api/heard")
def heard(req: HeardReq):
    p = project()
    if req.id not in p.records:
        raise HTTPException(404)
    p.set_heard(req.id, req.text)
    return p.full_record(req.id)


@app.get("/api/report.csv")
def report_csv():
    p = project()
    data = "﻿" + p.report_csv()
    name = (p.root.name or "biblioteca") + "_relatorio_IA.csv"
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/project")
def get_project():
    return _summary()


@app.get("/api/file/{rid}")
def get_file(rid: str):
    p = project()
    if rid not in p.records:
        raise HTTPException(404)
    return p.full_record(rid)


@app.get("/api/similar/{rid}")
def similar(rid: str):
    return project().similar(rid)


# --- análise -----------------------------------------------------------------

class AnalyzeReq(BaseModel):
    ids: list[str] | None = None
    force: bool = False


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    if not project().start_analysis(req.ids, req.force):
        raise HTTPException(409, "Já há uma análise a decorrer")
    return {"started": True}


@app.get("/api/job")
def job():
    p = project()
    return {"job": dict(p.job), "model": get_tagger().status}


# --- edição ------------------------------------------------------------------

class UpdateReq(BaseModel):
    ids: list[str]
    fields: dict[str, str] = {}
    status: str | None = None


@app.post("/api/update")
def update(req: UpdateReq):
    if req.status and req.status not in ("novo", "sugerido", "revisto", "gravado"):
        raise HTTPException(400, "Estado inválido")
    p = project()
    p.update_fields(req.ids, req.fields, req.status)
    return {"records": [_light(p.records[i]) for i in req.ids if i in p.records]}


class DefaultsReq(BaseModel):
    defaults: dict[str, str]
    apply_to_all: bool = True
    remember: bool = True


@app.post("/api/defaults")
def defaults(req: DefaultsReq):
    p = project()
    p.set_defaults(req.defaults, req.apply_to_all)
    if req.remember:
        s = load_settings()
        for k in ("designer", "creator_id"):
            if req.defaults.get(k):
                s[k] = req.defaults[k]
        save_settings(s)
    return _summary()


class SettingsReq(BaseModel):
    dictate_translate: bool | None = None


@app.post("/api/settings")
def update_settings(req: SettingsReq):
    s = load_settings()
    if req.dictate_translate is not None:
        s["dictate_translate"] = req.dictate_translate
    save_settings(s)
    return s


class WriteReq(BaseModel):
    ids: list[str]
    rename: bool = False
    backup: bool = True


@app.post("/api/write")
def write(req: WriteReq):
    p = project()
    if p.job["running"]:
        raise HTTPException(409, "Espera que a análise termine antes de gravar")
    s = load_settings()
    s.update(rename=req.rename, backup=req.backup)
    save_settings(s)
    result = p.write(req.ids, req.rename, req.backup)
    return {"result": result, **_summary()}


@app.get("/api/export.csv")
def export_csv():
    p = project()
    data = "﻿" + p.export_csv()  # BOM para o Excel abrir acentos corretamente
    name = (p.root.name or "biblioteca") + "_metadados.csv"
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --- áudio -------------------------------------------------------------------

# o Safari (Mac) não toca WAV em vírgula flutuante; esses são convertidos para pré-escuta
BROWSER_OK = {"PCM_16", "PCM_24"}


@app.get("/api/audio/{rid}")
def audio_file(rid: str):
    p = project()
    rec = p.records.get(rid)
    if rec is None:
        raise HTTPException(404)
    path = p.abspath(rec)
    if not path.exists():
        raise HTTPException(404, "Ficheiro não encontrado")
    info = sf.info(str(path))
    if info.subtype in BROWSER_OK and info.channels <= 2 and info.samplerate <= 96000:
        return FileResponse(path, media_type="audio/wav")
    # formatos que o browser não toca (multicanal, 192 kHz, 32-bit int…): pré-escuta
    prev = p.state_dir / "preview" / f"{rid}.wav"
    if not prev.exists() or prev.stat().st_mtime < path.stat().st_mtime:
        prev.parent.mkdir(parents=True, exist_ok=True)
        from .audio import _resample
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 2:
            data = data[:, :2]
        if sr > 96000:
            import numpy as np
            data = np.stack([_resample(data[:, c], sr, 48000) for c in range(data.shape[1])], 1)
            sr = 48000
        sf.write(str(prev), data, sr, subtype="PCM_16")
    return FileResponse(prev, media_type="audio/wav")
