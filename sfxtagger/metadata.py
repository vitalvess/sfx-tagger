"""Ler e escrever metadados em ficheiros WAV/BWF.

Escreve três blocos, para máxima compatibilidade (Soundminer, BaseHead, Soundly, Reaper):
  * bext  (Broadcast Wave) -> Description, Originator, data/hora
  * iXML  -> bloco <USER> com os campos UCS (CATID, FXNAME, KEYWORDS, ...)
  * LIST/INFO -> INAM, ICMT, IKEY, IGNR, IART, IPRD

O áudio nunca é re-codificado: os chunks de áudio são copiados byte a byte para um
ficheiro temporário que depois substitui o original de forma atómica.
"""
from __future__ import annotations

import os
import shutil
import struct
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Campos que a app gere (nome interno -> tag iXML USER)
IXML_FIELDS = {
    "category": "CATEGORY",
    "subcategory": "SUBCATEGORY",
    "catid": "CATID",
    "category_full": "CATEGORYFULL",
    "fxname": "FXNAME",
    "description": "DESCRIPTION",
    "keywords": "KEYWORDS",
    "notes": "NOTES",
    "library": "LIBRARY",
    "designer": "DESIGNER",
    "microphone": "MICROPHONE",
    "rec_medium": "RECMEDIUM",
    "location": "LOCATION",
    "creator_id": "CREATORID",
    "source_id": "SOURCEID",
    "manufacturer": "MANUFACTURER",
    "url": "URL",
}

BEXT_FIXED = 602  # tamanho da parte fixa do chunk bext (v1/v2)


class MetadataError(Exception):
    pass


@dataclass
class Chunk:
    id: bytes
    offset: int  # posição dos dados (depois do cabeçalho de 8 bytes)
    size: int
    list_type: bytes | None = None


def _scan_chunks(f) -> list[Chunk]:
    f.seek(0)
    header = f.read(12)
    if len(header) < 12:
        raise MetadataError("Ficheiro demasiado pequeno")
    riff, _, wave = struct.unpack("<4sI4s", header)
    if riff == b"RF64":
        raise MetadataError("Ficheiros RF64 (>4 GB) ainda não são suportados")
    if riff != b"RIFF" or wave != b"WAVE":
        raise MetadataError("Não é um ficheiro WAV")
    file_size = os.fstat(f.fileno()).st_size
    chunks = []
    pos = 12
    while pos + 8 <= file_size:
        f.seek(pos)
        cid, size = struct.unpack("<4sI", f.read(8))
        data_off = pos + 8
        if data_off + size > file_size:  # chunk truncado: usa o que existe
            size = file_size - data_off
        ltype = None
        if cid == b"LIST" and size >= 4:
            ltype = f.read(4)
        chunks.append(Chunk(cid, data_off, size, ltype))
        pos = data_off + size + (size & 1)
    return chunks


def _read(f, ch: Chunk) -> bytes:
    f.seek(ch.offset)
    return f.read(ch.size)


def _cstr(b: bytes) -> str:
    b = b.split(b"\x00", 1)[0]
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return ""


# --- Leitura -----------------------------------------------------------------

def read_metadata(path: str | Path) -> dict:
    """Devolve {'fields': {...}, 'bext': {...}|None, 'has_ixml': bool, 'info': {...}}."""
    out = {"fields": {}, "bext": None, "has_ixml": False, "info": {}}
    with open(path, "rb") as f:
        chunks = _scan_chunks(f)
        for ch in chunks:
            if ch.id == b"bext" and ch.size >= BEXT_FIXED:
                out["bext"] = _parse_bext(_read(f, ch))
            elif ch.id == b"iXML":
                out["has_ixml"] = True
                out["fields"].update(_parse_ixml_user(_read(f, ch)))
            elif ch.id == b"LIST" and ch.list_type == b"INFO":
                out["info"] = _parse_info(_read(f, ch))
    fields = out["fields"]
    info = out["info"]
    if not fields.get("description"):
        desc = (out["bext"] or {}).get("description") or info.get("ICMT", "")
        if desc:
            fields["description"] = desc
    if not fields.get("keywords") and info.get("IKEY"):
        fields["keywords"] = info["IKEY"]
    if not fields.get("fxname") and info.get("INAM"):
        fields["fxname"] = info["INAM"]
    if not fields.get("designer") and info.get("IART"):
        fields["designer"] = info["IART"]
    if not fields.get("library") and info.get("IPRD"):
        fields["library"] = info["IPRD"]
    return out


def _parse_bext(data: bytes) -> dict:
    desc, orig, origref, date, time = struct.unpack("<256s32s32s10s8s", data[:338])
    return {
        "description": _cstr(desc),
        "originator": _cstr(orig),
        "originator_reference": _cstr(origref),
        "origination_date": _cstr(date),
        "origination_time": _cstr(time),
    }


def _parse_ixml_user(data: bytes) -> dict:
    try:
        root = ET.fromstring(data.rstrip(b"\x00").strip())
    except ET.ParseError:
        return {}
    user = root.find("USER")
    if user is None:
        return {}
    rev = {v: k for k, v in IXML_FIELDS.items()}
    out = {}
    for el in user:
        key = rev.get(el.tag.upper())
        if key and (el.text or "").strip():
            out[key] = el.text.strip()
    return out


def _parse_info(data: bytes) -> dict:
    out = {}
    pos = 4  # salta "INFO"
    while pos + 8 <= len(data):
        cid, size = struct.unpack("<4sI", data[pos:pos + 8])
        out[cid.decode("latin-1")] = _cstr(data[pos + 8:pos + 8 + size])
        pos += 8 + size + (size & 1)
    return out


# --- Escrita -----------------------------------------------------------------

def _fit(s: str, n: int) -> bytes:
    """O bext só aceita ASCII: tira acentos ('Pássaro' -> 'Passaro') e corta em n bytes."""
    s = unicodedata.normalize("NFKD", s or "")
    b = s.encode("ascii", errors="ignore")
    return b[:n].ljust(n, b"\x00")


def _build_bext(old: bytes | None, fields: dict) -> bytes:
    now = datetime.now()
    if old and len(old) >= BEXT_FIXED:
        buf = bytearray(old)
    else:
        buf = bytearray(BEXT_FIXED)
        buf[256:288] = _fit("SFX Tagger", 32)
        buf[320:330] = _fit(now.strftime("%Y-%m-%d"), 10)
        buf[330:338] = _fit(now.strftime("%H:%M:%S"), 8)
        struct.pack_into("<H", buf, 346, 1)  # versão 1
    buf[0:256] = _fit(fields.get("description", ""), 256)
    if fields.get("designer"):
        buf[256:288] = _fit(fields["designer"], 32)
    return bytes(buf)


def _build_ixml(old: bytes | None, fields: dict) -> bytes:
    root = None
    if old:
        try:
            root = ET.fromstring(old.rstrip(b"\x00").strip())
        except ET.ParseError:
            root = None
    if root is None or root.tag != "BWFXML":
        root = ET.Element("BWFXML")
        ET.SubElement(root, "IXML_VERSION").text = "2.10"
    user = root.find("USER")
    if user is None:
        user = ET.SubElement(root, "USER")
    for key, tag in IXML_FIELDS.items():
        val = (fields.get(key) or "").strip()
        el = user.find(tag)
        if val:
            if el is None:
                el = ET.SubElement(user, tag)
            el.text = val
        elif el is not None:
            user.remove(el)
    xml = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8")
    return xml


def _build_info(old: bytes | None, fields: dict) -> bytes:
    entries: dict[bytes, str] = {}
    if old:
        for k, v in _parse_info(old).items():
            entries[k.encode("latin-1")[:4].ljust(4)] = v
    mapping = {
        b"INAM": fields.get("fxname", ""),
        b"ICMT": fields.get("description", ""),
        b"IKEY": fields.get("keywords", ""),
        b"IGNR": fields.get("category_full", ""),
        b"IART": fields.get("designer", ""),
        b"IPRD": fields.get("library", ""),
        b"ISFT": "SFX Tagger",
    }
    for k, v in mapping.items():
        if v:
            entries[k] = v
        else:
            entries.pop(k, None)
    body = bytearray(b"INFO")
    for k, v in entries.items():
        val = v.encode("utf-8") + b"\x00"
        body += k + struct.pack("<I", len(val)) + val
        if len(val) & 1:
            body += b"\x00"
    return bytes(body)


def _write_chunk(out, cid: bytes, data: bytes) -> None:
    out.write(cid + struct.pack("<I", len(data)))
    out.write(data)
    if len(data) & 1:
        out.write(b"\x00")


def _copy_chunk(f, out, ch: Chunk) -> None:
    out.write(ch.id + struct.pack("<I", ch.size))
    f.seek(ch.offset)
    remaining = ch.size
    while remaining:
        block = f.read(min(remaining, 4 * 1024 * 1024))
        if not block:
            raise MetadataError("Leitura incompleta do ficheiro")
        out.write(block)
        remaining -= len(block)
    if ch.size & 1:
        out.write(b"\x00")


def write_metadata(path: str | Path, fields: dict, backup_dir: Path | None = None) -> None:
    """Grava os campos no WAV. Se backup_dir for dado, guarda lá uma cópia do original."""
    path = Path(path)
    tmp = path.with_name(path.name + ".sfxtagger-tmp")
    with open(path, "rb") as f:
        chunks = _scan_chunks(f)
        if not any(c.id == b"data" for c in chunks):
            raise MetadataError("WAV sem chunk de áudio (data)")
        old_bext = next((_read(f, c) for c in chunks if c.id == b"bext"), None)
        old_ixml = next((_read(f, c) for c in chunks if c.id == b"iXML"), None)
        old_info = next((_read(f, c) for c in chunks
                         if c.id == b"LIST" and c.list_type == b"INFO"), None)
        new = [
            (b"bext", _build_bext(old_bext, fields)),
            (b"iXML", _build_ixml(old_ixml, fields)),
            (b"LIST", _build_info(old_info, fields)),
        ]
        try:
            with open(tmp, "wb") as out:
                out.write(b"RIFF\x00\x00\x00\x00WAVE")
                inserted = False
                for ch in chunks:
                    if ch.id in (b"bext", b"iXML") or (ch.id == b"LIST" and ch.list_type == b"INFO"):
                        continue
                    if ch.id == b"data" and not inserted:
                        for cid, data in new:
                            _write_chunk(out, cid, data)
                        inserted = True
                    _copy_chunk(f, out, ch)
                total = out.tell()
                if total - 8 > 0xFFFFFFFF:
                    raise MetadataError("O ficheiro resultante excede 4 GB")
                out.seek(4)
                out.write(struct.pack("<I", total - 8))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / path.name
        if not dest.exists():  # guarda só o original mais antigo
            shutil.copy2(path, dest)
    os.replace(tmp, path)
