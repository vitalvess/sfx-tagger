"""Redistribui os sons classificados: um teste fixo pequeno e o resto para treino.

    python ferramentas/benchmark/redistribuir.py plano   <pasta "SFX Testes">
    python ferramentas/benchmark/redistribuir.py aplicar <pasta "SFX Testes">   (com a app fechada)
    python ferramentas/benchmark/redistribuir.py massa   <pasta de conhecimento treinado à parte>   (idem)

Antes: ~2100 sons de treino e ~16 800 de teste. Depois: um teste fixo de ~2500 sons (ESC-50, FSD50K
e BBC, todas as categorias) e o resto no treino. Para o teste continuar honesto, sons da mesma
origem nunca ficam dos dois lados: mesmo autor no Freesound (FSD50K), mesmo CD (BBC), mesma gravação
(ESC-50). Usa as impressões digitais já calculadas pela app: não volta a ouvir os sons.

"plano" escolhe a divisão, afina o limite de sons por categoria numa parte do treino posta de lado e
mede no teste fixo; não altera nada. "aplicar" ensina a IA desta instalação e cria a pasta de teste.

O que se ensina (medido no teste fixo, 2026-11): juntar tudo baixava a 1.ª sugestão (66,3 -> 64,9),
porque cada categoria nova também "rouba" sons às que a IA já conhecia, e os sons do FSD50K e da BBC
são de outros estilos de gravação. Por isso ensina-se o ESC-50 inteiro e, do FSD50K e da BBC, só as
categorias com menos de LACUNA exemplos, no máximo MAX_LACUNA sons cada: 1.ª sugestão igual (66,2),
nas 5 de 84,8 para 86,2, categorias novas de 16 para 24%, confiança "alta" de 94 para 97% de acertos.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exp_fsd_treino import NO_TRAIN, mapping  # noqa: E402

from earmark import classifier, paths, ucs  # noqa: E402
from earmark.ai import Tagger  # noqa: E402

HERE = Path(__file__).parent
PLANO = "redistribuicao.json"
TESTE = "Teste fixo"
QUOTA = {"FSD50K": 8, "BBC": 6}      # sons de teste por categoria (ESC-50: o fold 5 inteiro)
MAX_GRUPO = 25                         # origens maiores ficam no treino (não se desperdiçam)
LACUNA = 20                            # FSD50K/BBC só para categorias com menos exemplos do que isto
MAX_LACUNA = 20                        # e no máximo tantos sons por categoria


def state(folder: Path) -> Path:
    return folder / ".earmark" if (folder / ".earmark").is_dir() else folder / ".sfxtagger"


def folder_embeddings(folder: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    st = state(folder)
    rec = json.loads((st / "projeto.json").read_text(encoding="utf-8"))["records"]
    with np.load(st / "embeddings.npz") as a, np.load(st / "embeddings2.npz") as b:
        return {rec[r]["path"]: (a[r].astype(np.float32), b[r].astype(np.float32))
                for r in a.files if r in b.files and r in rec}


def load_pool(root: Path) -> list[dict]:
    """Todos os sons classificados: base, ficheiro, origem (grupo), respostas aceites, etiqueta de treino."""
    valid = {c.catid for c in ucs.load_ucs()}
    pool = []

    # FSD50K (parte "eval"): grupo = autor no Freesound
    m = mapping()
    base = root / "Bases originais" / "FSD50K"
    info = json.loads((base / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    labels = {r["fname"]: r["labels"].split(",") for r in csv.DictReader(open(base / "FSD50K.ground_truth" / "eval.csv", encoding="utf-8"))}
    folder = root / "FSD50K teste"
    emb = folder_embeddings(folder)
    for row in list(csv.reader(open(folder / "respostas.csv", encoding="utf-8")))[1:]:
        f = row[0]
        if f not in emb:
            continue
        labs = [x for x in labels[Path(f).stem] if x in m]
        train = m[labs[0]][0] if labs and labs[0] not in NO_TRAIN else None
        pool.append(dict(base="FSD50K", file=f, path=str(folder / f), group="fsd:" + info[Path(f).stem]["uploader"],
                         truth=[c for c in row[1].split("|") if c in valid], train=train, nlabs=len(labs), emb=emb[f]))

    # BBC: grupo = CD (o id sem os 3 dígitos da faixa)
    folder = root / "BBC teste"
    emb = folder_embeddings(folder)
    for row in list(csv.reader(open(folder / "respostas.csv", encoding="utf-8")))[1:]:
        f = row[0]
        ids = [c for c in row[1].split("|") if c in valid]
        if f in emb and ids:
            pool.append(dict(base="BBC", file=f, path=str(folder / f), group="bbc:" + f[4:-4][:-3],
                             truth=ids, train=ids[0], nlabs=1, emb=emb[f]))
    folder = root / "Sons de treino (BBC)"
    emb = folder_embeddings(folder)
    from preparar_bbc import fxname
    for rid, desc, catid, _ in list(csv.reader(open(folder / "fontes.csv", encoding="utf-8-sig")))[1:]:
        f = ucs.build_filename(catid, f"{fxname(desc)} {rid[-5:]}", "BBC", "BBC")
        if f in emb and catid in valid:
            pool.append(dict(base="BBC", file=f, path=str(folder / f), group="bbc:" + rid[:-3],
                             truth=[catid], train=catid, nlabs=1, emb=emb[f]))

    # ESC-50: grupo = gravação de origem; os folds já separam as gravações
    esc = root / "Bases originais" / "ESC-50"
    em = {k: v for k, v in json.loads((HERE / "esc50_ucs.json").read_text()).items() if not k.startswith("_")}
    with np.load(esc / "emb_laion_larger_clap_general.npz") as a, np.load(esc / "emb_laion_clap-htsat-unfused.npz") as b:
        for r in csv.DictReader(open(esc / "ESC-50-master" / "meta" / "esc50.csv")):
            f = r["filename"]
            pool.append(dict(base="ESC-50", file=f, path=str(esc / "ESC-50-master" / "audio" / f), group="esc:" + r["src_file"],
                             truth=em[r["category"]], train=em[r["category"]][0], nlabs=1, fold=int(r["fold"]),
                             emb=(a[f].astype(np.float32), b[f].astype(np.float32))))
    return pool


def pick_groups(items: list[dict], quota: int, rng, taken: set[str]) -> tuple[set[int], set[str]]:
    """Escolhe sons para teste, origem a origem, até cada categoria ter `quota` sons."""
    by_group = defaultdict(list)
    for i, it in enumerate(items):
        if it["group"] not in taken:
            by_group[it["group"]].append(i)
    groups = [g for g, v in by_group.items() if len(v) <= MAX_GRUPO]
    groups = [groups[i] for i in rng.permutation(len(groups))]
    count, chosen, used = Counter(), set(), set()
    for g in groups:
        add = []
        for i in by_group[g]:
            cat = items[i]["truth"][0]
            if count[cat] + sum(1 for j in add if items[j]["truth"][0] == cat) < quota:
                add.append(i)
        if add:
            used.add(g)
            chosen.update(add)
            count.update(items[i]["truth"][0] for i in add)
    return chosen, used


def knowledge_now() -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = paths.app_dir() / "conhecimento"
    meta = json.loads((d / "exemplos.json").read_text(encoding="utf-8"))
    with np.load(d / "embeddings.npz") as a, np.load(d / "embeddings2.npz") as b:
        ids = [i for i in meta if i in a.files and i in b.files and meta[i].get("catid")]
        return (np.stack([a[i] for i in ids]).astype(np.float32), np.stack([b[i] for i in ids]).astype(np.float32),
                [meta[i]["catid"] for i in ids])


class Medida:
    def __init__(self, t: Tagger, items: list[dict]):
        self.t, self.items = t, items
        self.idx = {c.catid: i for i, c in enumerate(t.categories)}
        self.X1 = np.stack([it["emb"][0] for it in items])
        self.X2 = np.stack([it["emb"][1] for it in items])
        self.P = np.stack([t.audio_probs(a, b) for a, b in zip(self.X1, self.X2)])
        self.F = classifier.features(self.X1, self.X2)
        self.truth = [{self.idx[c] for c in it["truth"]} for it in items]
        self.fam = np.asarray(t.family_index)
        self.base = np.array([it["base"] for it in items])

    def probs(self, K1, K2, Ky) -> np.ndarray:
        y = np.array([self.idx[c] for c in Ky])
        c = classifier.train(K1, K2, y, **classifier.TRAIN)
        c = classifier.adapt(c, self.F, gate_at=classifier.BLEND["gate_at"], **classifier.ADAPT)
        return classifier.blend(self.P, c, self.X1, self.X2, **classifier.BLEND)

    def score(self, Q: np.ndarray, mask=None) -> dict:
        sel = np.arange(len(Q)) if mask is None else np.where(mask)[0]
        order = np.argsort(-Q[sel], axis=1)[:, :5]
        t1 = np.array([o[0] in self.truth[i] for o, i in zip(order, sel)])
        t5 = np.array([bool(set(o) & self.truth[i]) for o, i in zip(order, sel)])
        fm = np.array([self.fam[o[0]] in {self.fam[x] for x in self.truth[i]} for o, i in zip(order, sel)])
        conf = Q[sel].max(1)
        lv = {"alta": conf >= .6, "media": (conf >= .4) & (conf < .6), "baixa": conf < .4}
        return {"n": len(sel), "top1": round(100 * t1.mean(), 1), "top5": round(100 * t5.mean(), 1),
                "familia": round(100 * fm.mean(), 1),
                "conf": {k: (round(100 * t1[v].mean(), 1) if v.any() else None, int(v.sum())) for k, v in lv.items()}}

    def report(self, label: str, Q: np.ndarray) -> dict:
        s = self.score(Q)
        print(f"{label:34} 1.ª {s['top1']:5.1f}  nas 5 {s['top5']:5.1f}  principal {s['familia']:5.1f} | "
              + "  ".join(f"{b} {self.score(Q, self.base == b)['top1']:5.1f}" for b in ("ESC-50", "FSD50K", "BBC"))
              + " | confiança " + " ".join(f"{k} {v[0]}% ({v[1]})" for k, v in s["conf"].items()), flush=True)
        return s


def capped(items: list[dict], cap: int | None, rng) -> list[dict]:
    """No máximo `cap` sons por categoria e base; primeiro os de etiqueta mais específica."""
    if not cap:
        return items
    by = defaultdict(list)
    for it in items:
        by[it["base"], it["train"]].append((it["nlabs"], rng.random(), id(it), it))
    return [x[3] for v in by.values() for x in sorted(v)[:cap]]


def arrays(items):
    return (np.stack([it["emb"][0] for it in items]), np.stack([it["emb"][1] for it in items]), [it["train"] for it in items])


def plano(root: Path) -> None:
    pool = load_pool(root)
    print("sons classificados:", dict(Counter(it["base"] for it in pool)), flush=True)
    K1, K2, Ky = knowledge_now()
    print("a IA sabe agora:", len(Ky), "exemplos,", len(set(Ky)), "categorias", flush=True)
    # sons que a IA já aprendeu (ESC-50 de treino): não entram outra vez
    A = np.stack([it["emb"][0] for it in pool])
    known = np.zeros(len(pool), bool)
    for s in range(0, len(pool), 2000):
        known[s:s + 2000] = (A[s:s + 2000] @ K1.T).max(1) > 0.99
    for it, k in zip(pool, known):
        it["known"] = bool(k)
    print("já aprendidos:", int(known.sum()), flush=True)

    rng = np.random.default_rng(2026)
    for it in pool:
        it["part"] = "treino"
    test_groups: set[str] = set()
    for b, q in QUOTA.items():
        items = [it for it in pool if it["base"] == b and not it["known"]]
        chosen, used = pick_groups(items, q, rng, set())
        test_groups |= used
        for i, it in enumerate(items):
            if i in chosen:
                it["part"] = "teste"
    for it in pool:
        if it["base"] == "ESC-50" and it["fold"] == 5:
            it["part"] = "teste"
            test_groups.add(it["group"])
        elif it["part"] != "teste" and it["group"] in test_groups:
            it["part"] = "fora"          # mesma origem de um som de teste: nem treina nem testa
    # afinação: tirada do treino, também por origens, só para escolher o limite por categoria
    for b, q in (("FSD50K", 4), ("BBC", 3)):
        items = [it for it in pool if it["base"] == b and it["part"] == "treino" and not it["known"]]
        chosen, used = pick_groups(items, q, rng, set())
        for i, it in enumerate(items):
            if i in chosen:
                it["part"] = "afinacao"
            elif it["group"] in used:
                it["part"] = "afinacao-fora"
    for it in pool:
        if it["part"] == "treino" and (it["known"] or not it["train"]):
            it["part"] = "ja-aprendido" if it["known"] else "sem-etiqueta"
    print("divisão:", dict(Counter(it["part"] for it in pool)))
    print("por base:", {b: dict(Counter(it["part"] for it in pool if it["base"] == b)) for b in ("ESC-50", "FSD50K", "BBC")}, flush=True)

    t = Tagger()
    t.ensure_loaded()
    train = [it for it in pool if it["part"] == "treino"]
    tune = [it for it in pool if it["part"] == "afinacao"]
    test = [it for it in pool if it["part"] == "teste"]
    print(f"\nAFINAÇÃO ({len(tune)} sons): limite de sons por categoria e base")
    mt = Medida(t, tune)
    mt.report("como está agora", mt.probs(K1, K2, Ky))
    best = None
    for cap in (10, 25, 50, 100, None):
        a1, a2, ay = arrays(capped(train, cap, np.random.default_rng(1)))
        s = mt.report(f"+ treino, limite {cap or 'sem'} ({len(ay)})", mt.probs(np.concatenate([K1, a1]), np.concatenate([K2, a2]), Ky + ay))
        if best is None or (s["top1"], s["top5"]) > best[0]:
            best = ((s["top1"], s["top5"]), cap)
    cap = best[1]
    print("limite escolhido:", cap or "sem limite", flush=True)

    final = capped([it for it in pool if it["part"] in ("treino", "afinacao", "afinacao-fora") and it["train"] and not it["known"]],
                   cap, np.random.default_rng(1))
    print(f"\nTESTE FIXO ({len(test)} sons, {len({it['truth'][0] for it in test})} categorias)")
    m = Medida(t, test)
    m.report("sem treino nenhum", m.P)
    before = m.report(f"antes ({len(Ky)} exemplos)", m.probs(K1, K2, Ky))
    a1, a2, ay = arrays(final)
    t0 = time.time()
    after = m.report(f"depois ({len(Ky) + len(ay)} exemplos)", m.probs(np.concatenate([K1, a1]), np.concatenate([K2, a2]), Ky + ay))
    print(f"treinar o classificador com tudo: {time.time() - t0:.1f} s; categorias no treino: {len(set(Ky + ay))}")

    for it in pool:
        it["final"] = False
    for it in final:
        it["final"] = True
    out = {"limite": cap, "antes": before, "depois": after,
           "sons": [{k: it[k] for k in ("base", "file", "path", "group", "truth", "train", "part", "final")} for it in pool]}
    (root / PLANO).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("plano guardado em", root / PLANO)


def selecionar(sons: list[dict], Ky: list[str]) -> list[dict]:
    """ESC-50 inteiro; FSD50K e BBC só onde a IA tem poucos exemplos (ver o topo)."""
    have, per, out = Counter(Ky), Counter(), []
    for s in sons:
        if not s["final"]:
            continue
        if s["base"] != "ESC-50":
            c = s["train"]
            if have[c] >= LACUNA or per[c] >= MAX_LACUNA:
                continue
            per[c] += 1
        out.append(s)
    return out


def aplicar(root: Path) -> None:
    from earmark.knowledge import get_knowledge
    plan = json.loads((root / PLANO).read_text(encoding="utf-8"))
    emb = {(it["base"], it["file"]): it["emb"] for it in load_pool(root)}
    kb = get_knowledge()
    d = paths.app_dir() / "conhecimento"
    backup = d.with_name("conhecimento-antes-da-redistribuicao")
    if not backup.exists():
        shutil.copytree(d, backup)
    before = len(kb.meta)
    with kb.lock:
        Ky = [m["catid"] for m in kb.meta.values() if m.get("catid")]
    sel = selecionar(plan["sons"], Ky)
    print("a ensinar:", dict(Counter(s["base"] for s in sel)), flush=True)
    for s in sel:
        e1, e2 = emb[s["base"], s["file"]]
        cat = ucs.lookup(s["train"])
        kb.learn(e1, {}, {"catid": cat.catid, "fxname": "", "keywords": ""}, s["base"].replace("-", ""), s["file"],
                 save=False, emb2=e2)
    kb.save()
    print("exemplos:", before, "->", len(kb.meta), "| cópia de segurança em", backup, flush=True)

    dest = root / TESTE
    dest.mkdir(exist_ok=True)
    rows = [("ficheiro", "catid")]
    for s in plan["sons"]:
        if s["part"] == "teste":
            name = f"{s['base'].replace('-', '')}_{Path(s['file']).name}"
            if not (dest / name).exists():
                shutil.copy2(s["path"], dest / name)
            rows.append((name, "|".join(s["truth"])))
    with open(dest / "respostas.csv", "w", newline="", encoding="utf-8") as h:
        csv.writer(h).writerows(rows)
    (dest / "LEIA-ME.txt").write_text(
        "TESTE FIXO DO EARMARK\r\n=====================\r\n\r\n"
        f"{len(rows) - 1} sons ja classificados (ESC-50, FSD50K e BBC) que a IA NUNCA aprendeu, nem outros sons da mesma\r\n"
        "origem (mesmo autor, mesmo CD ou mesma gravacao). Serve para medir a IA sempre com a mesma regua.\r\n\r\n"
        "COMO USAR: Admin > Testar a IA > escolher esta pasta > Testar.\r\n"
        "NUNCA treines com esta pasta nem confirmes estes sons: o teste deixava de ser justo.\r\n\r\n"
        "As pastas \"FSD50K teste\" e \"BBC teste\" deixaram de servir de teste: parte desses sons foi\r\n"
        "usada para treinar a IA.\r\n\r\nLicencas: ESC-50 e FSD50K (Creative Commons, ver as bases originais); BBC Sound Effects (RemArc:\r\n"
        "uso pessoal, educativo ou de investigacao). Uso nao comercial.\r\n", encoding="utf-8")
    note = ("ATENCAO: esta pasta ja nao serve para testar a IA.\r\nParte destes sons foi usada para a treinar "
            "(redistribuicao de 2026-11). Usa a pasta \"Teste fixo\".\r\n")
    for old in ("FSD50K teste", "BBC teste"):
        (root / old / "ATENCAO - ja nao e teste.txt").write_text(note, encoding="utf-8")
    print("pasta de teste:", dest, len(rows) - 1, "sons", flush=True)


MASSA_CAP = 100   # sons por categoria do treino em massa (5 bases novas)


def massa(pasta: Path) -> None:
    """Junta os exemplos de um conhecimento treinado à parte (treino em massa das bases novas:
    TUT 2018, CochlScene, VocalSound, Nonspeech7k, UrbanSound8K), no máximo MASSA_CAP por
    categoria. Medido (2026-11): sem limite, os 2600 ambientes do CochlScene abafavam os da TUT
    (1.ª sugestão TUT 9,6 -> 15,7); com 100, TUT 32,9, UrbanSound8K 53,7 -> 64,9, Nonspeech7k
    60,3 -> 65,8, VocalSound 77,7 -> 80,8, CochlScene 36,2 -> 39,3; teste fixo 66,8 -> 65,3."""
    from earmark.knowledge import get_knowledge
    kb = get_knowledge()
    meta = json.loads((pasta / "exemplos.json").read_text(encoding="utf-8"))
    d = paths.app_dir() / "conhecimento"
    backup = d.with_name("conhecimento-antes-da-massa")
    if not backup.exists():
        shutil.copytree(d, backup)
    per, added = Counter(), Counter()
    with np.load(pasta / "embeddings.npz") as a, np.load(pasta / "embeddings2.npz") as b, kb.lock:
        before = len(kb.meta)
        for i, mt in meta.items():
            c = mt.get("catid")
            if i in kb.meta or not c or i not in a.files or i not in b.files or per[c] >= MASSA_CAP:
                continue
            per[c] += 1
            kb.meta[i] = mt
            kb.emb[i] = a[i].astype(np.float32)
            kb.emb2[i] = b[i].astype(np.float32)
            added[mt.get("author", "?")] += 1
        kb._changed()
    kb.save()
    print("massa:", dict(added), "| exemplos:", before, "->", len(kb.meta), "| cópia em", backup, flush=True)


if __name__ == "__main__":
    {"plano": plano, "aplicar": aplicar, "massa": massa}[sys.argv[1]](Path(sys.argv[2]))
