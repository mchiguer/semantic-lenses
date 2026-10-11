"""Web interface for the semantic lens selector (standard library only).

    python app.py            # then open http://127.0.0.1:8000
    python app.py --port 8123
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lenses import LENS_KEYS, LENS_TITLES, RuleLenses, TfidfBaseline, compare
from lenses.evaluation import all_texts, baseline_sims, load_pairs, tune_threshold

DEV, TEST = load_pairs("dev"), load_pairs("test")
BASELINE = TfidfBaseline(all_texts(DEV, TEST))
THRESHOLD = tune_threshold(baseline_sims(DEV, BASELINE), [p["same"] for p in DEV])
BACKENDS = {"rules": RuleLenses()}
if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        from lenses.llm import LLMLenses
        BACKENDS["llm"] = LLMLenses()
    except Exception as exc:  # anthropic package missing, bad key...
        print(f"LLM backend unavailable: {exc}")

MAX_LEN = 2000


def api_compare(payload: dict) -> dict:
    a, b = str(payload.get("a", "")).strip(), str(payload.get("b", "")).strip()
    if not a or not b:
        raise ValueError("Both texts are required.")
    if len(a) > MAX_LEN or len(b) > MAX_LEN:
        raise ValueError(f"Texts are limited to {MAX_LEN} characters.")
    lenses = [l for l in payload.get("lenses", list(LENS_KEYS)) if l in LENS_KEYS] or list(LENS_KEYS)
    backend = BACKENDS.get(payload.get("backend", "rules"), BACKENDS["rules"])
    cmp = compare(backend.profile(a), backend.profile(b), lenses)
    sim = BASELINE.similarity(a, b)
    return {
        "baseline": {"similarity": round(sim, 4), "threshold": round(THRESHOLD, 4), "same": sim >= THRESHOLD},
        "backend": backend.name,
        "titles": LENS_TITLES,
        "comparison": cmp.to_dict(),
    }


def api_meta() -> dict:
    return {
        "backends": {k: v.name for k, v in BACKENDS.items()},
        "lenses": {k: LENS_TITLES[k] for k in LENS_KEYS},
        "pairs": [{"id": p["id"], "a": p["a"], "b": p["b"], "set": s}
                for s, ps in (("dev", DEV), ("test", TEST)) for p in ps],
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic lenses</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e2e1db;--ok:#1f7a4d;--ko:#b3402a;--acc:#2f5fb3;--barbg:#ecebe5}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe5;--mut:#9a9a92;--card:#1f1f1c;--line:#34342f;--ok:#58c08a;--ko:#e8836b;--acc:#7da2e8;--barbg:#2a2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.4rem;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
label{font-size:.85rem;color:var(--mut)}textarea,select{width:100%;font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
textarea{min-height:84px;resize:vertical}.row{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:14px 0}
.row label{display:flex;gap:6px;align-items:center;color:var(--fg);font-size:.95rem}button{font:inherit;padding:9px 18px;border-radius:8px;border:0;background:var(--acc);color:#fff;cursor:pointer}
.banner{border-radius:10px;padding:14px 16px;margin:16px 0;border:1px solid var(--line);background:var(--card)}
.banner b{display:block;font-size:1.05rem}.ok{border-left:6px solid var(--ok)}.ko{border-left:6px solid var(--ko)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top;font-size:.92rem}th{color:var(--mut);font-weight:500}
.bar{display:flex;align-items:center;gap:6px;font-size:.8rem;margin:2px 0}.bar i{display:block;height:8px;background:var(--acc);border-radius:4px;min-width:2px}.bar span.t{width:96px;color:var(--mut)}
.pill{padding:2px 8px;border-radius:99px;font-size:.8rem;border:1px solid var(--line)}.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.ko{color:var(--ko);border-color:var(--ko)}
.gauge{height:10px;background:var(--barbg);border-radius:5px;position:relative;margin:8px 0}.gauge i{position:absolute;left:0;top:0;bottom:0;background:var(--acc);border-radius:5px}.gauge u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ko)}
details{margin-top:6px;color:var(--mut);font-size:.82rem}.err{color:var(--ko)}
</style></head><body><main>
<h1>Semantic lens selector</h1>
<p class="sub">Two texts that look alike can say different things. Pick the dimensions to read them through and see whether the meaning is the same.</p>
<div class="row"><div style="flex:1;min-width:240px"><label for="preset">Examples from the corpus</label><select id="preset"><option value="">— write my own texts —</option></select></div>
<div id="backendbox" style="display:none"><label for="backend">Backend</label><select id="backend"></select></div></div>
<div class="grid"><div><label for="a">Text A</label><textarea id="a">The number of malaria cases is rising in the district.</textarea></div>
<div><label for="b">Text B</label><textarea id="b">The number of malaria cases is falling in the district.</textarea></div></div>
<div class="row" id="lensbox"></div>
<div class="row"><button id="go">Compare</button><span id="err" class="err"></span></div>
<div id="out"></div>
</main><script>
const $=id=>document.getElementById(id);
function el(tag,attrs,...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==='class')e.className=v;else if(k==='style')e.setAttribute('style',v);else e[k]=v}
 for(const k of kids)e.append(k instanceof Node?k:document.createTextNode(k));return e}
let meta;
async function init(){
 meta=await (await fetch('/api/meta')).json();
 for(const p of meta.pairs){const o=el('option',{value:p.id},`[${p.set}] ${p.id} — ${p.a.slice(0,40)} / ${p.b.slice(0,40)}`);$('preset').append(o)}
 $('preset').onchange=()=>{const p=meta.pairs.find(x=>x.id===$('preset').value);if(p){$('a').value=p.a;$('b').value=p.b;run()}};
 for(const[k,t]of Object.entries(meta.lenses)){const c=el('input',{type:'checkbox',checked:true,value:k});c.onchange=run;$('lensbox').append(el('label',{},c,t))}
 const bs=Object.entries(meta.backends);if(bs.length>1){$('backendbox').style.display='';for(const[k,n]of bs)$('backend').append(el('option',{value:k},n));$('backend').onchange=run}
 $('go').onclick=run;run()}
function bars(r){const box=el('div');if(!r.determined){box.append(el('span',{style:'color:var(--mut)'},'— undetermined'));return box}
 r.labels.map((l,i)=>[l,r.vector[i]]).filter(x=>x[1]>0.01).sort((x,y)=>y[1]-x[1]).forEach(([l,w])=>{
  box.append(el('div',{class:'bar'},el('span',{class:'t'},l),el('i',{style:`width:${Math.round(w*120)}px`}),String(Math.round(w*100)/100)))});
 if(r.evidence.length){const d=el('details',{},el('summary',{},'evidence'));d.append(r.evidence.join(' ; '));box.append(d)}return box}
async function run(){
 $('err').textContent='';
 const lenses=[...document.querySelectorAll('#lensbox input:checked')].map(c=>c.value);
 const body={a:$('a').value,b:$('b').value,lenses,backend:$('backend').value||'rules'};
 const resp=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data=await resp.json();if(!resp.ok){$('err').textContent=data.error||'Error';return}
 const out=$('out');out.replaceChildren();const c=data.comparison,T=data.titles,bl=data.baseline;
 const names=c.differing.map(k=>T[k]).join(', ');
 out.append(el('div',{class:'banner '+(c.same?'ok':'ko')},el('b',{},c.same?'Same meaning under the selected lenses':`Different meaning — they diverge on: ${names}`),
  c.same?'None of the determined dimensions diverges.':'These texts may look alike on the surface, but they do not read the same way.'));
 const g=el('div',{class:'gauge'},el('i',{style:`width:${Math.round(bl.similarity*100)}%`}),el('u',{style:`left:${Math.round(bl.threshold*100)}%`}));
 out.append(el('div',{class:'card'},el('b',{},'Baseline (TF-IDF similarity): '+bl.similarity.toFixed(2)),g,
  el('span',{style:'color:var(--mut);font-size:.85rem'},`Red mark = threshold tuned on the dev set (${bl.threshold.toFixed(2)}). Baseline verdict: ${bl.same?'same meaning':'different meaning'}. `+
  'Caution: on this corpus lexical similarity is higher for near-identical pairs than for paraphrases, so the threshold is of little use.')));
 const tb=el('table',{},el('tr',{},...['Lens','Text A','Text B','Similarity','Verdict'].map(h=>el('th',{},h))));
 for(const k of Object.keys(c.per_lens)){const s=c.per_lens[k];const diff=c.differing.includes(k);
  const v=s===null?el('span',{class:'pill'},'ignored'):el('span',{class:'pill '+(diff?'ko':'ok')},diff?'differs':'same');
  tb.append(el('tr',{},el('td',{},T[k]),el('td',{},bars(c.profile_a[k])),el('td',{},bars(c.profile_b[k])),el('td',{},s===null?'—':s.toFixed(2)),el('td',{},v)))}
 out.append(el('div',{class:'card'},tb));
 out.append(el('div',{style:'color:var(--mut);font-size:.82rem'},'Backend: '+data.backend))}
init().catch(e=>{$('err').textContent=String(e)});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._json(200, api_meta())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/compare":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000:
                raise ValueError("Request too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, api_compare(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # backend error (LLM API, etc.)
            self._json(500, {"error": f"Internal error: {exc}"})

    def log_message(self, fmt: str, *args) -> None:  # silence
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Semantic lenses → http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
