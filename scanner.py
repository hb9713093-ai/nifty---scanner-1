#!/usr/bin/env python3
"""
Nifty 500 daily trend-flip scanner.

Trend colour logic (replicated from the SSL Hybrid baseline in Ricky's indicator):
  baseline = EMA(close, 30)
  channel  = baseline +/- 0.2 * EMA(TrueRange, 30)
  Blue  = close above upper channel
  Red   = close below lower channel
  Grey  = close inside channel
A "flip" is any day whose colour differs from the previous day.

Usage:
  python scanner.py                 full run, writes index.html
  python scanner.py --check AXISBANK  prints recent colours + last flips for one stock
  python scanner.py --limit 40      quick test on first 40 stocks
"""
import argparse
import io
import json
import os
import re
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- settings -----------------------------
MA_LEN = 30
KELT_MULT = 0.2
HISTORY = "5y"
BATCH_SIZE = 50
LAYOUT_ID = "AXNXjCz0"          # your saved TradingView layout (indicator pre-loaded)
FUND_MAX_AGE_DAYS = 7           # refresh PE/ROE/EPS weekly
FUND_TIME_BUDGET_S = 600        # max seconds per run spent on fundamentals
STALE_DAYS = 10                 # skip stocks whose last candle is older than this

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UNIVERSE_CACHE = "nifty500.csv"
FUND_CACHE = "fundamentals.json"
DHAN_CACHE = "dhan_slugs.json"
OUT_HTML = "index.html"

NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
DHAN_PAGES = [
    "https://dhan.co/all-stocks-list/",
    "https://dhan.co/nifty-stocks-list/nifty-50/",
    "https://dhan.co/nifty-stocks-list/nifty-next-50/",
    "https://dhan.co/nifty-stocks-list/nifty-bank/",
    "https://dhan.co/nifty-stocks-list/nifty-fin-services/",
    "https://dhan.co/nifty-stocks-list/nifty-it/",
    "https://dhan.co/nifty-stocks-list/nifty-auto/",
    "https://dhan.co/nifty-stocks-list/nifty-fmcg/",
    "https://dhan.co/nifty-stocks-list/nifty-pharma/",
    "https://dhan.co/nifty-stocks-list/nifty-energy/",
    "https://dhan.co/nifty-stocks-list/nifty-metal/",
    "https://dhan.co/nifty-stocks-list/nifty-realty/",
    "https://dhan.co/stocks/market/large-cap-stocks/",
    "https://dhan.co/stocks/market/mid-cap-stocks/",
    "https://dhan.co/stocks/market/small-cap-stocks/",
    "https://dhan.co/sitemap/",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
NAMES = {1: "Blue", -1: "Red", 2: "Grey"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------- universe -----------------------------
def load_universe():
    for url in NSE_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            if {"Company Name", "Symbol"} <= set(df.columns) and len(df) > 300:
                df["Symbol"] = df["Symbol"].astype(str).str.strip()
                df.to_csv(UNIVERSE_CACHE, index=False)
                log(f"Universe: {len(df)} stocks from {url}")
                return df
        except Exception as e:
            log(f"Universe download failed ({url}): {e}")
    if os.path.exists(UNIVERSE_CACHE):
        df = pd.read_csv(UNIVERSE_CACHE)
        log(f"Universe: using cached {UNIVERSE_CACHE} ({len(df)} stocks)")
        return df
    raise SystemExit("Could not download Nifty 500 list and no cached copy exists.")


def tv_symbol(sym):
    return sym.replace("&", "_").replace("-", "_")


def tv_link(sym):
    return f"https://www.tradingview.com/chart/{LAYOUT_ID}/?symbol=" + quote(f"NSE:{tv_symbol(sym)}", safe="")


# ----------------------------- Dhan links (best effort) -----------------------------
DROP = {"ltd", "limited", "the", "and", "co", "company"}


def toks(s):
    s = s.lower().replace("&", " and ")
    return frozenset(t for t in re.findall(r"[a-z0-9]+", s) if t not in DROP)


def load_dhan_slugs():
    cached = []
    if os.path.exists(DHAN_CACHE):
        try:
            cached = json.load(open(DHAN_CACHE))
        except Exception:
            cached = []
    found = set(cached)
    rx = re.compile(r"/stocks/([a-z0-9][a-z0-9\-]*?)-share-price/?")
    for url in DHAN_PAGES:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code != 200:
                continue
            for m in rx.findall(r.text):
                found.add(m)
        except Exception as e:
            log(f"Dhan page failed ({url}): {e}")
    found = sorted(found)
    try:
        json.dump(found, open(DHAN_CACHE, "w"))
    except Exception:
        pass
    log(f"Dhan slugs known: {len(found)}")
    return found


def match_dhan(name, slug_toks):
    a = toks(name)
    if not a:
        return None
    best, best_j, tie = None, 0.0, False
    for slug, b in slug_toks.items():
        inter = len(a & b)
        if inter == 0:
            continue
        j = inter / len(a | b)
        ok = (a == b) or ((a <= b or b <= a) and j >= 0.6) or j >= 0.85
        if not ok:
            continue
        if j > best_j:
            best, best_j, tie = slug, j, False
        elif j == best_j:
            tie = True
    if best is None or tie:
        return None
    return f"https://dhan.co/stocks/{best}-share-price/"


# ----------------------------- indicator -----------------------------
def pine_ema(src, length):
    """EMA that matches Pine ta.ema (seeded with SMA of first `length` valid values)."""
    src = np.asarray(src, dtype=float)
    out = np.full(len(src), np.nan)
    alpha = 2.0 / (length + 1)
    prev = np.nan
    for i in range(len(src)):
        if np.isnan(prev):
            if i >= length - 1:
                w = src[i - length + 1:i + 1]
                if not np.isnan(w).any():
                    prev = float(w.mean())
                    out[i] = prev
        else:
            v = src[i]
            if np.isnan(v):
                continue
            prev = alpha * v + (1 - alpha) * prev
            out[i] = prev
    return out


def compute_states(df):
    c = df["Close"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))  # ta.tr (first bar na)
    ema = pine_ema(c, MA_LEN)
    rng = pine_ema(tr, MA_LEN)
    up = ema + rng * KELT_MULT
    lo = ema - rng * KELT_MULT
    valid = ~np.isnan(up)
    state = np.zeros(len(c), dtype=int)
    state[valid & (c > up)] = 1
    state[valid & (c < lo)] = -1
    state[valid & ~(c > up) & ~(c < lo)] = 2
    return ema, up, lo, state, valid


def find_flips(state, valid):
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return []
    sv = state[idx]
    ch = np.where(sv[1:] != sv[:-1])[0] + 1
    return [(int(idx[k]), int(sv[k - 1]), int(sv[k])) for k in ch]


def num(x, d=2):
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return round(x, d)


def analyse(df):
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < MA_LEN * 3:
        return None
    ema, up, lo, state, valid = compute_states(df)
    flips = find_flips(state, valid)
    c = df["Close"].to_numpy(float)
    last, prev = c[-1], c[-2]
    w = df.tail(252)

    def ret(n):
        return num((last / c[-1 - n] - 1) * 100, 1) if len(c) > n else None

    vol = df["Volume"].iloc[-1]
    res = {
        "last_date": df.index[-1],
        "ltp": num(last),
        "chg": num((last / prev - 1) * 100),
        "vol": int(vol) if np.isfinite(vol) else None,
        "hi": num(w["High"].max()),
        "lo": num(w["Low"].min()),
        "r1": ret(252), "r2": ret(504), "r3": ret(756),
        "d50": num(c[-50:].mean()) if len(c) >= 50 else None,
        "d200": num(c[-200:].mean()) if len(c) >= 200 else None,
        "cur": NAMES.get(int(state[-1])) if valid[-1] else None,
        "fd": None, "dir": None,
    }
    if flips:
        i, a, b = flips[-1]
        res["fd"] = df.index[i]
        res["dir"] = f"{NAMES[a]}>{NAMES[b]}"
    return res


# ----------------------------- prices -----------------------------
def download_batch(tickers):
    for attempt in range(3):
        try:
            df = yf.download(tickers, period=HISTORY, interval="1d", auto_adjust=False,
                             group_by="ticker", threads=True, progress=False)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log(f"Batch download error: {e}")
        time.sleep(3 * (attempt + 1))
    return None


def extract(df, t):
    if df is None:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if t in df.columns.get_level_values(0):
                d = df[t]
            elif t in df.columns.get_level_values(1):
                d = df.xs(t, axis=1, level=1)
            else:
                return None
        else:
            d = df
        d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        d = d.dropna(subset=["Close"])
        return d if len(d) else None
    except Exception:
        return None


# ----------------------------- fundamentals -----------------------------
def fetch_fund(sym):
    try:
        info = yf.Ticker(sym + ".NS").info
        if not isinstance(info, dict) or len(info) < 8:
            return None
        roe = num(info.get("returnOnEquity"), 4)
        return {
            "pe": num(info.get("trailingPE")),
            "roe": None if roe is None else round(roe * 100, 1),
            "eps": num(info.get("trailingEps")),
        }
    except Exception:
        return None


def refresh_fundamentals(symbols):
    cache = {}
    if os.path.exists(FUND_CACHE):
        try:
            cache = json.load(open(FUND_CACHE))
        except Exception:
            cache = {}
    now = time.time()
    maxage = FUND_MAX_AGE_DAYS * 86400
    todo = [s for s in symbols if now - cache.get(s, {}).get("ts", 0) > maxage]
    todo.sort(key=lambda s: cache.get(s, {}).get("ts", 0))
    log(f"Fundamentals: {len(todo)} stale/missing of {len(symbols)}")
    deadline = time.time() + FUND_TIME_BUDGET_S
    fails = ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i in range(0, len(todo), 16):
            if time.time() > deadline:
                log("Fundamentals: time budget reached, rest next run")
                break
            chunk = todo[i:i + 16]
            for s, r in zip(chunk, ex.map(fetch_fund, chunk)):
                if r is None:
                    fails += 1
                else:
                    cache[s] = {**r, "ts": now}
                    ok += 1
                    fails = 0
            if fails >= 40:
                log("Fundamentals: too many consecutive failures, stopping")
                break
    log(f"Fundamentals refreshed: {ok}")
    try:
        json.dump(cache, open(FUND_CACHE, "w"))
    except Exception:
        pass
    return cache


# ----------------------------- html -----------------------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Nifty 500 Trend Flips</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--tx:#14181f;--mut:#6b7280;--bd:#e1e5eb;--blue:#00c3ff;--red:#ff0062;--grey:#8a8f98;--up:#0f9d58;--dn:#d93025;--acc:#0b72c4}
@media(prefers-color-scheme:dark){:root{--bg:#0d1015;--card:#161a21;--tx:#e8eaed;--mut:#9aa3af;--bd:#272d37;--grey:#9aa0a6;--up:#34d399;--dn:#f87171;--acc:#4da3ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom)}
header{padding:10px 12px 4px}
h1{font-size:17px;margin:0 0 2px}
.sub{color:var(--mut);font-size:11.5px}
.ctl{padding:6px 12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
input[type=search],select{background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:8px;padding:7px 9px;font-size:13px}
input[type=search]{flex:1 1 150px;min-width:130px}
.chip{border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:5px 9px;font-size:12px;cursor:pointer;display:inline-flex;align-items:center;gap:4px}
.chip.on{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
label.tg{display:inline-flex;align-items:center;gap:5px;font-size:12px}
.wrap{margin:4px 8px 16px;border:1px solid var(--bd);border-radius:10px;background:var(--card);overflow:auto;max-height:calc(100vh - 190px);max-height:calc(100dvh - 190px);-webkit-overflow-scrolling:touch}
table{border-collapse:separate;border-spacing:0;width:100%;min-width:1500px}
th,td{padding:7px 9px;border-bottom:1px solid var(--bd);white-space:nowrap;text-align:right}
th{position:sticky;top:0;background:var(--card);z-index:2;cursor:pointer;font-weight:600;font-size:12px;user-select:none}
th.nos{cursor:default}
td:nth-child(1),th:nth-child(1){position:sticky;left:0;z-index:1;background:var(--card);text-align:left;max-width:170px;overflow:hidden;text-overflow:ellipsis;border-right:1px solid var(--bd)}
th:nth-child(1){z-index:3}
td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3),td:nth-child(15),th:nth-child(15),td:nth-child(16),th:nth-child(16){text-align:left}
tr:hover td{background:rgba(120,140,170,.08)}
a{color:var(--acc);text-decoration:none;font-weight:600}
.up{color:var(--up)}.dn{color:var(--dn)}.m{color:var(--mut)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle}
.dot.blue{background:var(--blue)}.dot.red{background:var(--red)}.dot.grey{background:var(--grey)}
.bd{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--bd);border-radius:999px;padding:2px 8px;font-weight:600}
.fs{display:block;color:var(--mut);font-size:11px;margin-top:2px}
.foot{padding:0 12px 18px;color:var(--mut);font-size:11.5px}
</style>
</head>
<body>
<header>
<h1>Nifty 500 Trend Flips</h1>
<div class="sub" id="meta"></div>
</header>
<div class="ctl">
<input type="search" id="q" placeholder="Search stock or symbol">
<select id="days"><option value="0">Any flip date</option><option value="1">Flipped 1d</option><option value="3">Last 3d</option><option value="7">Last 7d</option><option value="14">Last 14d</option><option value="30">Last 30d</option><option value="90">Last 90d</option></select>
<select id="cur"><option value="">Current: all</option><option>Blue</option><option>Grey</option><option>Red</option></select>
<label class="tg"><input type="checkbox" id="hg"> Hide Grey flips</label>
</div>
<div class="ctl" id="chips"></div>
<div class="sub" style="padding:0 12px" id="cnt"></div>
<div class="wrap"><table id="t"></table></div>
<div class="foot">Blue = close above channel, Red = close below, Grey = inside (EMA 30 baseline, 0.2 x ATR-style channel, daily). A flip is a change versus the previous daily close. Not investment advice.</div>
<script>
const D=__DATA__;
const ROWS=D.rows;
const DIRS=['Blue>Red','Blue>Grey','Red>Blue','Red>Grey','Grey>Red','Grey>Blue'];
const S={q:'',dirs:new Set(),days:0,cur:'',hg:false,sk:'fd',sd:-1};
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const nf=(v,d=2)=>v==null?'-':Number(v).toLocaleString('en-IN',{minimumFractionDigits:d,maximumFractionDigits:d});
const pc=v=>v==null?'<span class="m">-</span>':'<span class="'+(v>=0?'up':'dn')+'">'+(v>0?'+':'')+nf(v,1)+'%</span>';
const fdate=s=>{if(!s)return '-';const p=s.split('-');return +p[2]+' '+MON[+p[1]-1]+' '+p[0]};
const dot=c=>'<i class="dot '+c.toLowerCase()+'"></i>';
const dirBadge=d=>{if(!d)return '<span class="m">no flip</span>';const p=d.split('>');return '<span class="bd">'+dot(p[0])+'→'+dot(p[1])+' '+p[0]+' → '+p[1]+'</span>'};
const curBadge=c=>c?'<span class="bd">'+dot(c)+' '+c+'</span>':'<span class="m">-</span>';
const ago=n=>n==null?'':(n===0?'today':n+'d ago');
const COLS=[
{t:'Stock',h:r=>'<b>'+esc(r.n)+'</b>',s:r=>r.n.toLowerCase(),a:1},
{t:'Symbol',h:r=>esc(r.s),s:r=>r.s,a:1},
{t:'Chart',h:r=>'<a href="'+r.tv+'" target="_blank" rel="noopener">TV</a>'+(r.dh?' · <a href="'+r.dh+'" target="_blank" rel="noopener">Dhan</a>':''),nos:1},
{t:'LTP',h:r=>nf(r.ltp),s:r=>r.ltp},
{t:'Chg %',h:r=>pc(r.chg),s:r=>r.chg},
{t:'Volume',h:r=>r.vol==null?'-':Number(r.vol).toLocaleString('en-IN'),s:r=>r.vol},
{t:'52W High / Low',h:r=>nf(r.hi)+' / '+nf(r.lo),s:r=>r.hi},
{t:'1Y',h:r=>pc(r.r1),s:r=>r.r1},
{t:'2Y',h:r=>pc(r.r2),s:r=>r.r2},
{t:'3Y',h:r=>pc(r.r3),s:r=>r.r3},
{t:'PE',h:r=>nf(r.pe,1),s:r=>r.pe},
{t:'ROE %',h:r=>nf(r.roe,1),s:r=>r.roe},
{t:'EPS',h:r=>nf(r.eps),s:r=>r.eps},
{t:'50 / 200 DMA',h:r=>nf(r.d50)+' / '+nf(r.d200),s:r=>r.d200},
{t:'Direction',h:r=>dirBadge(r.dir)+(r.fd?'<span class="fs">'+fdate(r.fd)+' · '+ago(r.fdays)+'</span>':''),s:r=>r.fd},
{t:'Current',h:r=>curBadge(r.cur),s:r=>r.cur,a:1}
];
const SK=['n','s',null,'ltp','chg','vol','hi','r1','r2','r3','pe','roe','eps','d200','fd','cur'];
function filtered(){
 const q=S.q.trim().toLowerCase();
 let rows=ROWS.filter(r=>{
  if(q&&!(r.n.toLowerCase().includes(q)||r.s.toLowerCase().includes(q)))return false;
  if(S.dirs.size&&!(r.dir&&S.dirs.has(r.dir)))return false;
  if(S.hg&&r.dir&&r.dir.includes('Grey'))return false;
  if(S.days&&!(r.fdays!=null&&r.fdays<=S.days))return false;
  if(S.cur&&r.cur!==S.cur)return false;
  return true;});
 const ci=SK.indexOf(S.sk),c=COLS[ci];
 rows.sort((a,b)=>{const x=c.s(a),y=c.s(b);
  if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;
  return (x<y?-1:x>y?1:0)*S.sd;});
 return rows;}
function render(){
 const rows=filtered();
 let h='<thead><tr>';
 COLS.forEach((c,i)=>{const on=SK[i]===S.sk;
  h+='<th class="'+(c.nos?'nos':'')+'" data-i="'+i+'">'+c.t+(on?(S.sd>0?' ▲':' ▼'):'')+'</th>';});
 h+='</tr></thead><tbody>';
 rows.forEach(r=>{h+='<tr>'+COLS.map(c=>'<td>'+c.h(r)+'</td>').join('')+'</tr>';});
 h+='</tbody>';
 $('#t').innerHTML=h;
 $('#cnt').textContent=rows.length+' of '+ROWS.length+' stocks';
 document.querySelectorAll('th').forEach(th=>th.onclick=()=>{
  const i=+th.dataset.i;if(COLS[i].nos)return;const k=SK[i];
  if(S.sk===k)S.sd=-S.sd;else{S.sk=k;S.sd=COLS[i].a?1:-1;}
  render();});
}
function chips(){
 $('#chips').innerHTML=DIRS.map(d=>{const p=d.split('>');
  return '<span class="chip'+(S.dirs.has(d)?' on':'')+'" data-d="'+d+'">'+dot(p[0])+'→'+dot(p[1])+' '+p[0]+' → '+p[1]+'</span>';}).join('');
 document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  const d=c.dataset.d;S.dirs.has(d)?S.dirs.delete(d):S.dirs.add(d);chips();render();});
}
$('#meta').textContent='Last updated: '+D.updated+' · data through '+fdate(D.asof)+' · '+ROWS.length+' stocks'+(D.skipped?' · '+D.skipped+' skipped':'');
$('#q').oninput=e=>{S.q=e.target.value;render()};
$('#days').onchange=e=>{S.days=+e.target.value;render()};
$('#cur').onchange=e=>{S.cur=e.target.value;render()};
$('#hg').onchange=e=>{S.hg=e.target.checked;render()};
chips();render();
</script>
</body>
</html>
"""


def build_html(rows, meta):
    payload = json.dumps({"rows": rows, **meta}, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    return HTML.replace("__DATA__", payload)


# ----------------------------- main -----------------------------
def run_check(symbol):
    symbol = symbol.upper().replace(".NS", "")
    df = download_batch([symbol + ".NS"])
    d = extract(df, symbol + ".NS")
    if d is None:
        raise SystemExit(f"No data for {symbol}")
    d = d.dropna(subset=["Close", "High", "Low"]).copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    ema, up, lo, state, valid = compute_states(d)
    print(f"\n{symbol}: last 15 daily bars (split-adjusted close)")
    print(f"{'date':<12}{'close':>10}{'baseline':>11}{'upper':>10}{'lower':>10}  colour")
    for i in range(max(0, len(d) - 15), len(d)):
        col = NAMES.get(int(state[i]), "-") if valid[i] else "-"
        print(f"{d.index[i].strftime('%Y-%m-%d'):<12}{d['Close'].iloc[i]:>10.2f}{ema[i]:>11.2f}{up[i]:>10.2f}{lo[i]:>10.2f}  {col}")
    flips = find_flips(state, valid)
    print("\nLast 6 flips (date = the bar where the NEW colour first appears):")
    for i, a, b in flips[-6:][::-1]:
        print(f"  {d.index[i].strftime('%Y-%m-%d')}  {NAMES[a]} -> {NAMES[b]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="debug one symbol, e.g. AXISBANK")
    ap.add_argument("--limit", type=int, help="only scan first N stocks (testing)")
    args = ap.parse_args()
    if args.check:
        return run_check(args.check)

    uni = load_universe()
    if args.limit:
        uni = uni.head(args.limit)
    syms = uni["Symbol"].tolist()
    names = dict(zip(uni["Symbol"], uni["Company Name"]))

    # prices
    results, missing = {}, []
    for i in range(0, len(syms), BATCH_SIZE):
        batch = syms[i:i + BATCH_SIZE]
        log(f"Prices {i + 1}-{i + len(batch)} / {len(syms)}")
        df = download_batch([s + ".NS" for s in batch])
        for s in batch:
            d = extract(df, s + ".NS")
            if d is None:
                missing.append(s)
                continue
            try:
                r = analyse(d)
            except Exception as e:
                log(f"Analyse failed {s}: {e}")
                r = None
            if r is None:
                missing.append(s)
            else:
                results[s] = r
        time.sleep(1)

    # one retry for the ones that failed
    retry = list(missing)
    missing = []
    for s in retry:
        d = extract(download_batch([s + ".NS"]), s + ".NS")
        r = None
        try:
            r = analyse(d) if d is not None else None
        except Exception:
            r = None
        if r is None:
            missing.append(s)
        else:
            results[s] = r
    if not results:
        raise SystemExit("No price data downloaded; keeping previous index.html")

    asof = max(r["last_date"] for r in results.values())
    stale = [s for s, r in results.items() if (asof - r["last_date"]).days > STALE_DAYS]
    for s in stale:
        results.pop(s)
    skipped = sorted(set(missing) | set(stale))
    if skipped:
        log(f"Skipped {len(skipped)}: {', '.join(skipped)}")

    # extras
    fund = refresh_fundamentals(list(results.keys()))
    slug_toks = {s: toks(s) for s in load_dhan_slugs()}

    rows = []
    for s, r in results.items():
        nm = re.sub(r"\s+(Limited|Ltd\.?)$", "", str(names.get(s, s)), flags=re.I)
        f = fund.get(s, {})
        fd = r["fd"]
        rows.append({
            "n": nm, "s": s, "tv": tv_link(s), "dh": match_dhan(str(names.get(s, s)), slug_toks),
            "ltp": r["ltp"], "chg": r["chg"], "vol": r["vol"], "hi": r["hi"], "lo": r["lo"],
            "r1": r["r1"], "r2": r["r2"], "r3": r["r3"],
            "pe": f.get("pe"), "roe": f.get("roe"), "eps": f.get("eps"),
            "d50": r["d50"], "d200": r["d200"],
            "dir": r["dir"], "cur": r["cur"],
            "fd": fd.strftime("%Y-%m-%d") if fd is not None else None,
            "fdays": (asof - fd).days if fd is not None else None,
        })
    rows.sort(key=lambda x: (x["fd"] is None, "" if x["fd"] is None else x["fd"]), reverse=False)
    rows.sort(key=lambda x: x["fd"] or "", reverse=True)

    meta = {
        "updated": dt.datetime.now(IST).strftime("%d %b %Y, %H:%M IST"),
        "asof": asof.strftime("%Y-%m-%d"),
        "skipped": len(skipped),
    }
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(build_html(rows, meta))
    log(f"Wrote {OUT_HTML}: {len(rows)} stocks, data through {meta['asof']}")


if __name__ == "__main__":
    main()
