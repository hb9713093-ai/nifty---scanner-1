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

Outputs:
  index.html          the website (table + click-TV popup chart)
  data/<SYMBOL>.json  last ~6 months of candles + channel for each stock (popup chart)

Usage:
  python scanner.py                   full run
  python scanner.py --check AXISBANK  prints recent colours + last flips for one stock
  python scanner.py --limit 40        quick test on first 40 stocks
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
HISTORY_DAYS = 1830             # ~5 years of daily candles
FLIP_WINDOW_DAYS = 7            # only show stocks whose latest flip is within this many days
BATCH_SIZE = 50
CHART_BARS = 130                # ~6 months of daily candles in the popup chart
LAYOUT_ID = "AXNXjCz0"          # your saved TradingView layout (indicator pre-loaded)
FUND_MAX_AGE_DAYS = 7           # refresh PE/ROE/EPS weekly
FUND_TIME_BUDGET_S = 600        # max seconds per run spent on fundamentals
STALE_DAYS = 10                 # skip stocks whose last candle is older than this

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UNIVERSE_CACHE = "nifty500.csv"
FUND_CACHE = "fundamentals.json"
DATA_DIR = "data"
OUT_HTML = "index.html"

NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
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


def chart_payload(df, ema, up, lo):
    n = min(CHART_BARS, len(df))
    sl = slice(len(df) - n, len(df))

    def arr(a):
        return [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(a, float)[sl]]

    return {
        "d": [d.strftime("%Y-%m-%d") for d in df.index[sl]],
        "o": arr(df["Open"].to_numpy(float)),
        "h": arr(df["High"].to_numpy(float)),
        "l": arr(df["Low"].to_numpy(float)),
        "c": arr(df["Close"].to_numpy(float)),
        "e": arr(ema),
        "u": arr(up),
        "b": arr(lo),
    }


def analyse(df):
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    now = dt.datetime.now(IST)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (15, 45):
        df = df.iloc[:-1]            # today's candle is still forming: use closed daily candles only
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
        "chart": chart_payload(df, ema, up, lo),
    }
    if flips:
        i, a, b = flips[-1]
        res["fd"] = df.index[i]
        res["dir"] = f"{NAMES[a]}>{NAMES[b]}"
    return res


# ----------------------------- prices -----------------------------
def date_range():
    today = dt.datetime.now(IST).date()
    start = today - dt.timedelta(days=HISTORY_DAYS)
    end = today + dt.timedelta(days=1)          # end date is exclusive
    return start.isoformat(), end.isoformat()


def download_batch(tickers):
    start, end = date_range()
    for attempt in range(3):
        try:
            df = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False,
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
table{border-collapse:separate;border-spacing:0;width:100%;min-width:1400px}
th,td{padding:7px 9px;border-bottom:1px solid var(--bd);white-space:nowrap;text-align:right}
th{position:sticky;top:0;background:var(--card);z-index:2;cursor:pointer;font-weight:600;font-size:12px;user-select:none}
th.nos{cursor:default}
td:nth-child(1),th:nth-child(1){position:sticky;left:0;z-index:1;background:var(--card);text-align:left;max-width:170px;overflow:hidden;text-overflow:ellipsis;border-right:1px solid var(--bd)}
th:nth-child(1){z-index:3}
td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3),td:nth-child(15),th:nth-child(15),td:nth-child(16),th:nth-child(16){text-align:left}
tr:hover td{background:rgba(120,140,170,.08)}
a{color:var(--acc);text-decoration:none;font-weight:600}
.tvb{border:1px solid var(--acc);border-radius:6px;padding:2px 8px}
.up{color:var(--up)}.dn{color:var(--dn)}.m{color:var(--mut)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle}
.dot.blue{background:var(--blue)}.dot.red{background:var(--red)}.dot.grey{background:var(--grey)}
.bd{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--bd);border-radius:999px;padding:2px 8px;font-weight:600}
.fs{display:block;color:var(--mut);font-size:11px;margin-top:2px}
.warn{margin:6px 12px;padding:8px 10px;border:1px solid #d9a400;background:rgba(255,193,7,.14);border-radius:8px;font-size:12px}
.foot{padding:0 12px 18px;color:var(--mut);font-size:11.5px}
.md{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:50;padding:10px}
.md[hidden]{display:none}
.mc{background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:12px;width:min(1060px,100%);max-height:96vh;overflow:auto;padding:12px 14px}
.mh{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap;margin-bottom:8px}
.mh b{font-size:15px}
.mbtn{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.mbtn button{background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:8px;padding:5px 10px;font-size:12px;cursor:pointer}
.mbtn button.on{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.cvw{width:100%}
canvas{display:block;width:100%;touch-action:pan-y}
.lg{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;color:var(--mut);font-size:11.5px}
</style>
</head>
<body>
<header>
<h1>Nifty 500 Trend Flips</h1>
<div class="sub" id="meta"></div>
</header>
<div class="warn" id="warn" hidden></div>
<div class="ctl">
<input type="search" id="q" placeholder="Search stock or symbol">
<select id="days"><option value="0">Any (whole window)</option><option value="3">Last 3 days</option><option value="1">Last 1 day</option></select>
<select id="cur"><option value="">Current: all</option><option>Blue</option><option>Grey</option><option>Red</option></select>
<label class="tg"><input type="checkbox" id="hg"> Hide Grey flips</label>
</div>
<div class="ctl" id="chips"></div>
<div class="sub" style="padding:0 12px" id="cnt"></div>
<div class="wrap"><table id="t"></table></div>
<div class="foot">Only stocks whose latest colour flip happened in the last 7 days are listed. Click TV on any row to see its chart with your strategy colours. Blue = close above channel, Red = close below, Grey = inside (EMA 30 baseline, 0.2 x EMA of true range channel, daily). A flip is a colour change versus the previous daily close. Not investment advice.</div>

<div class="md" id="md" hidden>
<div class="mc">
<div class="mh">
<div><b id="mt"></b> <span id="mb"></span><div class="sub" id="ms"></div></div>
<div class="mbtn"><button data-r="63">3M</button><button data-r="130" class="on">6M</button><a id="mtv" target="_blank" rel="noopener">Open in TradingView ↗</a><button id="mx">✕</button></div>
</div>
<div class="cvw"><canvas id="cv"></canvas></div>
<div class="sub" id="mn"></div>
<div class="lg"><span><i class="dot blue"></i> Blue: close above channel</span><span><i class="dot grey"></i> Grey: inside channel</span><span><i class="dot red"></i> Red: close below channel</span><span>▲ = colour flip</span><span>Thick line = baseline, dashed = channel</span></div>
</div>
</div>

<script>
const D=__DATA__;
const ROWS=D.rows;
const BYS={};ROWS.forEach(r=>{BYS[r.s]=r});
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
{t:'Chart',h:r=>'<a class="tvb" data-s="'+esc(r.s)+'" href="'+r.tv+'" target="_blank" rel="noopener">TV</a>',nos:1},
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

/* ---------- popup chart ---------- */
const CC={'1':'#00c3ff','-1':'#ff0062','2':'#8a8f98','0':'#8a8f98'};
const CN={'1':'Blue','-1':'Red','2':'Grey','0':'-'};
let CH=null,TOKEN=0;
const stOf=(c,u,b,i)=>(u[i]==null||b[i]==null||c[i]==null)?0:(c[i]>u[i]?1:(c[i]<b[i]?-1:2));
async function openChart(r){
 const tok=++TOKEN;
 $('#md').hidden=false;document.body.style.overflow='hidden';
 $('#mt').textContent=r.n+' ('+r.s+')';
 $('#mb').innerHTML=curBadge(r.cur);
 $('#ms').textContent=r.dir?('Last flip: '+r.dir.replace('>',' → ')+' · '+fdate(r.fd)+' · '+ago(r.fdays)):'No colour flip in the available history';
 $('#mtv').href=r.tv;
 $('#mn').textContent='Loading chart...';
 CH=null;
 const cv=$('#cv');cv.getContext('2d').clearRect(0,0,cv.width,cv.height);
 try{
  const res=await fetch('data/'+r.f+'.json?v='+encodeURIComponent(D.asof+D.updated));
  if(!res.ok)throw new Error('HTTP '+res.status);
  const j=await res.json();
  if(tok!==TOKEN)return;
  CH=j;CH.range=Math.min(CUR_RANGE,j.d.length);CH.hover=-1;
  $('#mn').textContent='';
  drawChart();
 }catch(e){
  if(tok===TOKEN)$('#mn').textContent='Chart data not available. Open the website link (not a downloaded file) and wait for the next daily update if this is new.';
 }
}
let CUR_RANGE=130;
function closeChart(){TOKEN++;$('#md').hidden=true;document.body.style.overflow='';CH=null;}
function drawChart(){
 if(!CH)return;
 const cv=$('#cv'),box=cv.parentElement,dpr=window.devicePixelRatio||1;
 const W=Math.max(300,box.clientWidth),H=W<600?320:440;
 const pw0=Math.round(W*dpr),ph0=Math.round(H*dpr);
 if(cv.width!==pw0||cv.height!==ph0){cv.width=pw0;cv.height=ph0;}
 cv.style.height=H+'px';
 const x=cv.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);
 const cs=getComputedStyle(document.documentElement);
 const tx=cs.getPropertyValue('--tx').trim()||'#222',mut=cs.getPropertyValue('--mut').trim()||'#888',bd=cs.getPropertyValue('--bd').trim()||'#ddd',card=cs.getPropertyValue('--card').trim()||'#fff';
 const N=CH.d.length,n=CH.range,s0=N-n;
 const L=8,R=60,T=10,B=24,pw=W-L-R,ph=H-T-B;
 let mn=Infinity,mx=-Infinity;
 for(let i=s0;i<N;i++){
  if(CH.l[i]!=null)mn=Math.min(mn,CH.l[i]);
  if(CH.h[i]!=null)mx=Math.max(mx,CH.h[i]);
  if(CH.u[i]!=null){mx=Math.max(mx,CH.u[i]);mn=Math.min(mn,CH.b[i]);}
 }
 if(!isFinite(mn)||!isFinite(mx))return;
 const pad=(mx-mn)*0.05||1;mn-=pad;mx+=pad;
 const X=i=>L+(i-s0+0.5)*pw/n,Y=v=>T+(mx-v)/(mx-mn)*ph;
 CH.lay={L,pw,n,s0};
 x.clearRect(0,0,W,H);
 x.font='11px system-ui,-apple-system,Segoe UI,sans-serif';
 /* grid + price axis */
 x.textAlign='left';x.textBaseline='middle';x.lineWidth=1;
 for(let k=0;k<=5;k++){
  const v=mn+(mx-mn)*k/5,y=Y(v);
  x.strokeStyle=bd;x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();
  x.fillStyle=mut;x.fillText(nf(v),W-R+6,y);
 }
 /* date axis */
 x.textAlign='center';x.textBaseline='top';x.fillStyle=mut;
 const ticks=W<600?4:7;
 for(let k=0;k<ticks;k++){
  const i=s0+Math.round((n-1)*k/(ticks-1));
  const p=CH.d[i].split('-');
  x.fillText(+p[2]+' '+MON[+p[1]-1]+(k===0?' '+p[0].slice(2):''),X(i),H-B+6);
 }
 /* channel band */
 const idx=[];for(let i=s0;i<N;i++)if(CH.u[i]!=null&&CH.b[i]!=null)idx.push(i);
 if(idx.length>1){
  x.beginPath();
  idx.forEach((i,k)=>{k?x.lineTo(X(i),Y(CH.u[i])):x.moveTo(X(i),Y(CH.u[i]))});
  for(let k=idx.length-1;k>=0;k--)x.lineTo(X(idx[k]),Y(CH.b[idx[k]]));
  x.closePath();x.fillStyle='rgba(130,150,180,0.13)';x.fill();
  x.strokeStyle='rgba(130,150,180,0.6)';x.lineWidth=1;x.setLineDash([4,3]);
  [CH.u,CH.b].forEach(a=>{x.beginPath();idx.forEach((i,k)=>{k?x.lineTo(X(i),Y(a[i])):x.moveTo(X(i),Y(a[i]))});x.stroke();});
  x.setLineDash([]);
 }
 /* candles coloured by strategy state */
 const bw=Math.max(1,Math.min(14,pw/n*0.66));
 for(let i=s0;i<N;i++){
  if(CH.c[i]==null)continue;
  const col=CC[stOf(CH.c,CH.u,CH.b,i)];
  const o=CH.o[i]==null?CH.c[i]:CH.o[i],c=CH.c[i],h=CH.h[i]==null?Math.max(o,c):CH.h[i],l=CH.l[i]==null?Math.min(o,c):CH.l[i];
  x.strokeStyle=col;x.fillStyle=col;x.lineWidth=1;
  x.beginPath();x.moveTo(X(i),Y(h));x.lineTo(X(i),Y(l));x.stroke();
  const top=Y(Math.max(o,c)),bot=Y(Math.min(o,c));
  x.fillRect(X(i)-bw/2,top,bw,Math.max(1,bot-top));
 }
 /* baseline coloured like the indicator line */
 x.lineWidth=2.6;x.lineCap='round';
 for(let i=s0+1;i<N;i++){
  if(CH.e[i]==null||CH.e[i-1]==null)continue;
  x.strokeStyle=CC[stOf(CH.c,CH.u,CH.b,i)];
  x.beginPath();x.moveTo(X(i-1),Y(CH.e[i-1]));x.lineTo(X(i),Y(CH.e[i]));x.stroke();
 }
 /* flip markers */
 for(let i=s0;i<N;i++){
  const a=stOf(CH.c,CH.u,CH.b,i-1),b=stOf(CH.c,CH.u,CH.b,i);
  if(i<1||!a||!b||a===b||CH.l[i]==null)continue;
  const px=X(i),py=Y(CH.l[i])+12;
  x.beginPath();x.moveTo(px,py-6);x.lineTo(px-5,py+3);x.lineTo(px+5,py+3);x.closePath();
  x.fillStyle=CC[b];x.fill();x.strokeStyle=tx;x.lineWidth=1;x.stroke();
 }
 /* hover crosshair + tooltip */
 const hv=CH.hover;
 if(hv>=s0&&hv<N&&CH.c[hv]!=null){
  x.strokeStyle=mut;x.lineWidth=1;x.setLineDash([3,3]);
  x.beginPath();x.moveTo(X(hv),T);x.lineTo(X(hv),T+ph);x.stroke();x.setLineDash([]);
  const o=CH.o[hv]==null?CH.c[hv]:CH.o[hv];
  const sc=stOf(CH.c,CH.u,CH.b,hv);
  const txt=fdate(CH.d[hv])+'   O '+nf(o)+'  H '+nf(CH.h[hv])+'  L '+nf(CH.l[hv])+'  C '+nf(CH.c[hv])+'   '+CN[sc];
  x.font='12px system-ui,-apple-system,Segoe UI,sans-serif';
  x.textAlign='left';x.textBaseline='middle';
  const tw=x.measureText(txt).width+30;
  x.fillStyle=card;x.globalAlpha=0.92;x.fillRect(L+4,T+4,tw,22);x.globalAlpha=1;
  x.strokeStyle=bd;x.strokeRect(L+4,T+4,tw,22);
  x.fillStyle=CC[sc];x.beginPath();x.arc(L+14,T+15,5,0,Math.PI*2);x.fill();
  x.fillStyle=tx;x.fillText(txt,L+24,T+15);
 }
}
function hoverAt(clientX){
 if(!CH||!CH.lay)return;
 const rect=$('#cv').getBoundingClientRect();
 const {L,pw,n,s0}=CH.lay;
 let i=Math.floor((clientX-rect.left-L)/(pw/n))+s0;
 i=Math.max(s0,Math.min(s0+n-1,i));
 if(i!==CH.hover){CH.hover=i;drawChart();}
}

$('#meta').textContent='Last updated: '+D.updated+' · data through '+fdate(D.asof)+' · '+ROWS.length+' stocks flipped in the last '+D.window+' days (of '+D.scanned+' scanned)'+(D.skipped?' · '+D.skipped+' skipped':'');
$('#days').options[0].text='Last '+D.window+' days';
(function(){
 const n=new Date(Date.now()+5.5*3600*1000);
 const today=Date.UTC(n.getUTCFullYear(),n.getUTCMonth(),n.getUTCDate());
 const mins=n.getUTCHours()*60+n.getUTCMinutes();
 const a=D.asof.split('-');const as=Date.UTC(+a[0],+a[1]-1,+a[2]);
 let cnt=0;
 for(let t=as+864e5;t<=today;t+=864e5){
  const wd=new Date(t).getUTCDay();
  if(wd===0||wd===6)continue;
  if(t===today&&mins<17*60)continue;
  cnt++;
 }
 if(cnt>=1){
  const w=$('#warn');w.hidden=false;
  w.textContent='Latest candle in this data is '+fdate(D.asof)+'. About '+cnt+' newer trading day'+(cnt>1?'s':'')+' may be missing. The site refreshes after 4:45 PM IST on weekdays; if this stays after the next refresh, the data source has not published yet, or a market holiday is in between.';
 }
})();
$('#q').oninput=e=>{S.q=e.target.value;render()};
$('#days').onchange=e=>{S.days=+e.target.value;render()};
$('#cur').onchange=e=>{S.cur=e.target.value;render()};
$('#hg').onchange=e=>{S.hg=e.target.checked;render()};
$('#t').addEventListener('click',e=>{
 const a=e.target.closest('.tvb');if(!a)return;
 if(e.ctrlKey||e.metaKey||e.shiftKey||e.button===1)return;
 e.preventDefault();const r=BYS[a.dataset.s];if(r)openChart(r);
});
$('#mx').onclick=closeChart;
$('#md').addEventListener('click',e=>{if(e.target.id==='md')closeChart();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#md').hidden)closeChart();});
document.querySelectorAll('.mbtn button[data-r]').forEach(b=>b.onclick=()=>{
 CUR_RANGE=+b.dataset.r;
 document.querySelectorAll('.mbtn button[data-r]').forEach(x=>x.classList.toggle('on',x===b));
 if(CH){CH.range=Math.min(CUR_RANGE,CH.d.length);drawChart();}
});
const cvEl=$('#cv');
cvEl.addEventListener('mousemove',e=>hoverAt(e.clientX));
cvEl.addEventListener('mouseleave',()=>{if(CH){CH.hover=-1;drawChart();}});
cvEl.addEventListener('touchstart',e=>{if(e.touches[0])hoverAt(e.touches[0].clientX);},{passive:true});
cvEl.addEventListener('touchmove',e=>{if(e.touches[0])hoverAt(e.touches[0].clientX);},{passive:true});
window.addEventListener('resize',()=>{if(CH)drawChart();});
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


def write_chart_files(results, cleanup):
    os.makedirs(DATA_DIR, exist_ok=True)
    keep = set()
    for s, r in results.items():
        fn = tv_symbol(s) + ".json"
        keep.add(fn)
        with open(os.path.join(DATA_DIR, fn), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(r["chart"], separators=(",", ":")))
    if cleanup:
        for fn in os.listdir(DATA_DIR):
            if fn.endswith(".json") and fn not in keep:
                os.remove(os.path.join(DATA_DIR, fn))


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

    scanned = len(results)
    log(f"Latest candle in data: {asof.strftime('%Y-%m-%d')}")
    # keep only stocks whose latest flip is inside the window
    results = {s: r for s, r in results.items()
               if r["fd"] is not None and (asof - r["fd"]).days <= FLIP_WINDOW_DAYS}
    log(f"{len(results)} of {scanned} stocks flipped in the last {FLIP_WINDOW_DAYS} days")

    fund = refresh_fundamentals(list(results.keys()))

    rows = []
    for s, r in results.items():
        nm = re.sub(r"\s+(Limited|Ltd\.?)$", "", str(names.get(s, s)), flags=re.I)
        f = fund.get(s, {})
        fd = r["fd"]
        rows.append({
            "n": nm, "s": s, "f": tv_symbol(s), "tv": tv_link(s),
            "ltp": r["ltp"], "chg": r["chg"], "vol": r["vol"], "hi": r["hi"], "lo": r["lo"],
            "r1": r["r1"], "r2": r["r2"], "r3": r["r3"],
            "pe": f.get("pe"), "roe": f.get("roe"), "eps": f.get("eps"),
            "d50": r["d50"], "d200": r["d200"],
            "dir": r["dir"], "cur": r["cur"],
            "fd": fd.strftime("%Y-%m-%d") if fd is not None else None,
            "fdays": (asof - fd).days if fd is not None else None,
        })
    rows.sort(key=lambda x: x["fd"] or "", reverse=True)

    write_chart_files(results, cleanup=not args.limit)

    meta = {
        "updated": dt.datetime.now(IST).strftime("%d %b %Y, %H:%M IST"),
        "asof": asof.strftime("%Y-%m-%d"),
        "skipped": len(skipped),
        "scanned": scanned,
        "window": FLIP_WINDOW_DAYS,
    }
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(build_html(rows, meta))
    log(f"Wrote {OUT_HTML} and {len(results)} chart files, data through {meta['asof']}")


if __name__ == "__main__":
    main()
