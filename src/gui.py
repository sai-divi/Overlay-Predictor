import tkinter as tk
from tkinter import scrolledtext
import threading, sys, io, json, urllib.request, urllib.parse, ctypes
from ctypes import wintypes
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

from src.server.api import ApiServer

from src.config import Config
from src.data.fetcher import fetch_realtime, lookup_ticker_info, fetch_latest_price
from src.data.news import fetch_news, aggregate_sentiment, sentiment_label
from src.features.signals import create_target_labels
from src.features.indicators import add_all_indicators
from src.models.xgboost_model import XGBoostModel

GREEN = "#00c853"; RED = "#ff1744"; YELLOW = "#ffd600"
THEMES = {
    "dark":  {"BG":"#0d0d0d","FG":"#ffffff","AX":"#111111","BTN":"#1a1a1a","SEP":"#222222","STATUS":"#888888"},
    "light":{"BG":"#f0f0f0","FG":"#0d0d0d","AX":"#ffffff","BTN":"#e8e8e8","SEP":"#cccccc","STATUS":"#888888"},
}
_theme="dark"; BG=THEMES["dark"]["BG"]; FG=THEMES["dark"]["FG"]
AX_BG=THEMES["dark"]["AX"]; BTN_BG=THEMES["dark"]["BTN"]
SEP_COL=THEMES["dark"]["SEP"]; STATUS_COL=THEMES["dark"]["STATUS"]
def set_theme(n):
    global _theme,BG,FG,AX_BG,BTN_BG,SEP_COL,STATUS_COL
    t=THEMES[n]; BG,FG,AX_BG,BTN_BG,SEP_COL,STATUS_COL=t["BG"],t["FG"],t["AX"],t["BTN"],t["SEP"],t["STATUS"]; _theme=n

FONT=("Arial",12); FONT_SM=("Arial",10)

_REPUTABLE_SOURCES = {"bloomberg","reuters","cnbc","wsj","wall street journal",
    "financial times","ft","barron's","barrons","marketwatch",
    "investor's business daily","ibd","seeking alpha","yahoo finance",
    "nikkei","economist","the economist","forbes","fortune",
    "business insider","fox business","the street"}


class StdoutRedirector(io.StringIO):
    def __init__(self,w): super().__init__(); self.w=w
    def write(self,m): super().write(m); self.w.insert(tk.END,m); self.w.see(tk.END); self.w.update_idletasks()
    def flush(self): pass


class LiveChart:
    def __init__(self,parent,cfg):
        self.parent=parent; self.cfg=cfg
        self.ticker=tk.StringVar(value="")
        self.df=pd.DataFrame()
        self.news_articles=[]; self.news_sentiment=None
        self._model=None; self._feature_cols=None; self._trained=False
        self.price_history=deque(maxlen=120)
        self.timer_id=None
        self._pred_dir=0; self._pred_conf=0.0; self._latest_price=0.0
        self._reasons=[]
        self._mom1=0; self._mom5=0; self._vol=0; self._pl=0; self._ph=0; self._rsv=50; self._navg=0
        self._ml_dir=0; self._ml_conf=0; self._ind_values={}
        self.stock_info=None; self._stock_labels={}

        self.build_ui()
        root=self.parent.winfo_toplevel()
        root.attributes("-topmost",True); root.attributes("-alpha",0.85)
        self._start_predict_loop()

    # ================================================================
    # UI
    # ================================================================
    def build_ui(self):
        top=tk.Frame(self.parent,bg=BG); top.pack(fill=tk.X,pady=(0,2))
        tk.Label(top,text="Ticker:",font=FONT,bg=BG,fg=FG).pack(side=tk.LEFT,padx=(0,2))
        self.ticker_entry=tk.Entry(top,textvariable=self.ticker,width=10,font=FONT,bg=BG,fg=FG,
            insertbackground=FG,relief=tk.FLAT,highlightthickness=1,highlightbackground=FG)
        self.ticker_entry.pack(side=tk.LEFT,padx=(0,4))
        self.ticker_entry.bind("<KeyRelease>",self._on_ticker_key)
        self.ticker_entry.bind("<Escape>",lambda e:self._hide_ac())
        self.ticker_entry.bind("<Return>",lambda e:self._ac_pick_first())
        self.name_var=tk.StringVar(value="")
        self.name_label=tk.Label(top,textvariable=self.name_var,font=FONT,bg=BG,fg="#888888",anchor=tk.W)
        self.name_label.pack(side=tk.LEFT,padx=(0,6))
        cf=tk.Frame(top,bg=BG); cf.pack(side=tk.RIGHT)
        def detach():
            app=self.parent.winfo_toplevel()._app
            if app: app._detach(); self.status_var.set("Detached")
        btns=[("Refresh",self.fetch_data,FG),("Hook+",self.hook_app,"#555555"),
              ("Detach",detach,"#555555"),("Invert",self.toggle_invert,"#555555")]
        for t,c,h in btns:
            tk.Button(cf,text=t,command=c,font=FONT,bg="#1a1a1a",fg=FG,relief=tk.FLAT,bd=1,
                      highlightthickness=1,highlightbackground=h,padx=6).pack(side=tk.LEFT,padx=1)

        # Main: news panel (left), predict panel (right)
        mx=tk.Frame(self.parent,bg=BG); mx.pack(fill=tk.BOTH,expand=True,pady=(2,0))

        nf=tk.Frame(mx,bg=BG,width=260); nf.pack(side=tk.LEFT,fill=tk.Y); nf.pack_propagate(False)
        self.side_news_text=scrolledtext.ScrolledText(nf,font=FONT_SM,bg=AX_BG,fg=FG,insertbackground=FG,relief=tk.FLAT,highlightthickness=0)
        self.side_news_text.pack(fill=tk.BOTH,expand=True)

        pf=tk.Frame(mx,bg=BG); pf.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        self.si_frame=tk.Frame(pf,bg=AX_BG,highlightthickness=1,highlightbackground=SEP_COL)
        self.si_frame.pack(fill=tk.X,pady=(0,2))
        self._build_stock_info(self.si_frame)

        self.side_predict_text=scrolledtext.ScrolledText(pf,font=FONT_SM,bg=AX_BG,fg=FG,insertbackground=FG,relief=tk.FLAT,highlightthickness=0)
        self.side_predict_text.pack(fill=tk.BOTH,expand=True)

        # Status
        self.status_var=tk.StringVar(value="Enter ticker + Refresh to start. Install userscript for live prices.")
        self.status_bar=tk.Label(self.parent,textvariable=self.status_var,font=FONT,bg=BG,fg=STATUS_COL,anchor=tk.W)
        self.status_bar.pack(fill=tk.X)

    # ================================================================
    # Theme
    # ================================================================
    def toggle_invert(self):
        set_theme("light" if _theme=="dark" else "dark")
        self._apply_theme(self.parent.winfo_toplevel())
    def _apply_theme(self,w):
        try:
            if isinstance(w,tk.Frame): w.config(bg=BG)
            elif isinstance(w,tk.Label):
                if w.cget("bg") in ("#0d0d0d","#f0f0f0","#111111","#ffffff"): w.config(bg=BG,fg=FG)
                elif w.cget("bg") in ("#1a1a1a","#e8e8e8"): w.config(bg=BTN_BG)
            elif isinstance(w,tk.Button):
                if w.cget("bg") in ("#1a1a1a","#e8e8e8"): w.config(bg=BTN_BG)
                if w.cget("fg") in ("#ffffff","#0d0d0d"): w.config(fg=FG)
            elif isinstance(w,tk.Text): w.config(bg=BG,fg=FG,insertbackground=FG)
            elif isinstance(w,tk.Entry): w.config(bg=BG,fg=FG,insertbackground=FG)
        except: pass
        for c in w.winfo_children(): self._apply_theme(c)

    # ================================================================
    # Autocomplete
    # ================================================================
    _ac_listbox=None; _ac_after=None
    def _hide_ac(self):
        if self._ac_listbox: self._ac_listbox.destroy(); self._ac_listbox=None
    def _on_ticker_key(self,e):
        if self._ac_after: self.parent.after_cancel(self._ac_after)
        self._ac_after=self.parent.after(300,self._do_ac_search)
    def _do_ac_search(self):
        t=self.ticker.get().strip()
        if len(t)<1: self._hide_ac(); return
        self._hide_ac()
        try:
            url=f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(t)}&count=10"
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=3) as r: data=json.loads(r.read().decode())
            qs=data.get("quotes",[]); 
            if not qs: return
            lines=[f"{q.get('symbol','?'):10s} {(q.get('shortname') or q.get('longname') or '')[:50]:50s} {q.get('quoteType','')}" for q in qs]
            mw=max(len(l) for l in lines)+2
            lb=tk.Listbox(self.parent,height=min(len(qs),8),font=FONT_SM,width=mw,bg=AX_BG,fg=FG,
                selectbackground=BTN_BG,selectforeground=FG,relief=tk.FLAT,borderwidth=0,highlightthickness=1,highlightbackground=SEP_COL)
            for l in lines: lb.insert(tk.END,l)
            lb.bind("<Button-1>",lambda e,lb_=lb:self._ac_pick_click(e,lb_))
            lb.bind("<FocusOut>",lambda e:self._hide_ac())
            ex=self.ticker_entry.winfo_rootx()-self.parent.winfo_rootx()
            ey=self.ticker_entry.winfo_rooty()-self.parent.winfo_rooty()+self.ticker_entry.winfo_height()
            lb.place(x=ex,y=ey,anchor="nw"); self._ac_listbox=lb
        except: pass
    def _ac_pick_click(self,e,lb):
        i=lb.nearest(e.y); sym=lb.get(i)[:10].strip(); self.ticker.set(sym); self._hide_ac(); self.fetch_data()
    def _ac_pick_first(self):
        if self._ac_listbox: sym=self._ac_listbox.get(0)[:10].strip(); self.ticker.set(sym); self._hide_ac(); self.fetch_data()

    # ================================================================
    # Data
    # ================================================================
    def fetch_data(self):
        raw=self.ticker.get().strip()
        if not raw: self.status_var.set("Enter or scan a ticker"); return
        resolved=self._resolve_ticker(raw)
        if not resolved: self.status_var.set(f"Could not resolve: {raw}"); return
        self.ticker.set(resolved); ticker=resolved
        self.status_var.set(f"Fetching {ticker}...")
        def task():
            try:
                df=fetch_realtime(ticker,interval="1d",period="1y")
                if df.empty or len(df)<10:
                    self.parent.after(0,lambda: self.status_var.set(f"No data for {ticker}")); return
                df=add_all_indicators(df,self.cfg.indicators); self.df=df
                now=datetime.now(); self.price_history.clear()
                for i in range(min(20,len(df))):
                    self.price_history.appendleft((now-timedelta(seconds=(20-i)*60),float(df["Close"].iloc[-(i+1)])))
                self.news_articles=[]; self.news_sentiment=None
                self._train_model(df)
                name=self._lookup_name(ticker)
                self.parent.after(0,lambda n=name: self.name_var.set(n))
                self.parent.after(0,lambda: self.status_var.set(f"{ticker} ready - predicting..."))
                self.parent.after(0,lambda t=ticker: self._fetch_and_show_stock_info(t))
                self.parent.after(0,self.run_news)
            except Exception as exc: self.parent.after(0,lambda e=str(exc): self.status_var.set(f"Error: {e}"))
        threading.Thread(target=task,daemon=True).start()

    def _resolve_ticker(self,t):
        t=t.strip().upper()
        if not t: return None
        try:
            url=f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(t)}"
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=3) as r: data=json.loads(r.read().decode())
            for q in data.get("quotes",[]):
                sym=q.get("symbol","").upper(); nu=(q.get("shortname") or q.get("longname") or "").upper()
                if sym.startswith(t) or t in nu: return sym
        except: pass
        return t
    def _lookup_name(self,t):
        try:
            url=f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(t)}"
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=3) as r: data=json.loads(r.read().decode())
            for q in data.get("quotes",[]):
                if q.get("symbol","").upper()==t.upper(): return q.get("shortname") or q.get("longname") or t
        except: pass
        return t

    def _train_model(self,df):
        try:
            tc=[c for c in df.columns if c.lower().startswith("target_")]
            if not tc:
                df=create_target_labels(df,self.cfg.data.prediction_horizon)
                tc=[c for c in df.columns if c.lower().startswith("target_")]
            if not tc: return
            fc=[c for c in df.columns if c not in tc and c not in ("Buy","Sell","Signal_Strength","Confluence_Score")
                and df[c].dtype in (np.float64,np.int64) and df[c].notna().sum()>len(df)*0.5]
            fc=[c for c in fc if c not in ("Open","High","Low","Close","Volume","target","Dividends","Stock Splits")]
            target_col="target_Class" if "target_Class" in df.columns else tc[0]
            is_class=target_col=="target_Class"
            train=df.dropna(subset=fc+[target_col])
            if len(train)<30: return
            split=int(len(train)*0.8)
            X_tr,y_tr=train[fc].values[:split],train[target_col].values[:split]
            X_va,y_va=train[fc].values[split:],train[target_col].values[split:]
            if is_class:
                y_tr=y_tr.astype(int); y_va=y_va.astype(int)
                # Only use classification if all 3 classes present in training
                if len(np.unique(y_tr))<2: is_class=False
            self._model=XGBoostModel(early_stopping_rounds=50, num_class=(3 if is_class and len(np.unique(y_tr))>=2 else None))
            self._model.train(X_tr,y_tr,X_va,y_va if is_class else None)
            self._feature_cols=fc; self._trained=True
        except Exception as exc: self._trained=False; self.status_var.set(f"Model: {exc}")
        # Fallback: if classification failed, try regression
        if not self._trained and is_class:
            try:
                target_col="target_Return"
                train=df.dropna(subset=fc+[target_col])
                if len(train)>=30:
                    self._model=XGBoostModel(early_stopping_rounds=50)
                    split=int(len(train)*0.8)
                    self._model.train(train[fc].values[:split],train[target_col].values[:split],
                                      train[fc].values[split:],train[target_col].values[split:])
                    self._feature_cols=fc; self._trained=True
            except: pass

    def _fetch_and_show_stock_info(self,ticker):
        def task():
            try:
                info=lookup_ticker_info(ticker)
                latest=fetch_latest_price(ticker)
                info["price"]=latest if latest else info.get("prev_close")
                self.stock_info=info
                self.parent.after(0,self._update_stock_info)
            except:
                pass
        threading.Thread(target=task,daemon=True).start()

    def _hook_poll_ticker(self):
        """Poll attached window title for ticker changes."""
        if not hasattr(self, '_attached_hwnd') or not self._attached_hwnd: return
        try:
            ln=ctypes.windll.user32.GetWindowTextLengthW(self._attached_hwnd)
            if ln==0: return
            b=ctypes.create_unicode_buffer(ln+1)
            ctypes.windll.user32.GetWindowTextW(self._attached_hwnd,b,ln+1)
            tkr=self._extract_ticker_from_title(b.value)
            if tkr and tkr!=self.ticker.get().upper():
                self.ticker.set(tkr)
                self.fetch_data()
        except: pass

    # ================================================================
    # News
    # ================================================================
    def run_news(self):
        t=self.ticker.get().strip().upper()
        if not t: return
        def task():
            try:
                arts=fetch_news(t,max_articles=9999)
                f=[a for a in arts if not (a.get("publisher","")or"").strip()
                    or any(r in (a.get("publisher","")or"").lower() for r in _REPUTABLE_SOURCES)]
                self.news_articles=f
                aggr=aggregate_sentiment(f) if f else {"avg":0,"pos":0,"neg":0,"neutral":0,"total":0}
                self.news_sentiment=aggr; self.parent.after(0,self._populate_news_tab)
            except Exception as exc: self.parent.after(0,lambda e=str(exc): self.status_var.set(f"News fetch: {e}"))
        threading.Thread(target=task,daemon=True).start()

    def _populate_news_tab(self):
        w=self.side_news_text; w.delete("1.0",tk.END)
        w.tag_config("g",foreground=GREEN); w.tag_config("r",foreground=RED); w.tag_config("b",foreground=FG)
        if self.news_sentiment:
            avg=self.news_sentiment.get("avg",0.0)
            icon="[+]" if avg>0.15 else("[-]" if avg<-0.15 else "[=]")
            tag="g" if avg>0.15 else("r" if avg<-0.15 else "b")
            w.insert(tk.END,"Overall Sentiment: ","b")
            w.insert(tk.END,f"{icon} {sentiment_label(avg)} (avg:{avg:+.2f} pos:{self.news_sentiment['pos']} neg:{self.news_sentiment['neg']})\n\n",tag)
        for i,art in enumerate(self.news_articles or []):
            try:
                title=art.get("title",""); pub=art.get("date",""); src=art.get("publisher","")
                sc=art.get("sentiment",0.0); icon="[+]" if sc>0.15 else("[-]" if sc<-0.15 else "[=]")
                tag="g" if sc>0.15 else("r" if sc<-0.15 else "b")
                line=f"{icon} {title}  [{src}]  ({pub})"[:200]
                w.insert(tk.END,f"{i+1}. {line}\n",tag)
            except: pass

    # ================================================================
    # Window Hook — read ticker from trading app window title
    # ================================================================
    def hook_app(self):
        """Enumerate visible windows, show picker, extract ticker from title."""
        self._hook_after=None
        wins=[]
        def cb(hwnd,_):
            if not ctypes.windll.user32.IsWindowVisible(hwnd): return True
            ln=ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if ln==0: return True
            b=ctypes.create_unicode_buffer(ln+1)
            ctypes.windll.user32.GetWindowTextW(hwnd,b,ln+1)
            t=b.value
            if t: wins.append((hwnd,t))
            return True
        W=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
        ctypes.windll.user32.EnumWindows(W(cb),0)
        self_hwnd=self.parent.winfo_toplevel().winfo_id()
        matches=[]
        for hwnd,title in wins:
            if hwnd==self_hwnd: continue
            tkr=self._extract_ticker_from_title(title)
            if tkr: matches.append((hwnd,title,tkr))
        if not matches:
            self.status_var.set("No windows with ticker-like titles found.")
            return
        self._show_app_picker(matches)

    def _extract_ticker_from_title(self,title):
        """Extract probable ticker from common window title patterns."""
        t=title.strip()
        # Pattern: "AAPL - Thinkorswim" or "AAPL Chart — TradingView"
        import re
        # Try known delimiters
        for sep in [" - "," \u2014 "," | "," – "]:
            parts=t.split(sep)
            if len(parts)>=2:
                cand=parts[0].strip().upper()
                if 1<=len(cand)<=6 and cand.isalpha():
                    return cand
        # Try finding any 1-6 letter all-alpha word that looks like a ticker
        words=re.findall(r'\b[A-Z]{1,6}\b',t)
        # Prefer words that match known ticker patterns (all caps, 1-6 letters)
        for w in words:
            if 1<=len(w)<=6 and w.isalpha():
                return w
        return None

    def _show_app_picker(self,matches):
        tl=tk.Toplevel(self.parent.winfo_toplevel())
        tl.title("Select Trading App"); tl.configure(bg=BG)
        tl.geometry("500x300+200+200"); tl.attributes("-topmost",True)
        tk.Label(tl,text="Select a window to hook + attach:",font=FONT,bg=BG,fg=FG).pack(pady=(10,5))
        lb=tk.Listbox(tl,font=FONT_SM,bg=AX_BG,fg=FG,selectbackground=BTN_BG,selectforeground=FG,
            relief=tk.FLAT,borderwidth=0,highlightthickness=1,highlightbackground=SEP_COL)
        lb.pack(fill=tk.BOTH,expand=True,padx=10,pady=5)
        for hwnd,title,tkr in matches:
            lb.insert(tk.END,f"{tkr:>6s}  {title[:80]}")
        def pick():
            sel=lb.curselection()
            if not sel: return
            hwnd,title,tkr=matches[sel[0]]
            self.ticker.set(tkr)
            tl.destroy()
            self.fetch_data()
            app=self.parent.winfo_toplevel()._app
            if app: app._attach_to(hwnd)
            self.status_var.set(f"Hooked + attached: {tkr}")
        def cancel():
            tl.destroy()
        bf=tk.Frame(tl,bg=BG); bf.pack(pady=5)
        tk.Button(bf,text="Select",command=pick,font=FONT,bg=BTN_BG,fg=FG,
            relief=tk.FLAT,padx=10).pack(side=tk.LEFT,padx=5)
        tk.Button(bf,text="Cancel",command=cancel,font=FONT,bg=BTN_BG,fg=FG,
            relief=tk.FLAT,padx=10).pack(side=tk.LEFT,padx=5)
        lb.bind("<Double-Button-1>",lambda e:pick())

    # ================================================================
    # Predict loop (1s)
    # ================================================================
    def _start_predict_loop(self): self._predict_tick()

    def _predict_tick(self):
        try:
            self._hook_poll_ticker()
            # Ensure price_history always has at least one entry
            if len(self.price_history)==0:
                if not self.df.empty:
                    lc=float(self.df["Close"].iloc[-1]); nw=datetime.now()
                    for i in range(min(10,len(self.df))):
                        self.price_history.appendleft((nw-timedelta(seconds=(10-i)*60),float(self.df["Close"].iloc[-(i+1)])))
                elif self._latest_price>0:
                    self.price_history.append((datetime.now(),self._latest_price))
            if len(self.price_history)>0:
                self._run_prediction(self.price_history[-1][1])
                self._update_predict_display()
                if self.stock_info:
                    self.stock_info["price"]=self._latest_price
                    self._update_stock_info()
            else:
                self._update_predict_text("Waiting...\nEnter ticker + Refresh.")
        except Exception as exc: self.status_var.set(f"Loop: {exc}")
        finally: self.timer_id=self.parent.after(1000,self._predict_tick)

    # ================================================================
    # Prediction engine  —  confluence-based indicator scoring
    # ================================================================
    def _ind_signal(self,val,lo,hi,wt):
        """Return (dir, weight) for a single indicator."""
        if val<=lo: return (1,wt)
        if val>=hi: return (-1,wt)
        return (0,0)

    def _run_prediction(self,price):
        n=len(self.price_history)
        if n<2: return

        prices=np.array([p for _,p in self.price_history])
        times_arr=np.array([t for t,_ in self.price_history])
        lp=prices[-1]; self._latest_price=lp
        ret=np.diff(prices)/prices[:-1]
        def ma(x,w): return np.mean(x[-w:]) if len(x)>=w else np.mean(x)
        sma5=ma(prices,5); sma10=ma(prices,10)
        m1=(prices[-1]/prices[-2]-1)*100 if n>=2 else 0
        m3=(prices[-1]/prices[-min(3,n)]-1)*100; m5=(prices[-1]/prices[-min(5,n)]-1)*100
        vs5=(lp/sma5-1)*100 if sma5>0 else 0
        vol=np.std(ret[-min(10,len(ret)):])*100 if len(ret)>=2 else 0
        hh=np.max(prices[-5:]) if n>=5 else np.max(prices); ll=np.min(prices[-5:]) if n>=5 else np.min(prices)
        rsv=(lp-ll)/max(hh-ll,1e-8)*100
        uc=0;dc=0
        for i in range(min(10,n-1)):
            if prices[-(i+1)]>prices[-(i+2)]: uc+=1;dc=0
            elif prices[-(i+1)]<prices[-(i+2)]: dc+=1;uc=0
            else: break
        ph=np.max(prices); pl=np.min(prices)
        self._mom1=m1; self._mom5=m5; self._vol=vol; self._pl=pl; self._ph=ph; self._rsv=rsv

        navg=0.0
        if self.news_sentiment and isinstance(self.news_sentiment,dict): navg=self.news_sentiment.get("avg",0.0)
        self._navg=navg

        # ---- Confluence-based rule system ----
        votes=[]  # (direction, weight, label)

        # 1. Price position in range (oversold / overbought)
        d,w=self._ind_signal(rsv,20,80,0.6)
        if d:
            votes.append((d,w,"Price range"))
            # stronger conviction at extremes
            if rsv<10: votes.append((1,0.3,"Extreme oversold"))
            elif rsv>90: votes.append((-1,0.3,"Extreme overbought"))

        # 2. Short-term momentum
        if m1>0.8 and m5>1.5: votes.append((1,0.5,"Momentum up"))
        elif m1<-0.8 and m5<-1.5: votes.append((-1,0.5,"Momentum down"))
        elif m1>0.3: votes.append((1,0.2,"Mild uptick"))
        elif m1<-0.3: votes.append((-1,0.2,"Mild downtick"))

        # 3. Consecutive bars
        if uc>=5 and m3>0.3: votes.append((1,0.4,"Consecutive up"))
        elif dc>=5 and m3<-0.3: votes.append((-1,0.4,"Consecutive down"))
        elif uc>=3: votes.append((1,0.15,"Short up streak"))
        elif dc>=3: votes.append((-1,0.15,"Short down streak"))

        # 4. SMA offset + range confirmation
        if vs5<-0.3 and rsv<40: votes.append((1,0.5,"Below SMA5 + low"))
        elif vs5>0.3 and rsv>60: votes.append((-1,0.5,"Above SMA5 + high"))
        if vs5<-0.5: votes.append((1,0.3,"Well below SMA5"))
        elif vs5>0.5: votes.append((-1,0.3,"Well above SMA5"))

        # 5. Volatility confirmation
        if vol>2.0:
            if rsv<20: votes.append((1,0.4,"High vol + oversold"))
            elif rsv>80: votes.append((-1,0.4,"High vol + overbought"))

        # 6. Daily technical indicators (from df) — confluence layer
        df_ok=not self.df.empty
        df_row=self.df.iloc[-1] if df_ok else None
        if df_row is not None:
            # RSI
            rsi_v=float(df_row.get("RSI_14",50))
            d,w=self._ind_signal(rsi_v,30,70,0.7)
            if d: votes.append((d,w,f"RSI {rsi_v:.0f}"))

            # Stochastic
            sk=float(df_row.get("Stoch_%K",50))
            sd=float(df_row.get("Stoch_%D",50))
            d,w=self._ind_signal(sk,20,80,0.5)
            if d and abs(sk-sd)<5: votes.append((d,w+0.2,"Stoch %K/%D aligned"))
            elif d: votes.append((d,w,"Stochastic"))

            # Bollinger Bands
            bb_pos=float(df_row.get("BB_position",0.5))
            d,w=self._ind_signal(bb_pos*100,20,80,0.65)
            if d: votes.append((d,w,"Bollinger"))

            # MACD histogram
            macd_h=float(df_row.get("MACD_hist",0))
            if macd_h>0: votes.append((1,0.35,"MACD+"))
            elif macd_h<0: votes.append((-1,0.35,"MACD-"))

            # ADX trend strength
            adx_v=float(df_row.get("ADX",25))
            di_p=float(df_row.get("DI_plus",25))
            di_m=float(df_row.get("DI_minus",25))
            if adx_v>25:
                if di_p>di_m: votes.append((1,0.5,f"ADX {adx_v:.0f} +DI↑"))
                elif di_m>di_p: votes.append((-1,0.5,f"ADX {adx_v:.0f} -DI↑"))
                else: pass
            elif adx_v>20:
                if di_p>di_m: votes.append((1,0.2,"Trend building +DI"))
                elif di_m>di_p: votes.append((-1,0.2,"Trend building -DI"))

            # MFI (volume-weighted RSI)
            mfi_v=float(df_row.get("MFI",50))
            d,w=self._ind_signal(mfi_v,20,80,0.5)
            if d: votes.append((d,w,f"MFI {mfi_v:.0f}"))

            # Keltner Channel
            kp=float(df_row.get("Keltner_Pos",0.5))
            d,w=self._ind_signal(kp*100,20,80,0.4)
            if d: votes.append((d,w,"Keltner Ch."))

            # Aroon oscillator
            aroon=float(df_row.get("Aroon_Osc",0))
            if aroon>50: votes.append((1,0.3,"Aroon uptrend"))
            elif aroon<-50: votes.append((-1,0.3,"Aroon downtrend"))

            # Pivot distance
            pdist=float(df_row.get("Pivot_Dist",0.5))
            d,w=self._ind_signal(pdist*100,20,80,0.3)
            if d: votes.append((d,w,"Pivot level"))

            # Market regime context
            regime=float(df_row.get("Regime_SMA",1.0))
            regime_vol=float(df_row.get("Regime_Volatility",0.2))
            if regime>1.05: votes.append((1,0.2,"Bull market tailwind"))
            elif regime<0.95: votes.append((-1,0.2,"Bear market headwind"))
            if regime_vol>0.3: votes.append((-1,0.1,"High vol regime"))

        # ---- Store indicator values for explanation ----
        self._ind_values = {}
        self._ind_values["rsi"] = float(df_row.get("RSI_14", 50)) if df_row is not None else 50
        self._ind_values["stoch_k"] = float(df_row.get("Stoch_%K", 50)) if df_row is not None else 50
        self._ind_values["stoch_d"] = float(df_row.get("Stoch_%D", 50)) if df_row is not None else 50
        self._ind_values["bb_pos"] = float(df_row.get("BB_position", 0.5)) if df_row is not None else 0.5
        self._ind_values["macd_h"] = float(df_row.get("MACD_hist", 0)) if df_row is not None else 0
        self._ind_values["adx"] = float(df_row.get("ADX", 0)) if df_row is not None else 0
        self._ind_values["mfi"] = float(df_row.get("MFI", 50)) if df_row is not None else 50
        self._ind_values["aroon"] = float(df_row.get("Aroon_Osc", 0)) if df_row is not None else 0
        self._ind_values["keltner"] = float(df_row.get("Keltner_Pos", 0.5)) if df_row is not None else 0.5
        self._ind_values["regime"] = float(df_row.get("Regime_SMA", 1)) if df_row is not None else 1
        self._ind_values["regime_vol"] = float(df_row.get("Regime_Volatility", 0.2)) if df_row is not None else 0.2
        self._ind_values["uc"] = uc
        self._ind_values["dc"] = dc
        self._ind_values["votes"] = votes
        self._ind_values["rsv"] = rsv

        # ---- Aggregate confluence ----
        buy_w=sum(weight for d,weight,_ in votes if d==1)
        sell_w=sum(weight for d,weight,_ in votes if d==-1)
        total=len(votes)
        net=buy_w-sell_w
        mag=max(buy_w,sell_w,0.1)
        rd=0 if abs(net)/mag<0.15 else (1 if net>0 else -1)
        rc=min(abs(net)/mag*95,95)

        # Confluence bonus: when many indicators agree
        agreeing=sum(1 for d,_,_ in votes if d==rd)
        if agreeing>=4: rc=min(rc+12,95)
        elif agreeing>=3: rc=min(rc+8,95)

        # ---- ML (classification with probabilities) ----
        ml_dir,ml_conf,ml_reason=0,0.0,None
        if self._trained and self._model and self._feature_cols is not None and not self.df.empty:
            try:
                lr=self.df.iloc[-1:].copy(); lr["Close"]=lp
                feats={}
                for c in self._feature_cols:
                    alias={
                        "price_change":m1/100,"price_change_5":m5/100,"price_change_21":(prices[-1]/prices[-min(21,n)]-1 if n>=21 else 0),
                        "sma_5_off":vs5/100,"volatility_10":vol/100,"momentum_1":m1/100,"momentum_5":m5/100,
                        "high_low_ratio":(hh/ll if ll>0 else 1),"close_to_low":((lp-pl)/max(ph-pl,1e-8)),
                    }
                    c_lower=c.lower()
                    if c_lower in alias: feats[c]=alias[c_lower]
                    elif c in lr.columns: feats[c]=float(lr[c].values[0])
                    else: feats[c]=0.0
                X=np.array([[feats.get(c,0.0) for c in self._feature_cols]])
                if hasattr(self._model,"predict_proba"):
                    probs=self._model.predict_proba(X)[0]
                    pred_cls=int(np.argmax(probs))
                    ml_conf_v=float(probs[pred_cls])*100
                    # Map class -1,0,1 or 0,1,2
                    if len(probs)==3:
                        cls_map={0:-1,1:0,2:1}
                        ml_dir=cls_map.get(pred_cls,0)
                    elif len(probs)==2:
                        ml_dir=1 if pred_cls==1 else -1
                    else:
                        ml_dir=0
                    ml_conf=ml_conf_v
                    ppt_val=float(self._model.predict(X)[0]) if hasattr(self._model,"predict") else 0
                    ppt=lp*(1+ppt_val*0.01) if abs(ppt_val)<5 else None
                    ml_reason=f"ML (cls {ml_conf:.0f}%)"
                else:
                    rp=float(self._model.predict(X)[0])
                    if abs(rp)<0.001: ml_dir=0
                    elif rp>0: ml_dir=1
                    else: ml_dir=-1
                    ml_conf=min(abs(rp)*500,80)
                    ppt=lp*(1+rp)
                    ml_reason=f"ML (reg {ml_conf:.0f}%)"
            except Exception:
                pass

        # ---- Combine ML + Rules ----
        if ml_dir!=0 and ml_conf>30:
            if ml_dir==rd and rc>15:
                rc=min(rc+ml_conf*0.3,95)
                rd=ml_dir
                if ml_reason: ml_reason+=" confirmed rules"
            elif ml_dir!=rd:
                rc*=0.65
                if abs(mag)<0.3: rd=ml_dir; rc=max(ml_conf*0.5,rc)
                if ml_reason: ml_reason+=" conflicts rules"
        else:
            ml_reason=None

        # ---- News sentiment adjustment ----
        if navg>0.15:
            if rd==-1: rc*=0.75
            elif rd==1: rc=min(rc*1.15,95)
        elif navg<-0.15:
            if rd==1: rc*=0.75
            elif rd==-1: rc=min(rc*1.15,95)
        rc=min(max(rc,0),98)

        self._pred_dir=rd; self._pred_conf=rc
        self._ml_dir = ml_dir; self._ml_conf = ml_conf

        # ---- Build reasoning ----
        reasons=[]
        if self._trained: reasons.append(f"ML on {len(self.df)} bars")
        if ml_reason: reasons.append(ml_reason)
        # Top vote-getting indicators  (tuple: direction, weight, label)
        for d,w,label in sorted(votes,key=lambda x:abs(x[1]),reverse=True)[:4]:
            tag="BUY" if d==1 else("SELL" if d==-1 else "?")
            reasons.append(f"{label} -> {tag}")
        if navg!=0: reasons.append(f"News {navg:+.2f} ({sentiment_label(navg)})")
        reasons.append(f"Range ${pl:.2f}-${ph:.2f} | Vol {vol:.2f}%")
        self._reasons=reasons

        self._pred_text_kwargs = dict(lp=lp,m1=m1,m5=m5,vs5=vs5,vol=vol,pl=pl,ph=ph,n=n,
            navg=navg,m3=m3,times_arr=times_arr,ppt=None,reasons=reasons)

    def _generate_explanation(self):
        pd_=self._pred_dir; pc=self._pred_conf
        ds="BUY" if pd_==1 else("SELL" if pd_==-1 else "HOLD")
        iv=self._ind_values
        m1=self._mom1; m5=self._mom5; vol=self._vol
        navg=self._navg; rsv=iv.get("rsv",50)

        parts=[f"The model recommends {ds} with {pc:.0f}% confidence based on the following analysis."]

        # Momentum
        if abs(m1)>0.3:
            parts.append(f"Price momentum is {'up' if m1>0 else 'down'} {abs(m1):.2f}% in the last minute, "
                         f"{'supporting a bullish bias' if m1>0 else 'suggesting bearish pressure'}.")
        else:
            parts.append("Short-term momentum is relatively flat, offering no strong directional signal.")

        # RSI
        rsi_v=iv.get("rsi",50)
        if rsi_v>70:
            parts.append(f"RSI at {rsi_v:.0f} is in overbought territory, which historically signals a potential pullback or trend exhaustion.")
        elif rsi_v<30:
            parts.append(f"RSI at {rsi_v:.0f} is in oversold territory, often indicating the asset may be undervalued and due for a bounce.")
        else:
            parts.append(f"RSI at {rsi_v:.0f} is in neutral range, showing no extreme overbought or oversold conditions.")

        # Stochastic
        sk=iv.get("stoch_k",50)
        if sk>80:
            parts.append(f"Stochastic %K at {sk:.0f} is elevated, reinforcing the overbought picture.")
        elif sk<20:
            parts.append(f"Stochastic %K at {sk:.0f} is deeply oversold, confirming potential for a reversal upward.")

        # MACD
        mh=iv.get("macd_h",0)
        if mh>0:
            parts.append(f"The MACD histogram is positive ({mh:.3f}), indicating bullish momentum in the medium term.")
        elif mh<0:
            parts.append(f"The MACD histogram is negative ({mh:.3f}), reflecting bearish momentum.")

        # Bollinger
        bb=iv.get("bb_pos",0.5)
        if bb<0.2:
            parts.append("Price is near the lower Bollinger Band, suggesting the asset may be oversold and could revert upward.")
        elif bb>0.8:
            parts.append("Price is near the upper Bollinger Band, suggesting overextension to the upside.")

        # Volatility
        if vol>2:
            parts.append(f"Volatility is elevated at {vol:.2f}%, adding risk and suggesting larger-than-usual price swings.")
        elif vol<0.5:
            parts.append(f"Volatility is low at {vol:.2f}%, indicating a relatively stable price environment.")

        # Consecutive bars
        uc=iv.get("uc",0); dc=iv.get("dc",0)
        if uc>=5:
            parts.append(f"Price has been up {uc} consecutive periods, showing strong directional persistence.")
        elif dc>=5:
            parts.append(f"Price has been down {dc} consecutive periods, indicating sustained selling pressure.")

        # ADX trend strength
        adx_v=iv.get("adx",0)
        if adx_v>25:
            parts.append(f"ADX at {adx_v:.0f} indicates a strong trending market, which amplifies the conviction of directional signals.")
        elif adx_v>20:
            parts.append(f"ADX at {adx_v:.0f} suggests a developing trend, so directional signals are treated with moderate confidence.")

        # MFI
        mfi_v=iv.get("mfi",50)
        if mfi_v>80:
            parts.append(f"Money Flow Index at {mfi_v:.0f} is overbought on a volume-weighted basis, suggesting distribution.")
        elif mfi_v<20:
            parts.append(f"Money Flow Index at {mfi_v:.0f} is oversold, indicating accumulation on high volume.")

        # Aroon
        aroon=iv.get("aroon",0)
        if aroon>50:
            parts.append(f"Aroon Oscillator at {aroon:.0f} confirms a strong uptrend with fresh highs being made.")
        elif aroon<-50:
            parts.append(f"Aroon Oscillator at {aroon:.0f} confirms a strong downtrend with fresh lows being made.")

        # Market regime
        regime=iv.get("regime",1.0)
        regime_vol=iv.get("regime_vol",0.2)
        if regime>1.05:
            parts.append(f"The broader market regime is bullish (price {((regime-1)*100):.0f}% above 200-SMA), favoring long positions overall.")
        elif regime<0.95:
            parts.append(f"The broader market regime is bearish (price {((1-regime)*100):.0f}% below 200-SMA), favoring caution on longs.")
        if regime_vol>0.3:
            parts.append(f"Annualized volatility is elevated at {regime_vol*100:.0f}%, indicating a high-risk environment.")
        elif regime_vol<0.15:
            parts.append(f"Annualized volatility is low at {regime_vol*100:.0f}%, suggesting a calm market.")

        # ML
        if self._trained:
            ml_dir=self._ml_dir; ml_conf=self._ml_conf
            ml_tag="BUY" if ml_dir==1 else("SELL" if ml_dir==-1 else "HOLD")
            if ml_conf>30:
                parts.append(f"The ML model (XGBoost trained on {len(self.df)} bars) agrees with a {ml_tag} signal "
                             f"at {ml_conf:.0f}% probability, which has been factored into the final confidence.")
            else:
                parts.append(f"The ML model outputs a {ml_tag} signal but at low confidence ({ml_conf:.0f}%), so rule-based indicators are given more weight.")

        # News
        if abs(navg)>0.15:
            snl=sentiment_label(navg)
            parts.append(f"News sentiment is {snl} ({navg:+.2f}), {'reinforcing' if (navg>0.15)==(pd_==1) else 'partially offsetting'} the overall direction.")

        # Final summary
        votes=iv.get("votes",[])
        try:
            buy_w=sum(float(w) for d,_,w in votes if d==1)
            sell_w=sum(float(w) for d,_,w in votes if d==-1)
        except (TypeError, ValueError):
            buy_w=0; sell_w=0
        if buy_w>sell_w and pd_==1:
            parts.append(f"Overall, {len([v for v in votes if v[0]==1])} indicators favor buying with total weight {buy_w:.1f} vs {sell_w:.1f} for selling, leading to the {ds} recommendation.")
        elif sell_w>buy_w and pd_==-1:
            parts.append(f"Overall, {len([v for v in votes if v[0]==-1])} indicators favor selling with total weight {sell_w:.1f} vs {buy_w:.1f} for buying, supporting the {ds} call.")
        else:
            parts.append(f"The evidence is mixed ({buy_w:.1f} buy vs {sell_w:.1f} sell weight), resulting in a low-conviction {ds}.")

        return " ".join(parts)

    def _update_predict_display(self):
        kw=getattr(self,"_pred_text_kwargs",None)
        if not kw: return
        lp=kw["lp"]; m1=kw["m1"]; m5=kw["m5"]; vs5=kw["vs5"]; vol=kw["vol"]
        pl=kw["pl"]; ph=kw["ph"]; n=kw["n"]; navg=kw["navg"]; times_arr=kw["times_arr"]
        ppt=kw["ppt"]; reasons=kw["reasons"]
        pd_=self._pred_dir; pc=self._pred_conf
        si="[+]" if navg>0.15 else("[-]" if navg<-0.15 else "[=]")
        sl=sentiment_label(navg)
        ds="BUY" if pd_==1 else("SELL" if pd_==-1 else "HOLD")
        di="[+]" if pd_==1 else("[-]" if pd_==-1 else "[=]")
        tss=(times_arr[-1]-times_arr[0]).total_seconds() if len(times_arr)>1 else 0
        cpm=n/max(tss/60,0.1)
        tl=f"Target: ${ppt:.2f}\n" if ppt else ""
        explanation=self._generate_explanation()
        self._update_predict_text(
            f"Live Screen Analysis\n{'='*30}\n"
            f"Rec: {di} {ds}\nConf: {pc:.0f}%\n"
            f"Price: ${lp:.2f} ({m1:+.2f}%)\n"
            f"News: {si} {sl}\n{'='*30}\n{tl}"
            f"Momentum(5): {m5:+.2f}%\nSMA5 off: {vs5:+.2f}%\n"
            f"Volatility: {vol:.2f}%\nRange: ${pl:.2f} - ${ph:.2f}\n"
            f"Samples: {n} @ {cpm:.0f}/min\n{'='*30}\nAnalysis:\n"
            + "\n".join(f"- {r}" for r in reasons)
            + f"\n{'='*30}\nAI Explanation:\n{explanation}"
            + f"\n{'='*30}\nLast: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _update_predict_text(self,content):
        w=self.side_predict_text; w.delete("1.0",tk.END)
        w.tag_config("g",foreground=GREEN); w.tag_config("r",foreground=RED); w.tag_config("b",foreground=FG)
        for line in content.split("\n"):
            tag="g" if line.startswith("Rec: [+]") or "[+]" in line else("r" if line.startswith("Rec: [-]") or "[-]" in line else "b")
            w.insert(tk.END,line+"\n",tag)

    def _build_stock_info(self,parent):
        r1=tk.Frame(parent,bg=AX_BG); r1.pack(fill=tk.X,padx=4,pady=(2,0))
        self._stock_labels["ticker"]=tk.Label(r1,text="",font=("Arial",10,"bold"),bg=AX_BG,fg=FG)
        self._stock_labels["ticker"].pack(side=tk.LEFT)
        self._stock_labels["name"]=tk.Label(r1,text="",font=FONT_SM,bg=AX_BG,fg="#888888")
        self._stock_labels["name"].pack(side=tk.LEFT,padx=(6,0))

        r2=tk.Frame(parent,bg=AX_BG); r2.pack(fill=tk.X,padx=4,pady=(1,0))
        self._stock_labels["price"]=tk.Label(r2,text="",font=("Arial",14,"bold"),bg=AX_BG,fg=FG)
        self._stock_labels["price"].pack(side=tk.LEFT)
        self._stock_labels["change"]=tk.Label(r2,text="",font=("Arial",10),bg=AX_BG,fg=GREEN)
        self._stock_labels["change"].pack(side=tk.LEFT,padx=(6,0))

        r3=tk.Frame(parent,bg=AX_BG); r3.pack(fill=tk.X,padx=4,pady=(0,2))
        self._stock_labels["details"]=tk.Label(r3,text="",font=FONT_SM,bg=AX_BG,fg="#aaaaaa",anchor=tk.W)
        self._stock_labels["details"].pack(fill=tk.X)

    def _fmt_volume(self,v):
        if v>=1e9: return f"{v/1e9:.2f}B"
        if v>=1e6: return f"{v/1e6:.1f}M"
        if v>=1e3: return f"{v/1e3:.0f}K"
        return str(int(v))

    def _fmt_market_cap(self,c):
        if c>=1e12: return f"${c/1e12:.2f}T"
        if c>=1e9: return f"${c/1e9:.2f}B"
        if c>=1e6: return f"${c/1e6:.2f}M"
        return f"${c:.0f}"

    def _update_stock_info(self):
        if not self.stock_info: return
        i=self.stock_info; tkr=self.ticker.get().strip().upper()
        nm=i.get("name","")[:40]; pr=i.get("price",0)
        pc=i.get("prev_close"); op=i.get("open"); dh=i.get("day_high"); dl=i.get("day_low")
        vo=i.get("volume"); mc=i.get("market_cap"); pe=i.get("pe_ratio")
        wh=i.get("52w_high"); wl=i.get("52w_low")

        self._stock_labels["ticker"].config(text=tkr)
        self._stock_labels["name"].config(text=nm)

        if pr and pc:
            ch=pr-pc; cp=(ch/pc*100) if pc else 0
            cl=GREEN if ch>=0 else RED; ar="▲" if ch>=0 else "▼"
            self._stock_labels["price"].config(text=f"${pr:.2f}")
            self._stock_labels["change"].config(text=f"{ar} {ch:+.2f} ({cp:+.2f}%)",fg=cl)
        elif pr:
            self._stock_labels["price"].config(text=f"${pr:.2f}")
            self._stock_labels["change"].config(text="")

        det=[]
        if dl and dh: det.append(f"Day: ${dl:.2f} - ${dh:.2f}")
        if op: det.append(f"Open: ${op:.2f}")
        if vo: det.append(f"Vol: {self._fmt_volume(int(vo))}")
        if mc: det.append(f"Mkt Cap: {self._fmt_market_cap(mc)}")
        if pe: det.append(f"PE: {pe:.1f}")
        if wl and wh: det.append(f"52W: ${wl:.2f} - ${wh:.2f}")
        self._stock_labels["details"].config(text=" | ".join(det))

    def destroy(self):
        if self.timer_id: self.parent.after_cancel(self.timer_id)


class OverlayApp:
    def __init__(self,root:tk.Tk):
        self.root=root; self.root.title("Overlay Predictor"); self.root.configure(bg=BG)
        self.root.geometry("650x600")
        self._attached_hwnd=None; self._attach_timer=None
        main=tk.Frame(root,bg=BG); main.pack(fill=tk.BOTH,expand=True,padx=4,pady=4)
        self.chart=LiveChart(main,Config())
        self.root.attributes("-topmost",True)
        self.root._app=self  # allow chart to reach us
        # Start API server for browser injection
        self._api=ApiServer(self.chart,port=8765); self._api.start()
        self.chart.status_var.set(f"Server running at {self._api.url} - Install inject/overlay.user.js in Tampermonkey")
        self.root.protocol("WM_DELETE_WINDOW",self.on_close)

    def _attach_to(self,target_hwnd):
        """Make this window a child of target_hwnd and position at its right edge."""
        self._attached_hwnd=target_hwnd
        if hasattr(self.chart,'_attached_hwnd'): self.chart._attached_hwnd=target_hwnd
        user32=ctypes.windll.user32
        user32.SetParent(self.root.winfo_id(),target_hwnd)
        self.root.attributes("-topmost",False)
        self._position_attached()
        self._attach_poll()

    def _detach(self):
        if self._attached_hwnd:
            ctypes.windll.user32.SetParent(self.root.winfo_id(),0)
            self.root.attributes("-topmost",True)
            self._attached_hwnd=None
        if self._attach_timer:
            self.root.after_cancel(self._attach_timer); self._attach_timer=None

    def _position_attached(self):
        if not self._attached_hwnd: return
        user32=ctypes.windll.user32; rect=wintypes.RECT()
        if not user32.GetWindowRect(self._attached_hwnd,ctypes.byref(rect)): self._detach(); return
        my_w=max(self.root.winfo_width(),50); my_h=max(self.root.winfo_height(),50)
        user32.SetWindowPos(self.root.winfo_id(),0,rect.right+2,rect.top,my_w,my_h,0x0004)

    def _attach_poll(self):
        if not self._attached_hwnd: return
        self._position_attached()
        self._attach_timer=self.root.after(500,self._attach_poll)

    def on_close(self):
        self._detach(); self.chart.destroy(); self._api.stop(); self.root.destroy()

def main():
    root=tk.Tk(); app=OverlayApp(root); root.mainloop()
if __name__=="__main__": main()
