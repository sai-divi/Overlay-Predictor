import json, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime


def _make_handler(chart):
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin","*")
            self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers","Content-Type")
            self.end_headers()

        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=="/api/status":
                t=chart.ticker.get().strip() if chart else ""
                self._json({"status":"ok" if chart else "no_chart","ticker":t,
                    "trained":chart._trained if chart else False,
                    "samples":len(chart.price_history) if chart else 0})
            elif parsed.path.startswith("/api/train"):
                p=parse_qs(parsed.query); ticker=(p.get("ticker") or [""])[0].upper()
                if ticker and chart:
                    chart.ticker.set(ticker)
                    threading.Thread(target=chart.fetch_data,daemon=True).start()
                self._json({"training":True,"ticker":ticker})
            else:
                self._json({"error":"not found"},404)

        def do_POST(self):
            parsed=urlparse(self.path)
            if parsed.path!="/api/predict":
                self._json({"error":"not found"},404); return
            try:
                clen=int(self.headers.get("Content-Length",0))
                body=json.loads(self.rfile.read(clen).decode("utf-8")) if clen else {}
            except Exception:
                self._json({"error":"bad request"},400); return
            ticker=(body.get("ticker") or "").upper().strip()
            price=body.get("price")
            if not ticker or price is None or not chart:
                self._json({"error":"missing ticker or price"},400); return
            if chart.ticker.get().upper()!=ticker:
                chart.ticker.set(ticker)
                chart._trained=False
                threading.Thread(target=chart.fetch_data,daemon=True).start()
            try:
                chart.price_history.append((datetime.now(),float(price)))
                if not chart._trained and not chart.df.empty:
                    chart._train_model(chart.df)
                chart._run_prediction(float(price))
                try: chart.parent.after(0,chart._update_predict_display)
                except: pass
            except Exception as exc:
                self._json({"error":str(exc)},500); return
            dm={1:"BUY",-1:"SELL",0:"HOLD"}
            self._json({
                "direction":chart._pred_dir,"prediction":dm.get(chart._pred_dir,"HOLD"),
                "confidence":chart._pred_conf,"latest_price":chart._latest_price,
                "reasons":chart._reasons[:4],"trained":chart._trained,
                "samples":len(chart.price_history),
                "news_sentiment":chart._navg})

        def _json(self,data,status=200):
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self,fmt,*a): pass
    return Handler


class ApiServer:
    def __init__(self,chart,port=8765):
        self.chart=chart; self.port=port; self._server=None; self._thread=None

    def start(self):
        if self._server: return
        try:
            self._server=HTTPServer(("127.0.0.1",self.port),_make_handler(self.chart))
            self._thread=threading.Thread(target=self._server.serve_forever,daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"API server failed to start on port {self.port}: {e}")

    def stop(self):
        if self._server:
            self._server.shutdown(); self._server=None

    @property
    def url(self): return f"http://localhost:{self.port}"
