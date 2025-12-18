from __future__ import annotations

import time
import math
import random
import threading
from dataclasses import dataclass
from typing import Dict, List, Any
import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

# -----------------------
# Global sim constants
# -----------------------

DAY_MIN = 1440
TARGET_SESSIONS_PER_DAY = 6800

# economics
MARGIN_PER_KWH = 0.004          # $0.004 per kWh
PROTOCOL_FEE_SHARE = 0.20       # 20% to protocol


VEHICLE_TYPES = [
    {"type": "car", "peakKW": 40.0, "baseDurMin": 90.0, "prob": 0.01},
    {"type": "3w",  "peakKW": 4.0,  "baseDurMin": 60.0, "prob": 0.29},
    {"type": "2w",  "peakKW": 2.0,  "baseDurMin": 45.0, "prob": 0.70},
]

THEME = {
    "bgGrid": "rgba(255,255,255,0.10)",
    "powerLine": "rgba(0, 255, 200, 0.95)",
    "powerDash": "rgba(0, 255, 200, 0.55)",
    "oiLine": "rgba(255, 215, 0, 0.95)",
    "oiDash": "rgba(255, 215, 0, 0.55)",
    "scatter": "rgba(185, 120, 255, 0.85)",
    "topBarFill": "rgba(80, 220, 140, 0.70)",
    "topBarStroke": "rgba(80, 220, 140, 0.95)",
    "text": "rgba(255,255,255,0.78)",
}

# -----------------------
# Redis: shared snapshot
# -----------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)
SNAPSHOT_KEY = "degen:mark1:snapshot"


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def rand(a: float, b: float) -> float:
    return a + random.random() * (b - a)


def fmt_time(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds // 60) % 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def corr(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        vx = xs[i] - mx
        vy = ys[i] - my
        num += vx * vy
        dx += vx * vx
        dy += vy * vy
    den = math.sqrt(dx * dy) or 1.0
    return clamp(num / den, -1.0, 1.0)


@dataclass
class Session:
    id: int
    vtype: str
    startMin: float
    durMin: float
    peakKW: float
    rampMin: float
    holdMin: float
    taperMin: float


class Config(BaseModel):
    # target 19 MWh / day
    targetMWh: float = 19.0
    mode: str = "office"
    capacity: int = 1000
    # fixed simulated minutes per engine step
    simMinPerStep: float = 0.25   # 15 seconds of sim time per step
    engineHz: float = 4.0         # 4 steps/sec => 60x realtime (fine for a dashboard)


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = Config()

        self.paused = False
        self.simMin = 0.0
        self.simSec = 0
        self.energyKWh = 0.0
        self.sessions: List[Session] = []
        self.sessionId = 1

        # revenue metrics
        self.lifetimeRevenueUSD = 0.0  # cumulative margin across all time

        self.series_t: List[float] = []
        self.series_pkw: List[float] = []
        self.series_oi: List[float] = []
        self.max_points = 420

        self.tape: List[Dict[str, Any]] = []
        self.tape_max = 32

        self.arrivalScale = 1.0
        self.arrivalNudge = 1.0

        self._recompute_arrival_scale()

    def _push_tape(self, kind: str, msg: str):
        t = fmt_time(int(self.simMin * 60))
        self.tape.insert(0, {"t": t, "msg": msg, "kind": kind})
        if len(self.tape) > self.tape_max:
            self.tape.pop()

    def _base_arrival_rate_per_min(self, mode: str, t01: float) -> float:
        twoPi = math.pi * 2
        if mode == "flat":
            base = 0.14 + 0.02 * math.sin(twoPi * (t01 - 0.15))
        elif mode == "residential":
            morning = math.exp(-((t01 - 0.30) / 0.08) ** 2)
            evening = math.exp(-((t01 - 0.80) / 0.10) ** 2)
            base = 0.06 + 0.12 * morning + 0.25 * evening
        elif mode == "degen":
            day = math.exp(-((t01 - 0.55) / 0.16) ** 2)
            base = 0.10 + 0.22 * day + 0.10 * max(0.0, math.sin(twoPi * (t01 * 3.7)))
        else:  # office
            day = math.exp(-((t01 - 0.55) / 0.16) ** 2)
            base = 0.05 + 0.26 * day

        # overall scaling to make things lively
        return base * 4.0

    def _recompute_arrival_scale(self):
        mode = self.cfg.mode
        samples = 240
        s = 0.0
        for i in range(samples):
            t01 = i / (samples - 1)
            s += self._base_arrival_rate_per_min(mode, t01)
        avg = s / samples
        self.arrivalScale = TARGET_SESSIONS_PER_DAY / max(1e-6, (avg * DAY_MIN))
        self.arrivalScale = clamp(self.arrivalScale, 0.25, 10.0)

    def _sample_poisson(self, lam: float) -> int:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def _pick_vehicle(self) -> Dict[str, Any]:
        r = random.random()
        acc = 0.0
        for v in VEHICLE_TYPES:
            acc += v["prob"]
            if r <= acc:
                return v
        return VEHICLE_TYPES[-1]

    def _power_at(self, s: Session, nowMin: float) -> float:
        age = nowMin - s.startMin
        if age < 0 or age >= s.durMin:
            return 0.0
        if age <= s.rampMin:
            return s.peakKW * (age / max(1.0, s.rampMin))
        if age <= s.rampMin + s.holdMin:
            return s.peakKW
        ta = age - (s.rampMin + s.holdMin)
        t01 = clamp(ta / max(1.0, s.taperMin), 0.0, 1.0)
        ease = 1.0 - (t01 * t01)
        return s.peakKW * clamp(ease, 0.0, 1.0)

    def _new_session(self, nowMin: float) -> Session:
        mode = self.cfg.mode
        v = self._pick_vehicle()

        vtype = v["type"]
        peak = float(v["peakKW"])
        dur = float(v["baseDurMin"]) * rand(0.90, 1.15)

        if mode == "degen" and vtype == "car" and random.random() < 0.10:
            peak = rand(32.0, 40.0)

        ramp = rand(5, 10) if vtype == "car" else rand(2, 6)
        hold = dur * (rand(0.20, 0.35) if vtype == "car" else rand(0.45, 0.65))
        taper = max(8.0, dur - ramp - hold)

        s = Session(
            id=self.sessionId,
            vtype=vtype,
            startMin=nowMin,
            durMin=dur,
            peakKW=peak,
            rampMin=ramp,
            holdMin=hold,
            taperMin=taper,
        )
        self.sessionId += 1
        return s

    def _update_arrival_nudge(self):
        targetKWh = max(0.1, self.cfg.targetMWh) * 1000.0
        elapsedFrac = clamp(self.simMin / DAY_MIN, 0.0, 1.0)
        expectedSoFar = targetKWh * elapsedFrac
        err = expectedSoFar - self.energyKWh
        # more aggressive feedback but bounded
        self.arrivalNudge = clamp(1.0 + (err / max(1.0, targetKWh)) * 4.0, 0.5, 3.5)

    def restart(self):
        self.simMin = 0.0
        self.simSec = 0
        self.energyKWh = 0.0
        self.sessions = []
        self.sessionId = 1
        self.series_t = []
        self.series_pkw = []
        self.series_oi = []
        self.tape = []
        self.arrivalNudge = 1.0
        # IMPORTANT: do NOT reset lifetimeRevenueUSD here
        self._recompute_arrival_scale()

    def step(self):
        if self.paused:
            return

        # fixed sim step -> no jitter
        mult = max(0.05, float(self.cfg.simMinPerStep))  # minutes per tick
        cap = max(1, int(self.cfg.capacity))
        mode = self.cfg.mode

        t01 = (self.simMin % DAY_MIN) / DAY_MIN
        self._update_arrival_nudge()

        lam = (
            self._base_arrival_rate_per_min(mode, t01)
            * self.arrivalScale
            * self.arrivalNudge
            * mult
        )
        if mode == "degen" and random.random() < 0.05 * mult:
            lam *= rand(1.6, 2.8)

        arrivals = self._sample_poisson(lam)

        for _ in range(arrivals):
            if len(self.sessions) >= cap:
                break
            s = self._new_session(self.simMin)
            self.sessions.append(s)
            self._push_tape("in", f"+ Plugged: #{s.id} {s.vtype.upper()} @ {s.peakKW:.0f}kW")

            totalKW = sum(self._power_at(s, self.simMin) for s in self.sessions)
            # kW * hours = kWh; mult / 60 is hours
            delta_kwh = totalKW * (mult / 60.0)
            self.energyKWh += delta_kwh
            # revenue = margin * energy
            delta_revenue = delta_kwh * MARGIN_PER_KWH
            self.lifetimeRevenueUSD += delta_revenue


        kept: List[Session] = []
        for s in self.sessions:
            done = (self.simMin - s.startMin) >= s.durMin
            if done:
                self._push_tape("out", f"- Unplugged: #{s.id} {s.vtype.upper()}")
            else:
                kept.append(s)
        self.sessions = kept

        self.simMin += mult
        self.simSec = int(self.simMin * 60)

        if self.simMin >= DAY_MIN:
            self.restart()
            return

        self.series_t.append(self.simMin)
        self.series_pkw.append(totalKW)
        self.series_oi.append(float(len(self.sessions)))

        if len(self.series_t) > self.max_points:
            self.series_t.pop(0)
            self.series_pkw.pop(0)
            self.series_oi.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        totalKW = sum(self._power_at(s, self.simMin) for s in self.sessions)
        ...
        targetKWh = max(0.1, self.cfg.targetMWh) * 1000.0

        # approximate revenue over next 60s at current power
        # power (kW) * 1h = kWh, so 60s is 1/60th of an hour
        last60s_rev = totalKW * MARGIN_PER_KWH / 60.0
        last60s_protocol = last60s_rev * PROTOCOL_FEE_SHARE

        lifetime_protocol = self.lifetimeRevenueUSD * PROTOCOL_FEE_SHARE

        return {
            "meta": {"mark": "Mark1", "theme": THEME},
            "config": self.cfg.model_dump(),
            "now": {
                "simMin": self.simMin,
                "simSec": self.simSec,
                "simTime": fmt_time(self.simSec),
                "activeVehicles": len(self.sessions),
                "totalKW": totalKW,
                "energyKWh": self.energyKWh,
                "targetKWh": targetKWh,
                "arrivalsPerMin": self._base_arrival_rate_per_min(
                    self.cfg.mode, (self.simMin % DAY_MIN) / DAY_MIN
                )
                * self.arrivalScale
                * self.arrivalNudge,
                "corr": rho,
            },
            "series": {
                "t": self.series_t,
                "pkw": self.series_pkw,
                "oi": self.series_oi,
            },
            "top": top,
            "tape": self.tape,
            "revenue": {
                "lifetimeUSD": self.lifetimeRevenueUSD,
                "lifetimeProtocolUSD": lifetime_protocol,
                "last60sUSD": last60s_rev,
                "last60sProtocolUSD": last60s_protocol,
            },
        }



engine = Engine()


def save_snapshot_to_store():
    try:
        snap = engine.snapshot()
        redis_client.set(SNAPSHOT_KEY, json.dumps(snap))
    except Exception as e:
        print("Redis save error:", e)


# -----------------------
# FastAPI app + CORS
# -----------------------

origins = ["*"]

app = FastAPI(title="DeGen Energy Terminal API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "mark": "Mark1"}


@app.get("/api/state")
def api_state():
    # try Redis first
    try:
        raw = redis_client.get(SNAPSHOT_KEY)
        if raw:
            return json.loads(raw)
    except Exception as e:
        print("Redis read error:", e)

    # fallback to in-memory snapshot
    with engine.lock:
        return engine.snapshot()


@app.post("/api/config")
def api_config(cfg: Config):
    with engine.lock:
        engine.cfg = cfg
        engine._recompute_arrival_scale()
        return {"ok": True, "config": engine.cfg.model_dump()}


@app.post("/api/restart")
def api_restart():
    with engine.lock:
        engine.restart()
        return {"ok": True}


@app.post("/api/pause")
def api_pause(pause: bool):
    with engine.lock:
        engine.paused = pause
        return {"ok": True, "paused": engine.paused}


def _run_loop():
    while True:
        with engine.lock:
            engine.step()
            save_snapshot_to_store()
            hz = max(1.0, float(engine.cfg.engineHz))
        time.sleep(1.0 / hz)


threading.Thread(target=_run_loop, daemon=True).start()
