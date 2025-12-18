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

# =====================================================
# Constants
# =====================================================

DAY_MIN = 1440
TARGET_SESSIONS_PER_DAY = 6800

MARGIN_PER_KWH = 0.004
PROTOCOL_FEE_SHARE = 0.20

# =====================================================
# Vehicle definitions
# =====================================================

VEHICLE_TYPES = [
    {"type": "car", "peakKW": 40.0, "baseDurMin": 90.0, "prob": 0.01},
    {"type": "3w",  "peakKW": 4.0,  "baseDurMin": 60.0, "prob": 0.29},
    {"type": "2w",  "peakKW": 2.0,  "baseDurMin": 45.0, "prob": 0.70},
]

THEME = {
    "bgGrid": "rgba(255,255,255,0.10)",
    "powerLine": "rgba(0,255,200,0.95)",
    "oiLine": "rgba(255,215,0,0.95)",
}

# =====================================================
# Redis
# =====================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
SNAPSHOT_KEY = "degen:mark1:snapshot"

# =====================================================
# Helpers
# =====================================================

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def rand(a: float, b: float) -> float:
    return a + random.random() * (b - a)


def fmt_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds // 60) % 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def corr(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 10:
        return 0.0
    mx = sum(xs[-n:]) / n
    my = sum(ys[-n:]) / n
    num = sum((xs[-n+i] - mx) * (ys[-n+i] - my) for i in range(n))
    dx = sum((xs[-n+i] - mx) ** 2 for i in range(n))
    dy = sum((ys[-n+i] - my) ** 2 for i in range(n))
    return clamp(num / (math.sqrt(dx * dy) + 1e-6), -1.0, 1.0)

# =====================================================
# Models
# =====================================================

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
    targetMWh: float = 19.0
    mode: str = "office"
    capacity: int = 1000
    simMinPerStep: float = 0.25   # 15s sim
    engineHz: float = 4.0

# =====================================================
# Engine
# =====================================================

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = Config()

        self.paused = False
        self.simMin = 0.0
        self.simSec = 0

        self.energyKWh = 0.0
        self.lifetimeRevenueUSD = 0.0

        self.sessions: List[Session] = []
        self.sessionId = 1

        self.series_t: List[float] = []
        self.series_pkw: List[float] = []
        self.series_oi: List[float] = []
        self.max_points = 420

        self.tape: List[Dict[str, Any]] = []

        self.arrivalScale = 1.0
        self.arrivalNudge = 1.0
        self._recompute_arrival_scale()

    # -------------------------

    def _base_arrival_rate_per_min(self, mode: str, t01: float) -> float:
        if mode == "office":
            return 0.25 * math.exp(-((t01 - 0.55) / 0.18) ** 2)
        return 0.15

    def _recompute_arrival_scale(self):
        avg = sum(
            self._base_arrival_rate_per_min(self.cfg.mode, i / 240)
            for i in range(240)
        ) / 240
        self.arrivalScale = TARGET_SESSIONS_PER_DAY / max(1e-6, avg * DAY_MIN)

    def _sample_poisson(self, lam: float) -> int:
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def _power_at(self, s: Session, nowMin: float) -> float:
        age = nowMin - s.startMin
        if age <= 0 or age >= s.durMin:
            return 0.0
        if age < s.rampMin:
            return s.peakKW * age / s.rampMin
        if age < s.rampMin + s.holdMin:
            return s.peakKW
        t = (age - s.rampMin - s.holdMin) / s.taperMin
        return s.peakKW * (1 - t * t)

    def _new_session(self) -> Session:
        v = random.choices(VEHICLE_TYPES, weights=[v["prob"] for v in VEHICLE_TYPES])[0]
        dur = v["baseDurMin"] * rand(0.9, 1.1)
        ramp = rand(3, 8)
        hold = dur * 0.5
        taper = max(5, dur - ramp - hold)

        s = Session(
            id=self.sessionId,
            vtype=v["type"],
            startMin=self.simMin,
            durMin=dur,
            peakKW=v["peakKW"],
            rampMin=ramp,
            holdMin=hold,
            taperMin=taper,
        )
        self.sessionId += 1
        return s

    # -------------------------

    def step(self):
        if self.paused:
            return

        mult = self.cfg.simMinPerStep
        t01 = (self.simMin % DAY_MIN) / DAY_MIN

        lam = (
            self._base_arrival_rate_per_min(self.cfg.mode, t01)
            * self.arrivalScale
            * mult
        )

        arrivals = self._sample_poisson(lam)

        for _ in range(arrivals):
            if len(self.sessions) < self.cfg.capacity:
                self.sessions.append(self._new_session())

        # 🔴 FIX: energy & revenue ALWAYS integrate
        totalKW = sum(self._power_at(s, self.simMin) for s in self.sessions)
        delta_kwh = totalKW * (mult / 60.0)

        self.energyKWh += delta_kwh
        self.lifetimeRevenueUSD += delta_kwh * MARGIN_PER_KWH

        self.sessions = [
            s for s in self.sessions
            if (self.simMin - s.startMin) < s.durMin
        ]

        self.simMin += mult
        self.simSec = int(self.simMin * 60)

        if self.simMin >= DAY_MIN:
            self.simMin = 0
            self.simSec = 0
            self.energyKWh = 0.0
            self.sessions.clear()

        self.series_t.append(self.simMin)
        self.series_pkw.append(totalKW)
        self.series_oi.append(len(self.sessions))

        if len(self.series_t) > self.max_points:
            self.series_t.pop(0)
            self.series_pkw.pop(0)
            self.series_oi.pop(0)

    # -------------------------

    def snapshot(self) -> Dict[str, Any]:
        totalKW = sum(self._power_at(s, self.simMin) for s in self.sessions)
        rho = corr(self.series_pkw, self.series_oi)

        targetKWh = self.cfg.targetMWh * 1000.0

        arrivals_per_min = (
            self._base_arrival_rate_per_min(
                self.cfg.mode,
                (self.simMin % DAY_MIN) / DAY_MIN
            )
            * self.arrivalScale
            * self.arrivalNudge
        )

        # revenue
        last60s_rev = totalKW * MARGIN_PER_KWH / 60.0
        lifetime_protocol = self.lifetimeRevenueUSD * PROTOCOL_FEE_SHARE

        # top consumers
        top = sorted(
            [
                {
                    "id": s.id,
                    "vtype": s.vtype,
                    "p": self._power_at(s, self.simMin),
                }
                for s in self.sessions
            ],
            key=lambda x: x["p"],
            reverse=True,
        )[:10]

        return {
            "meta": {
                "mark": "Mark1",
                "theme": THEME,
            },
            "config": self.cfg.model_dump(),
            "now": {
                "simMin": self.simMin,
                "simSec": self.simSec,
                "simTime": fmt_time(self.simSec),
                "activeVehicles": len(self.sessions),
                "totalKW": totalKW,
                "energyKWh": self.energyKWh,
                "targetKWh": targetKWh,
                "arrivalsPerMin": arrivals_per_min,
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
                "last60sUSD": last60s_rev,
                "last60sProtocolUSD": last60s_rev * PROTOCOL_FEE_SHARE,
                "lifetimeUSD": self.lifetimeRevenueUSD,
                "lifetimeProtocolUSD": lifetime_protocol,
            },
        }


# =====================================================
# App
# =====================================================

engine = Engine()

def load_engine_from_redis():
    try:
        raw = redis_client.get(SNAPSHOT_KEY)
        if not raw:
            return
        snap = json.loads(raw)

        # restore only what you need for continuity
        now = snap.get("now", {})
        engine.simMin = float(now.get("simMin", 0.0))
        engine.simSec = int(now.get("simSec", 0))
        engine.energyKWh = float(now.get("energyKWh", 0.0))

        rev = snap.get("revenue", {})
        engine.lifetimeRevenueUSD = float(rev.get("lifetimeUSD", 0.0))

        # restore series so charts don't restart empty
        series = snap.get("series", {})
        engine.series_t = list(series.get("t", []))
        engine.series_pkw = list(series.get("pkw", []))
        engine.series_oi = list(series.get("oi", []))

        # restore tape if you want
        engine.tape = list(snap.get("tape", []))

        print("Loaded state from Redis.")
    except Exception as e:
        print("Redis load error:", e)

load_engine_from_redis()


app = FastAPI(title="DeGen Energy Terminal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "mark": "Mark1"}

@app.get("/api/state")
def api_state():
    snap = engine.snapshot()
    redis_client.set(SNAPSHOT_KEY, json.dumps(snap))
    return snap

@app.post("/api/pause")
def api_pause(pause: bool):
    engine.paused = pause
    return {"ok": True, "paused": engine.paused}

@app.post("/api/restart")
def api_restart():
    engine.simMin = 0
    engine.simSec = 0
    engine.energyKWh = 0
    engine.sessions.clear()
    return {"ok": True}

@app.post("/api/config")
def api_config(cfg: Config):
    engine.cfg = cfg
    engine._recompute_arrival_scale()
    return {"ok": True, "config": cfg.model_dump()}


@app.get("/debug/redis")
def debug_redis():
    ttl = redis_client.ttl(SNAPSHOT_KEY)
    return {"ttl": ttl}


def loop():
    while True:
        with engine.lock:
            engine.step()
        time.sleep(1.0 / engine.cfg.engineHz)

threading.Thread(target=loop, daemon=True).start()
