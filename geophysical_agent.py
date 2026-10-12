import os
import struct
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="AI Geophysical Analyst API",
    description="API REST pour les moteurs SEG-Y, Well et Geo (Projet AGENT Z)",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "agent": "AI Geophysical Analyst",
        "mission": "Analyse et interprétation géophysique et pétrophysique (Bassin Ivoirien)",
        "engines": ["SEG-Y Engine", "Well Engine", "Geo Engine"],
        "status": "Operational"
    }

# --- SEG-Y ENGINE ENDPOINT ---
@app.post("/run-segy-analysis")
def run_segy_analysis(num_traces: int = 10, num_samples: int = 500):
    filename = "api_test_synthetic.sgy"
    try:
        # Création et parsing SEG-Y
        with open(filename, "wb") as f:
            text_header = "C 1 CLIENT: AGENT Z EXPLORATION                             C 2 AREA: IVORIAN BASIN - API TEST                   "
            f.write(text_header.ljust(3200, " ").encode('ascii', errors='ignore'))
            
            bin_header = bytearray(400)
            struct.pack_into(">I", bin_header, 0, 1)
            struct.pack_into(">H", bin_header, 16, 2000)
            struct.pack_into(">H", bin_header, 20, num_samples)
            struct.pack_into(">H", bin_header, 24, 5)
            f.write(bin_header)
            
            for i in range(num_traces):
                t_header = bytearray(240)
                struct.pack_into(">I", t_header, 0, i + 1)
                struct.pack_into(">I", t_header, 8, 100 + i)
                struct.pack_into(">i", t_header, 181, 500000 + i * 50)
                struct.pack_into(">i", t_header, 185, 1200000)
                f.write(t_header)
                
                t = np.arange(num_samples) * 0.002
                f_peak = 35.0
                ricker = (1.0 - 2.0 * (np.pi**2) * (f_peak**2) * ((t - 0.25)**2)) * np.exp(-(np.pi**2) * (f_peak**2) * ((t - 0.25)**2))
                trace_data = ricker + np.random.normal(0, 0.02, num_samples)
                f.write(struct.pack(f">{num_samples}f", *trace_data))

        # Lecture & Analyse
        with open(filename, "rb") as f:
            f.read(3200)
            bin_bytes = f.read(400)
            sample_interval = struct.unpack_from(">H", bin_bytes, 16)[0] / 1000.0
            
            traces_meta = []
            for i in range(num_traces):
                t_head = f.read(240)
                t_data = struct.unpack(f">{num_samples}f", f.read(num_samples * 4))
                rms = float(np.sqrt(np.mean(np.array(t_data)**2)))
                traces_meta.append({"trace": i + 1, "rms": rms})

        return {
            "status": "success",
            "engine": "SEG-Y Engine",
            "sample_interval_ms": sample_interval,
            "num_traces": num_traces,
            "traces_summary": traces_meta[:5] # Aperçu des 5 premières
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WELL ENGINE ENDPOINT ---
@app.post("/run-well-analysis")
def run_well_analysis(num_points: int = 250):
    filename = "api_test_well.las"
    try:
        depths = np.linspace(2000.0, 2500.0, num_points)
        np.random.seed(42)
        gr = np.clip(70 + 40 * np.sin(depths / 10.0) + np.random.normal(0, 8, num_points), 20, 150)
        rhob = np.clip(2.65 - 0.3 * (gr - 30) / 120.0 + np.random.normal(0, 0.02, num_points), 2.1, 2.8)
        
        df = pd.DataFrame({"DEPT": depths, "GR": gr, "RHOB": rhob})
        
        gr_min, gr_max = df["GR"].quantile(0.05), df["GR"].quantile(0.95)
        df["VSH"] = ((df["GR"] - gr_min) / (gr_max - gr_min)).clip(0.0, 1.0)
        df["PHIE"] = ((2.65 - df["RHOB"]) / (2.65 - 1.0) - df["VSH"] * 0.15).clip(0.01, 0.40)
        
        reservoir = df[(df["PHIE"] > 0.15) & (df["VSH"] < 0.30)]
        
        return {
            "status": "success",
            "engine": "Well Engine",
            "mean_vsh": float(df["VSH"].mean()),
            "mean_phie": float(df["PHIE"].mean()),
            "reservoir_intervals_count": len(reservoir),
            "reservoir_thickness_m": float(len(reservoir) * 0.2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- GEO ENGINE ENDPOINT ---
@app.post("/run-geo-analysis")
def run_geo_analysis(num_points: int = 150):
    try:
        np.random.seed(42)
        x = np.random.uniform(500000, 520000, num_points)
        y = np.random.uniform(1200000, 1220000, num_points)
        r = np.sqrt((x - 510000)**2 + (y - 1210000)**2)
        twt = 2200.0 - 350.0 * np.exp(-(r**2) / (2 * (15000**2))) + np.random.normal(0, 10, num_points)
        
        v0, k = 1600.0, 0.8
        twt_sec = twt / 1000.0
        depth_z = (twt_sec / 2.0) * (v0 + 0.5 * k * twt_sec)
        
        return {
            "status": "success",
            "engine": "Geo Engine",
            "min_depth_m": float(depth_z.min()),
            "max_depth_m": float(depth_z.max()),
            "structure_type": "Anticlinal",
            "points_processed": num_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
