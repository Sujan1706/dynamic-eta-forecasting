# Frontend Architecture & Usage Guide

## Overview
The frontend for the **Dynamic Train ETA Forecasting Engine** is built using **Next.js (App Router), React 19, and TypeScript**. It acts as a client interface communicating directly with the FastAPI backend without duplicating any business or prediction logic.

---

## Directory Structure
```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx             # Root layout with top Navbar & footer
│   │   ├── page.tsx               # 1. Control Room Dashboard
│   │   ├── trains/[id]/page.tsx   # 2. Train Details (Multi-station ETA & Chained Segments)
│   │   ├── passenger/page.tsx     # 3. Passenger ETA Lookup
│   │   ├── station/page.tsx       # 4. Station Arrivals Board
│   │   └── globals.css            # Tailwind CSS styles
│   ├── components/
│   │   └── Navbar.tsx             # Main navigation and live FastAPI health badge
│   ├── lib/
│   │   └── api.ts                 # Reusable, typed FastAPI client & formatters
│   └── types/
│       └── api.ts                 # TypeScript interfaces matching backend Pydantic models
├── package.json
└── tsconfig.json
```

---

## Pages & Capabilities

### 1. Control Room Dashboard (`/`)
- **Fleet Monitor**: Live table displaying all managed trains (`GET /trains`), their running status (`RUNNING`, `HALTED`), instantaneous speed (`speed_kmh`), current station checkpoint, and current delay.
- **Disruption Simulator Panel**: Interactive form to inject operational events (`POST /simulate/event`) for `SIGNAL_HALT`, `CONGESTION`, `SPEED_RESTRICTION`, `UNSCHEDULED_HALT`, and `WEATHER`.
- **Live Event Response**: Injects events and immediately displays the updated train state, updated Baseline ETA, and updated XGBoost ML ETA.
- **Model Provenance**: Displays overall MAE/RMSE comparisons and performance by station horizon and disruption regime from `GET /model/metrics`.

### 2. Train Details (`/trains/[id]`)
- **Telemetry Overview**: Current station, delay, speed, and status for the specific train (`GET /train/{train_id}`).
- **Multi-Station ETA Table**: Chained arrival forecasts for every upcoming station on the route, showing Timetable Scheduled Time, Baseline Heuristic ETA, XGBoost ML ETA, and heuristic Confidence Intervals.
- **Segment Breakdown**: Step-by-step display of individual segment predictions chained to produce the final arrival time.
- **Route Timetable**: Full fixed timetable for the route.

### 3. Passenger ETA Lookup (`/passenger`)
- Inquire arrival forecasts for any train at any destination station (`GET /train/{train_id}/eta/{station_code}`).
- Pre-configured quick buttons for seeded trains (`12302 Howrah Rajdhani`, `12952 Mumbai Rajdhani`, `12028 Shatabdi`).
- Clear visual hierarchy highlighting Expected Arrival Time, Timetable Time, Delay Badge, and Confidence Interval.

### 4. Station Arrivals Board (`/station`)
- Real-time station arrivals board (`GET /station/{station_code}/arrivals?window_hours=...`).
- Shows approaching trains sorted primarily by XGBoost ML ETA.
- Station selector for major hubs (`NDLS`, `CNB`, `PRYJ`, `DDU`, `KOTA`, `MAS`, `HWH`, `MMCT`).
- Arrivals time window filter (2h, 4h, 8h, 24h, All).

---

## Running the Application

### 1. Start the FastAPI Backend
```bash
# From repository root
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Next.js Frontend
```bash
# Navigate to frontend/
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
