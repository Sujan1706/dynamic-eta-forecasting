"use client";

import { useState, useEffect, useCallback, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { api, formatTime } from "@/lib/api";
import { StationArrivalsResponse, StationArrivalItem } from "@/types/api";
import UncertaintyRangeBar from "@/components/UncertaintyRangeBar";

function StationArrivalsBoardContent() {
  const searchParams = useSearchParams();
  const initialStationParam = searchParams.get("code") || "CNB";

  const [stationCode, setStationCode] = useState<string>(initialStationParam.toUpperCase());
  const [windowHours, setWindowHours] = useState<number | undefined>(24);
  const [data, setData] = useState<StationArrivalsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Live Digital Station Clock
  const [clockTime, setClockTime] = useState<string>("");

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setClockTime(
        now.toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };
    updateClock();
    const timer = setInterval(updateClock, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch station arrivals from backend
  const fetchArrivals = useCallback(
    async (code: string, hours?: number, isSilent = false) => {
      if (!code.trim()) return;

      if (!isSilent) setLoading(true);
      else setRefreshing(true);
      setError(null);

      try {
        const cleanCode = code.trim().toUpperCase();
        const res = await api.getStationArrivals(cleanCode, hours);
        setData(res);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Failed to load station arrivals";
        setError(msg);
        setData(null);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    []
  );

  // Initial load on mount or station change
  useEffect(() => {
    if (stationCode) {
      fetchArrivals(stationCode, windowHours);
    }
  }, [stationCode, windowHours, fetchArrivals]);

  // Periodic Auto-refresh every 10 seconds for live board feel
  useEffect(() => {
    const interval = setInterval(() => {
      if (stationCode) {
        fetchArrivals(stationCode, windowHours, true);
      }
    }, 10000);
    return () => clearInterval(interval);
  }, [stationCode, windowHours, fetchArrivals]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchArrivals(stationCode, windowHours);
  };

  const majorStations = [
    { code: "CNB", name: "Kanpur Central" },
    { code: "PRYJ", name: "Prayagraj Junction" },
    { code: "DDU", name: "Pt. DD Upadhyaya" },
    { code: "NDLS", name: "New Delhi" },
    { code: "GAYA", name: "Gaya Junction" },
    { code: "DHN", name: "Dhanbad Junction" },
    { code: "HWH", name: "Howrah Terminus" },
    { code: "KOTA", name: "Kota Junction" },
    { code: "MMCT", name: "Mumbai Central" },
    { code: "SBC", name: "KSR Bengaluru" },
  ];

  // Derived telemetry metrics
  const delayedArrivalsCount = useMemo(() => {
    if (!data?.arrivals) return 0;
    return data.arrivals.filter((a) => (a.current_delay_minutes ?? 0) > 2.0).length;
  }, [data]);

  const onTimeArrivalsCount = useMemo(() => {
    if (!data?.arrivals) return 0;
    return data.arrivals.length - delayedArrivalsCount;
  }, [data, delayedArrivalsCount]);

  return (
    <div className="w-full max-w-[1560px] mx-auto space-y-4 pb-12">
      {/* Top Breadcrumb & Quick Actions Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5 pb-1 border-b border-[#D9DEE3]">
        <div className="flex items-center space-x-2">
          <Link
            href="/"
            className="text-xs font-semibold text-[#66717A] hover:text-[#172026] flex items-center transition-colors"
          >
            &larr; Control Room
          </Link>
          <span className="text-[#8A949C]">&bull;</span>
          <h1 className="text-sm font-bold text-[#172026]">Digital Station Arrivals Board</h1>
        </div>

        <div className="flex items-center space-x-2">
          <Link
            href="/passenger"
            className="inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-md border border-[#D9DEE3] bg-[#FFFFFF] hover:bg-[#F0F3F5] text-[#172026] shadow-2xs transition-colors"
          >
            Passenger Portal &rarr;
          </Link>
          <button
            type="button"
            onClick={() => fetchArrivals(stationCode, windowHours)}
            disabled={loading || refreshing}
            className="inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-md bg-[#172026] hover:bg-[#2563A8] text-white shadow-2xs transition-colors disabled:opacity-50 cursor-pointer"
          >
            <span className={`mr-1.5 ${refreshing ? "animate-spin" : ""}`}>
              ↻
            </span>
            {refreshing ? "Syncing..." : "Refresh Board"}
          </button>
        </div>
      </div>

      {/* Station Selector & Search Controls Toolbar */}
      <div className="bg-[#FFFFFF] p-3.5 sm:p-4 rounded-lg border border-[#D9DEE3] shadow-2xs space-y-3">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
          {/* Direct Search Form */}
          <form onSubmit={handleSubmit} className="flex flex-wrap items-center gap-2">
            <div className="flex items-center space-x-1.5">
              <label className="text-[11px] font-bold uppercase tracking-wider text-[#66717A] whitespace-nowrap">
                Station Code:
              </label>
              <input
                type="text"
                value={stationCode}
                onChange={(e) => setStationCode(e.target.value.toUpperCase())}
                placeholder="e.g. CNB"
                maxLength={8}
                className="w-28 sm:w-32 px-2.5 py-1.5 text-xs font-mono font-bold uppercase bg-[#F8FAFB] border border-[#D9DEE3] rounded-md focus:bg-[#FFFFFF] focus:border-[#2563A8] focus:outline-none"
              />
            </div>

            <div className="flex items-center space-x-1.5">
              <label className="text-[11px] font-bold uppercase tracking-wider text-[#66717A] whitespace-nowrap">
                Window:
              </label>
              <select
                value={windowHours || 24}
                onChange={(e) => setWindowHours(Number(e.target.value))}
                className="px-2.5 py-1.5 text-xs bg-[#F8FAFB] border border-[#D9DEE3] rounded-md focus:bg-[#FFFFFF] focus:border-[#2563A8] focus:outline-none font-mono"
              >
                <option value={2}>Next 2 Hours</option>
                <option value={4}>Next 4 Hours</option>
                <option value={8}>Next 8 Hours</option>
                <option value={24}>Next 24 Hours</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="px-3.5 py-1.5 bg-[#172026] hover:bg-[#2563A8] text-white font-semibold text-xs rounded-md shadow-2xs transition-colors cursor-pointer disabled:opacity-50"
            >
              {loading ? "Loading..." : "Load Board"}
            </button>
          </form>

          {/* Quick Station Select Buttons */}
          <div className="flex flex-wrap items-center gap-1.5 pt-2 lg:pt-0 border-t lg:border-t-0 border-[#D9DEE3]/70">
            <span className="text-[10px] text-[#8A949C] uppercase font-bold mr-1">
              Major Junctions:
            </span>
            {majorStations.map((stn) => {
              const isSelected = stationCode === stn.code;
              return (
                <button
                  key={stn.code}
                  type="button"
                  onClick={() => {
                    setStationCode(stn.code);
                    fetchArrivals(stn.code, windowHours);
                  }}
                  className={`px-2 py-0.5 rounded text-xs font-mono transition-all cursor-pointer ${
                    isSelected
                      ? "bg-[#172026] text-white border border-[#172026] font-bold shadow-2xs"
                      : "bg-[#F0F3F5] hover:bg-[#E2E8F0] text-[#66717A] border border-[#D9DEE3]"
                  }`}
                  title={stn.name}
                >
                  {stn.code}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-3 bg-rose-50 border border-rose-200 text-[#D64545] rounded-lg text-xs space-y-1">
          <div className="font-bold">Error Loading Station Arrivals</div>
          <div>{error}</div>
        </div>
      )}

      {/* ======================================================== */}
      {/* MAIN STATION BOARD CARD (OPERATIONAL CENTERPIECE) */}
      {/* ======================================================== */}
      <div className="bg-[#FFFFFF] border border-[#D9DEE3] rounded-lg shadow-2xs overflow-hidden">
        {/* Operational Board Header */}
        <div className="p-4 sm:p-5 border-b border-[#D9DEE3] bg-[#FFFFFF] flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Station Identity */}
          <div className="flex items-center space-x-3.5">
            <div className="px-3 py-1.5 rounded bg-[#172026] text-white font-mono font-bold text-base tracking-wider shadow-2xs">
              {stationCode}
            </div>
            <div>
              <h2 className="text-lg sm:text-xl font-bold text-[#172026] tracking-tight">
                {data?.station_name ? `${data.station_name} Arrivals` : `${stationCode} Junction Arrivals`}
              </h2>
              <div className="text-xs text-[#66717A] mt-0.5 flex flex-wrap items-center gap-2">
                <span>Indian Railways Zone Operations</span>
                <span>&bull;</span>
                <span className="font-semibold text-[#168A55] flex items-center space-x-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#168A55] animate-pulse" />
                  <span>XGBoost Dynamic Chained Forecasting Active</span>
                </span>
              </div>
            </div>
          </div>

          {/* Operational Telemetry KPI Blocks */}
          <div className="flex flex-wrap items-center gap-3 sm:gap-4 self-start md:self-auto">
            {/* 1. Digital Clock */}
            <div className="bg-[#F8FAFB] px-3 py-1.5 rounded border border-[#D9DEE3] text-right">
              <div className="text-[9px] text-[#8A949C] uppercase tracking-wider font-bold">
                Station Clock (IST)
              </div>
              <div className="text-sm font-bold text-[#172026] font-mono">
                {clockTime || "--:--:--"}
              </div>
            </div>

            {/* 2. Total Inbound */}
            <div className="bg-[#F8FAFB] px-3 py-1.5 rounded border border-[#D9DEE3] text-right">
              <div className="text-[9px] text-[#8A949C] uppercase tracking-wider font-bold">
                Approaching
              </div>
              <div className="text-sm font-bold text-[#2563A8] font-mono">
                {data?.total_arrivals ?? 0} trains
              </div>
            </div>

            {/* 3. On Time vs Delayed */}
            <div className="bg-[#F8FAFB] px-3 py-1.5 rounded border border-[#D9DEE3] text-right">
              <div className="text-[9px] text-[#8A949C] uppercase tracking-wider font-bold">
                Status Split
              </div>
              <div className="text-sm font-bold font-mono">
                <span className="text-[#168A55]">{onTimeArrivalsCount} On-Time</span>
                {delayedArrivalsCount > 0 && (
                  <span className="text-[#D64545] ml-1">({delayedArrivalsCount} Late)</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Board Sub-header / Observation Summary */}
        <div className="px-4 py-2 bg-[#F8FAFB] border-b border-[#D9DEE3] flex flex-wrap items-center justify-between text-xs text-[#66717A]">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-[#172026]">
              Operational Inbound Queue
            </span>
            <span className="text-[#8A949C]">&bull;</span>
            <span>Forecasts update automatically as trains report block telemetry</span>
          </div>
          <div className="font-mono text-[11px] text-[#8A949C]">
            Sync: {formatTime(data?.current_timestamp)} &bull; Horizon: {windowHours ? `${windowHours}h` : "24h"}
          </div>
        </div>

        {/* Board Arrivals Content */}
        {loading && !data ? (
          <div className="p-16 text-center text-[#66717A] space-y-2">
            <div className="text-2xl animate-spin">↻</div>
            <div className="text-xs font-semibold uppercase tracking-wider">
              Synchronizing Station Inbound Feed...
            </div>
          </div>
        ) : !data || data.arrivals.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <div className="text-3xl text-[#8A949C] font-mono">---</div>
            <div className="text-sm font-bold text-[#172026] uppercase">
              No trains approaching station {stationCode} within {windowHours || 24} hours
            </div>
            <p className="text-xs text-[#66717A] max-w-md mx-auto">
              No active trains were found approaching station <strong>{stationCode}</strong> in the selected lookahead window. Select another major junction above to view active traffic.
            </p>
            <div className="pt-2 flex flex-wrap justify-center gap-1.5">
              {["CNB", "PRYJ", "DDU", "NDLS", "HWH"].map((code) => (
                <button
                  key={code}
                  type="button"
                  onClick={() => {
                    setStationCode(code);
                    fetchArrivals(code, windowHours);
                  }}
                  className="px-2.5 py-1 bg-[#F0F3F5] hover:bg-[#E2E8F0] border border-[#D9DEE3] rounded text-xs font-mono font-bold text-[#172026] cursor-pointer"
                >
                  View {code} Board
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left border-collapse text-xs sm:text-[13px]">
              <thead>
                <tr className="bg-[#FFFFFF] text-[#66717A] text-[11px] font-semibold uppercase tracking-wider border-b border-[#D9DEE3]">
                  <th className="py-2.5 px-3.5 pl-4">Train</th>
                  <th className="py-2.5 px-3">Corridor</th>
                  <th className="py-2.5 px-3">Current Position</th>
                  <th className="py-2.5 px-3 font-mono">Timetable</th>
                  <th className="py-2.5 px-3 font-mono">Baseline</th>
                  <th className="py-2.5 px-3 min-w-[260px] sm:min-w-[300px]">
                    Expected Arrival (ML) &amp; Uncertainty Range
                  </th>
                  <th className="py-2.5 px-3 text-center">In</th>
                  <th className="py-2.5 px-3 text-center">Status / Delay</th>
                  <th className="py-2.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9DEE3]/70 bg-[#FFFFFF]">
                {data.arrivals.map((item: StationArrivalItem) => {
                  const delayMin = item.current_delay_minutes ?? 0;
                  const isCritical = delayMin > 15.0;
                  const isWarning = delayMin > 3.0 && delayMin <= 15.0;

                  // Severity stripe on left edge
                  const leftStripeClass = isCritical
                    ? "border-l-4 border-l-[#D64545]"
                    : isWarning
                    ? "border-l-4 border-l-[#B77900]"
                    : "border-l-4 border-l-[#168A55]";

                  return (
                    <tr
                      key={item.train_id}
                      className={`${leftStripeClass} hover:bg-[#F8FAFB] transition-colors`}
                    >
                      {/* 1. Train Number & Name */}
                      <td className="py-3 px-3.5 pl-3 whitespace-nowrap">
                        <div className="flex items-center space-x-1.5">
                          <span className="font-mono text-xs sm:text-[13px] font-bold text-[#172026]">
                            {item.train_number}
                          </span>
                          <span className="px-1 py-0.2 rounded text-[9px] font-mono font-medium bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3] uppercase">
                            {item.train_type}
                          </span>
                        </div>
                        <div className="text-[11px] text-[#66717A] truncate max-w-[200px] mt-0.5">
                          {item.train_name}
                        </div>
                      </td>

                      {/* 2. Route Corridor */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div className="font-mono text-xs font-semibold text-[#172026]">
                          {item.origin_station_code || "--"} &rarr; {item.destination_station_code || "--"}
                        </div>
                        <div className="text-[10px] text-[#8A949C]">
                          {item.segments_ahead} stop(s) ahead
                        </div>
                      </td>

                      {/* 3. Current Position */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div className="font-mono text-xs font-medium text-[#172026]">
                          At {item.current_station || "Origin"}
                        </div>
                        <div className="text-[10px] text-[#8A949C] font-mono">
                          {(item.distance_to_go_km ?? 0).toFixed(0)} km to {stationCode}
                        </div>
                      </td>

                      {/* 4. Timetable Scheduled */}
                      <td className="py-3 px-3 whitespace-nowrap font-mono text-xs text-[#66717A]">
                        {formatTime(item.scheduled_eta)}
                      </td>

                      {/* 5. Baseline ETA */}
                      <td className="py-3 px-3 whitespace-nowrap font-mono text-xs text-[#66717A]">
                        {item.baseline_eta ? formatTime(item.baseline_eta) : "--:--"}
                      </td>

                      {/* 6. Expected ML ETA & Uncertainty Range Bar */}
                      <td className="py-3 px-3">
                        <UncertaintyRangeBar
                          mlEta={item.ml_eta || item.baseline_eta}
                          lowerBound={item.confidence_lower}
                          upperBound={item.confidence_upper}
                          marginMinutes={item.confidence_range?.margin_minutes}
                          segmentsAhead={item.segments_ahead}
                          variant="table"
                          theme="light"
                        />
                      </td>

                      {/* 7. Minutes to Arrival */}
                      <td className="py-3 px-3 whitespace-nowrap text-center font-mono">
                        <div className="font-bold text-xs text-[#172026]">
                          {item.minutes_to_arrival !== undefined &&
                          item.minutes_to_arrival !== null
                            ? `~${item.minutes_to_arrival.toFixed(0)} min`
                            : "--"}
                        </div>
                      </td>

                      {/* 8. Status & Delay */}
                      <td className="py-3 px-3 whitespace-nowrap text-center">
                        {delayMin > 2.0 ? (
                          <div className="inline-block px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-[#FDF2F2] text-[#D64545] border border-[#F5C2C2]">
                            +{delayMin.toFixed(0)} MIN
                          </div>
                        ) : (
                          <div className="inline-block px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-[#EBF7EE] text-[#168A55] border border-[#B4E2C1]">
                            ON TIME
                          </div>
                        )}
                      </td>

                      {/* 9. Action */}
                      <td className="py-3 px-4 text-right whitespace-nowrap">
                        <Link
                          href={`/trains/${item.train_number}`}
                          className="px-2.5 py-1 bg-[#F0F3F5] hover:bg-[#172026] hover:text-white border border-[#D9DEE3] text-[#172026] rounded text-[11px] font-semibold transition-all inline-block shadow-2xs"
                        >
                          Details &rarr;
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default function StationArrivalsBoardPage() {
  return (
    <Suspense
      fallback={
        <div className="py-16 text-center text-xs text-[#8A949C]">
          Loading Station Arrivals Board...
        </div>
      }
    >
      <StationArrivalsBoardContent />
    </Suspense>
  );
}
