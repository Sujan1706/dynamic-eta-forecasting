"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { api, formatTime } from "@/lib/api";
import {
  TrainListItem,
  EventInjectionResponse,
  DemoScenarioResponse,
  DemoActionResult,
} from "@/types/api";
import { TableRowSkeleton } from "@/components/LoadingSkeleton";

export default function ControlRoomDashboard() {
  const router = useRouter();

  // Data states
  const [trains, setTrains] = useState<TrainListItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [isUpdating, setIsUpdating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isStale, setIsStale] = useState<boolean>(false);
  const [staleError, setStaleError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  // Retain trains across poll hiccups
  const trainsRef = useRef<TrainListItem[]>([]);
  useEffect(() => {
    trainsRef.current = trains;
  }, [trains]);

  // View state: "table" | "cards"
  const [viewMode, setViewMode] = useState<"table" | "cards">("table");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<"ALL" | "RUNNING" | "HALTED">("ALL");

  // Configurable Polling
  const [pollingIntervalMs] = useState<number>(15000);
  const [isPollingActive] = useState<boolean>(true);
  const pollingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Simulation & Event Drawer State
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  // Disruption simulation form
  const [selectedTrain, setSelectedTrain] = useState<string>("");
  const [eventType, setEventType] = useState<string>("SIGNAL_HALT");
  const [delayMinutes, setDelayMinutes] = useState<number>(15);
  const [severity, setSeverity] = useState<string>("HIGH");
  const [locationNote, setLocationNote] = useState<string>("Interlocking signal aspect red");
  const [submittingEvent, setSubmittingEvent] = useState<boolean>(false);
  const [injectionResult, setInjectionResult] = useState<EventInjectionResponse | null>(null);
  const [injectionError, setInjectionError] = useState<string | null>(null);

  // Deterministic Hackathon Demo Mode states
  const [demoScenario, setDemoScenario] = useState<DemoScenarioResponse | null>(null);
  const [demoLoading, setDemoLoading] = useState<boolean>(false);
  const [demoActionActive, setDemoActionActive] = useState<string | null>(null);
  const [demoResult, setDemoResult] = useState<DemoActionResult | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);

  // Fetch all dashboard data from backend
  const fetchDashboardData = useCallback(async (isInitial = false) => {
    if (isInitial) {
      setLoading(true);
      setError(null);
    } else {
      setIsUpdating(true);
    }

    try {
      const [trainsRes, demoRes] = await Promise.all([
        api.getTrains(),
        api.getDemoScenario().catch(() => null),
      ]);

      setTrains(trainsRes.trains);
      if (demoRes) setDemoScenario(demoRes);
      setLastRefreshed(new Date());
      setIsStale(false);
      setStaleError(null);
      setError(null);

      if (trainsRes.trains.length > 0 && !selectedTrain) {
        setSelectedTrain(trainsRes.trains[0].train_number);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load dashboard data";
      if (isInitial && trainsRef.current.length === 0) {
        setError(msg);
      } else {
        setIsStale(true);
        setStaleError(msg);
      }
    } finally {
      if (isInitial) setLoading(false);
      else setIsUpdating(false);
    }
  }, [selectedTrain]);

  // Initial load
  useEffect(() => {
    fetchDashboardData(true);
  }, [fetchDashboardData]);

  // Background polling
  useEffect(() => {
    if (!isPollingActive || pollingIntervalMs <= 0) {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
      return;
    }

    pollingTimerRef.current = setInterval(() => {
      fetchDashboardData(false);
    }, pollingIntervalMs);

    return () => {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    };
  }, [isPollingActive, pollingIntervalMs, fetchDashboardData]);

  // Event Injection Submission
  const handleInjectEvent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedTrain) return;

    setSubmittingEvent(true);
    setInjectionError(null);
    setInjectionResult(null);

    try {
      const res = await api.simulateEvent({
        train_id: selectedTrain,
        event_type: eventType,
        delay_minutes: Number(delayMinutes),
        severity: severity,
        metadata: {
          location: locationNote,
          source: "ControlRoomDashboard",
          timestamp: new Date().toISOString(),
        },
      });

      setInjectionResult(res);
      await fetchDashboardData(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to inject disruption event";
      setInjectionError(msg);
    } finally {
      setSubmittingEvent(false);
    }
  };

  // Demo Actions
  const handleResetDemo = async () => {
    setDemoLoading(true);
    setDemoError(null);
    setDemoResult(null);
    try {
      const res = await api.resetDemoScenario();
      setDemoScenario(res);
      await fetchDashboardData(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to reset demo scenario";
      setDemoError(msg);
    } finally {
      setDemoLoading(false);
    }
  };

  const handleExecuteDemoAction = async (actionId: string) => {
    setDemoActionActive(actionId);
    setDemoError(null);
    try {
      const res = await api.executeDemoAction(actionId);
      setDemoResult(res);
      await fetchDashboardData(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : `Failed to execute demo action ${actionId}`;
      setDemoError(msg);
    } finally {
      setDemoActionActive(null);
    }
  };

  // Status Counts
  const totalTrainsCount = trains.length;
  const runningTrainsCount = trains.filter(
    (t) => (t.current_state?.status || "RUNNING") === "RUNNING"
  ).length;
  const haltedTrainsCount = trains.filter(
    (t) => (t.current_state?.status || "") === "HALTED"
  ).length;

  // Filtered Trains computation
  const filteredTrains = trains.filter((t) => {
    const q = searchQuery.toLowerCase().trim();
    const matchesSearch =
      !q ||
      t.train_number.toLowerCase().includes(q) ||
      t.name.toLowerCase().includes(q) ||
      (t.current_station && t.current_station.toLowerCase().includes(q)) ||
      (t.next_station && t.next_station.toLowerCase().includes(q));

    const status = t.current_state?.status || "RUNNING";
    const matchesStatus =
      statusFilter === "ALL" ||
      (statusFilter === "RUNNING" && status === "RUNNING") ||
      (statusFilter === "HALTED" && status === "HALTED");

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="w-full max-w-[1560px] mx-auto space-y-4 pb-12">
      {/* Non-blocking Stale Notification Banner */}
      {isStale && (
        <div className="px-4 py-2.5 bg-amber-50 border border-amber-200 rounded-lg text-xs text-[#B77900] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[#B77900] animate-ping" />
            <span>
              <strong>Telemetry Polling Lag:</strong> Displaying cached fleet telemetry ({staleError || "Connection retry in progress"}).
            </span>
          </div>
          <button
            type="button"
            onClick={() => fetchDashboardData(false)}
            className="font-semibold underline hover:text-amber-900 ml-3"
          >
            Retry Now
          </button>
        </div>
      )}

      {/* Main Operations Header & Filter Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-1">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#172026]">
            Active Fleet Operations
          </h1>
          <p className="text-xs sm:text-[13px] text-[#66717A] mt-0.5">
            Real-time telemetry, baseline heuristic ETAs, and chained XGBoost forecasts.
          </p>
        </div>

        {/* Status Filter Segmented Controls & View Switcher */}
        <div className="flex items-center space-x-2 self-start sm:self-auto shrink-0">
          <div className="flex items-center p-0.5 bg-[#F0F3F5] border border-[#D9DEE3] rounded-lg text-xs">
            <button
              type="button"
              onClick={() => setStatusFilter("ALL")}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center space-x-1.5 ${
                statusFilter === "ALL"
                  ? "bg-[#FFFFFF] text-[#172026] font-semibold shadow-2xs border border-[#D9DEE3]"
                  : "text-[#66717A] hover:text-[#172026] border border-transparent"
              }`}
            >
              <span>All</span>
              <span className="px-1.5 py-0.2 rounded bg-[#F0F3F5] text-[11px] font-mono font-bold text-[#66717A]">
                {totalTrainsCount}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setStatusFilter("RUNNING")}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center space-x-1.5 ${
                statusFilter === "RUNNING"
                  ? "bg-[#FFFFFF] text-[#172026] font-semibold shadow-2xs border border-[#D9DEE3]"
                  : "text-[#66717A] hover:text-[#172026] border border-transparent"
              }`}
            >
              <span>Running</span>
              <span className="px-1.5 py-0.2 rounded bg-[#EBF7EE] text-[11px] font-mono font-bold text-[#168A55]">
                {runningTrainsCount}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setStatusFilter("HALTED")}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center space-x-1.5 ${
                statusFilter === "HALTED"
                  ? "bg-[#FFFFFF] text-[#172026] font-semibold shadow-2xs border border-[#D9DEE3]"
                  : "text-[#66717A] hover:text-[#172026] border border-transparent"
              }`}
            >
              <span>Halted</span>
              <span className="px-1.5 py-0.2 rounded bg-[#FDF2F2] text-[11px] font-mono font-bold text-[#D64545]">
                {haltedTrainsCount}
              </span>
            </button>
          </div>

          {/* View Mode Toggle Button */}
          <div className="flex items-center p-0.5 bg-[#F0F3F5] border border-[#D9DEE3] rounded-lg">
            <button
              type="button"
              onClick={() => setViewMode("table")}
              title="Table View"
              className={`p-1.5 rounded-md transition-all ${
                viewMode === "table"
                  ? "bg-[#FFFFFF] text-[#172026] shadow-2xs border border-[#D9DEE3]"
                  : "text-[#66717A] hover:text-[#172026] border border-transparent"
              }`}
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <button
              type="button"
              onClick={() => setViewMode("cards")}
              title="Grid View"
              className={`p-1.5 rounded-md transition-all ${
                viewMode === "cards"
                  ? "bg-[#FFFFFF] text-[#172026] shadow-2xs border border-[#D9DEE3]"
                  : "text-[#66717A] hover:text-[#172026] border border-transparent"
              }`}
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* ETA Prediction Tiers Legend Bar */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-4 py-2 px-3 bg-[#FFFFFF] border border-[#D9DEE3] rounded-lg text-xs text-[#66717A]">
        <span className="text-[10px] font-bold uppercase tracking-wider text-[#8A949C]">
          ETA PREDICTION TIERS:
        </span>
        <div className="flex items-center space-x-1.5">
          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
            SCHED
          </span>
          <span className="text-[#172026]">Fixed Timetable</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#F5F7F8] text-[#172026] border border-[#D9DEE3]">
            BASELINE
          </span>
          <span className="text-[#172026]">Speed heuristic</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#EBF3FC] text-[#2563A8] border border-[#BFDBFE]">
            ML ETA
          </span>
          <span className="text-[#2563A8] font-medium">XGBoost Forecast + Uncertainty Range</span>
        </div>
      </div>

      {/* Operational Table Container */}
      <div className="bg-[#FFFFFF] border border-[#D9DEE3] rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.03)] overflow-hidden">
        {/* Search & Actions Bar inside Table Card */}
        <div className="p-3 sm:p-3.5 border-b border-[#D9DEE3] flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5 bg-[#FFFFFF]">
          <div className="relative flex-1 max-w-md">
            <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#8A949C]">
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter train # or station..."
              className="w-full pl-9 pr-3 py-1.5 bg-[#FFFFFF] border border-[#D9DEE3] rounded-md text-xs sm:text-[13px] text-[#172026] placeholder-[#8A949C] focus:outline-none focus:border-[#2563A8] focus:ring-1 focus:ring-[#2563A8] transition-colors"
            />
          </div>

          {/* Quick Simulation / Disruption Trigger */}
          <div className="flex items-center space-x-2 shrink-0">
            <button
              type="button"
              onClick={() => setIsDrawerOpen(true)}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-[#FFFFFF] hover:bg-[#F0F3F5] text-[#172026] border border-[#D9DEE3] rounded-md text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
            >
              <svg className="w-3.5 h-3.5 text-[#B77900]" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>Inject Event</span>
            </button>
          </div>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="p-6 text-center bg-rose-50 border-b border-rose-200 text-[#D64545]">
            <p className="font-semibold text-sm">Failed to connect to backend server</p>
            <p className="text-xs text-rose-600 mt-1">{error}</p>
            <button
              type="button"
              onClick={() => fetchDashboardData(true)}
              className="mt-3 px-3 py-1.5 bg-[#FFFFFF] border border-rose-300 text-xs font-semibold rounded-md shadow-2xs hover:bg-rose-100/50"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* View Content: Table or Cards */}
        {loading ? (
          <div className="p-4">
            <table className="w-full">
              <tbody className="divide-y divide-[#D9DEE3]/70">
                {Array.from({ length: 8 }).map((_, i) => (
                  <TableRowSkeleton key={i} cols={10} />
                ))}
              </tbody>
            </table>
          </div>
        ) : filteredTrains.length === 0 ? (
          <div className="p-12 text-center text-[#66717A]">
            <p className="text-sm font-semibold">No active trains matching query.</p>
            <p className="text-xs text-[#8A949C] mt-1">Try clearing search or filters.</p>
            <button
              type="button"
              onClick={() => {
                setSearchQuery("");
                setStatusFilter("ALL");
              }}
              className="mt-3 px-3 py-1.5 bg-[#F0F3F5] text-xs font-medium rounded-md hover:bg-[#E2E8F0]"
            >
              Clear Filters
            </button>
          </div>
        ) : viewMode === "table" ? (
          /* Operational Train Table matching visual reference */
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left border-collapse text-xs sm:text-[13px]">
              <thead>
                <tr className="border-b border-[#D9DEE3] bg-[#FFFFFF] text-[11px] font-semibold text-[#66717A] tracking-wider uppercase">
                  <th className="py-2.5 px-3.5 pl-4">TRAIN</th>
                  <th className="py-2.5 px-3">SOURCE</th>
                  <th className="py-2.5 px-3">CURRENT STATION</th>
                  <th className="py-2.5 px-3 text-right">SPEED</th>
                  <th className="py-2.5 px-3 text-center">DELAY</th>
                  <th className="py-2.5 px-2 text-center">TREND</th>
                  <th className="py-2.5 px-3">NEXT STATION</th>
                  <th className="py-2.5 px-3 text-right">BASELINE ETA</th>
                  <th className="py-2.5 px-3 text-right">ML ETA</th>
                  <th className="py-2.5 px-3 text-right">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9DEE3]/70 bg-[#FFFFFF]">
                {filteredTrains.map((train) => {
                  const state = train.current_state;
                  const isHalted = state?.status === "HALTED";
                  const delayVal = train.current_delay_minutes ?? state?.current_delay_minutes ?? 0;
                  const speedVal = state?.speed_kmh ?? 0;
                  const trendVal = train.delay_trend ?? 0;

                  // High severity stripe on left edge (like 12302 in reference)
                  const leftStripeClass = isHalted || delayVal > 15
                    ? "border-l-4 border-l-[#D64545]"
                    : "border-l-4 border-l-transparent";

                  return (
                    <tr
                      key={train.id}
                      onClick={() => router.push(`/trains/${train.train_number}`)}
                      className={`${leftStripeClass} hover:bg-[#F8FAFB] transition-colors cursor-pointer group`}
                    >
                      {/* TRAIN: Status dot + Train number */}
                      <td className="py-2.5 px-3.5 pl-3 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <span
                            className={`w-2 h-2 rounded-full shrink-0 ${
                              isHalted
                                ? "bg-[#D64545]"
                                : "bg-[#168A55]"
                            }`}
                            title={isHalted ? "Train Halted" : "Running Normally"}
                          />
                          <span className="font-mono font-bold text-[#172026] text-xs sm:text-[13px] group-hover:text-[#2563A8] transition-colors">
                            {train.train_number}
                          </span>
                        </div>
                      </td>

                      {/* SOURCE */}
                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="inline-block px-1.5 py-0.2 rounded text-[10px] font-mono font-medium tracking-wider bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
                          {train.data_source === "LIVE_API" || state?.source === "external_api"
                            ? "LIVE"
                            : "SIMULATOR"}
                        </span>
                      </td>

                      {/* CURRENT STATION */}
                      <td className="py-2.5 px-3 whitespace-nowrap font-mono font-medium text-[#172026]">
                        {train.current_station || state?.current_station_code || "--"}
                      </td>

                      {/* SPEED */}
                      <td className="py-2.5 px-3 whitespace-nowrap text-right font-mono text-[#66717A]">
                        {speedVal.toFixed(0)} km/h
                      </td>

                      {/* DELAY */}
                      <td className="py-2.5 px-3 whitespace-nowrap text-center font-mono">
                        {delayVal > 1.0 ? (
                          <span className="font-bold text-[#D64545]">
                            +{delayVal.toFixed(1)} min
                          </span>
                        ) : (
                          <span className="text-[#66717A]">On Time</span>
                        )}
                      </td>

                      {/* TREND */}
                      <td className="py-2.5 px-2 whitespace-nowrap text-center font-mono text-xs">
                        {trendVal > 0 ? (
                          <span className="text-[#D64545] font-bold" title="Delay increasing">
                            ↗
                          </span>
                        ) : trendVal < 0 ? (
                          <span className="text-[#168A55] font-bold" title="Recovering time">
                            ↘
                          </span>
                        ) : (
                          <span className="text-[#8A949C]" title="Stable">
                            →
                          </span>
                        )}
                      </td>

                      {/* NEXT STATION */}
                      <td className="py-2.5 px-3 whitespace-nowrap font-mono font-medium text-[#172026]">
                        {train.next_station || state?.next_station_code || "Terminus"}
                      </td>

                      {/* BASELINE ETA */}
                      <td className="py-2.5 px-3 whitespace-nowrap text-right font-mono font-medium text-[#66717A]">
                        {train.baseline_eta ? formatTime(train.baseline_eta) : "--:--"}
                      </td>

                      {/* ML ETA */}
                      <td className="py-2.5 px-3 text-right whitespace-nowrap font-mono font-bold">
                        {train.ml_eta ? (
                          <span
                            className={
                              delayVal > 5.0
                                ? "text-[#D64545]"
                                : "text-[#168A55]"
                            }
                          >
                            {formatTime(train.ml_eta)}
                          </span>
                        ) : (
                          <span className="text-[#8A949C]">--:--</span>
                        )}
                      </td>

                      {/* ACTION */}
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <span className="inline-block px-2 py-0.5 rounded bg-[#F0F3F5] group-hover:bg-[#172026] group-hover:text-white border border-[#D9DEE3] text-[#172026] text-[11px] font-semibold transition-colors">
                          Stops &rarr;
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          /* Clean Cards Grid View */
          <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 bg-[#F5F7F8]">
            {filteredTrains.map((train) => {
              const state = train.current_state;
              const isHalted = state?.status === "HALTED";
              const delayVal = train.current_delay_minutes ?? state?.current_delay_minutes ?? 0;
              return (
                <div
                  key={train.id}
                  onClick={() => router.push(`/trains/${train.train_number}`)}
                  className={`bg-[#FFFFFF] p-3.5 rounded-lg border border-[#D9DEE3] hover:border-[#2563A8] transition-all cursor-pointer shadow-2xs space-y-2.5 ${
                    isHalted ? "border-l-4 border-l-[#D64545]" : ""
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          isHalted ? "bg-[#D64545]" : "bg-[#168A55]"
                        }`}
                      />
                      <span className="font-mono font-bold text-sm text-[#172026]">
                        {train.train_number}
                      </span>
                    </div>
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
                      {train.data_source === "LIVE_API" ? "LIVE" : "SIM"}
                    </span>
                  </div>
                  <div className="text-xs text-[#66717A] truncate font-medium">
                    {train.name}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-[#D9DEE3]/70 font-mono">
                    <div>
                      <div className="text-[10px] text-[#8A949C]">AT / NEXT</div>
                      <div className="font-semibold text-[#172026]">
                        {train.current_station || "--"} &rarr; {train.next_station || "--"}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] text-[#8A949C]">DELAY</div>
                      <div
                        className={
                          delayVal > 1.0
                            ? "font-bold text-[#D64545]"
                            : "text-[#66717A]"
                        }
                      >
                        {delayVal > 1.0 ? `+${delayVal.toFixed(1)}m` : "On Time"}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-[#D9DEE3]/70 text-xs font-mono">
                    <div className="text-[#66717A]">
                      Baseline: {train.baseline_eta ? formatTime(train.baseline_eta) : "--:--"}
                    </div>
                    <div className="font-bold text-[#168A55]">
                      ML: {train.ml_eta ? formatTime(train.ml_eta) : "--:--"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Table Footer with Summary & Refresh Status */}
        <div className="p-3 border-t border-[#D9DEE3] bg-[#FFFFFF] flex flex-col sm:flex-row items-center justify-between text-xs text-[#8A949C] gap-2">
          <div>
            Showing 1-{filteredTrains.length} of {totalTrainsCount} active trains
          </div>
          <div className="flex items-center space-x-2">
            {isUpdating && (
              <span className="w-1.5 h-1.5 rounded-full bg-[#2563A8] animate-ping" />
            )}
            <span>
              Last updated: {lastRefreshed ? formatTime(lastRefreshed.toISOString()) : "Just now"}
            </span>
            <button
              type="button"
              onClick={() => fetchDashboardData(false)}
              className="text-[#66717A] hover:text-[#172026] ml-1 p-0.5"
              title="Refresh Fleet State"
            >
              ↻
            </button>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SIMULATION & EVENT INJECTION DRAWER                                       */}
      {/* Preserves 100% of simulator and demo capabilities without cluttering      */}
      {/* the main operations floor.                                                */}
      {/* ========================================================================= */}
      {isDrawerOpen && (
        <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-[#172026]/30 backdrop-blur-2xs transition-opacity"
            onClick={() => setIsDrawerOpen(false)}
          />

          {/* Slide-over panel */}
          <div className="relative w-full max-w-lg bg-[#FFFFFF] border-l border-[#D9DEE3] shadow-xl flex flex-col h-full z-50">
            {/* Drawer Header */}
            <div className="p-4 border-b border-[#D9DEE3] flex items-center justify-between bg-[#F8FAFB]">
              <div className="flex items-center space-x-2">
                <span className="text-base">⚡</span>
                <h2 className="text-sm font-bold text-[#172026] uppercase tracking-wide">
                  Simulation & Disruption Control
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setIsDrawerOpen(false)}
                className="w-7 h-7 flex items-center justify-center rounded-md border border-[#D9DEE3] text-[#66717A] hover:text-[#172026] hover:bg-[#FFFFFF]"
              >
                ✕
              </button>
            </div>

            {/* Drawer Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 text-xs">
              {/* SECTION 1: Deterministic Demo Scenario (Seed 42) */}
              <div className="p-3.5 bg-[#F8FAFB] border border-[#D9DEE3] rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-[#172026] uppercase tracking-wider text-[11px]">
                    Deterministic Demo Scenario
                  </span>
                  <button
                    type="button"
                    onClick={handleResetDemo}
                    disabled={demoLoading}
                    className="px-2 py-1 rounded bg-[#FFFFFF] border border-[#D9DEE3] text-[11px] font-semibold text-[#172026] hover:bg-[#F0F3F5] cursor-pointer"
                  >
                    {demoLoading ? "Resetting..." : "Reset to Baseline"}
                  </button>
                </div>

                {demoScenario && (
                  <div className="p-2.5 bg-[#FFFFFF] border border-[#D9DEE3] rounded text-xs space-y-1.5 font-mono">
                    <div className="flex justify-between text-[#66717A]">
                      <span>Train: <strong>{demoScenario.train_number}</strong></span>
                      <span className="font-semibold text-[#172026]">{demoScenario.train_status}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Section: {demoScenario.current_station} &rarr; {demoScenario.next_station}</span>
                      <span className="font-bold text-[#D64545]">
                        +{demoScenario.current_delay_minutes?.toFixed(1) ?? "0.0"}m delay
                      </span>
                    </div>
                  </div>
                )}

                {/* Pre-packaged Actions */}
                <div className="space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-[#8A949C]">
                    Trigger Disruption Actions:
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <button
                      type="button"
                      onClick={() => handleExecuteDemoAction("signal_halt")}
                      disabled={demoActionActive !== null}
                      className="p-2 bg-[#FFFFFF] hover:bg-rose-50 border border-[#D9DEE3] hover:border-[#D64545] rounded text-left transition-colors cursor-pointer"
                    >
                      <div className="font-bold text-[#D64545] text-[11px]">1. Signal Halt</div>
                      <div className="text-[10px] text-[#8A949C]">+12m at red aspect</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleExecuteDemoAction("congestion")}
                      disabled={demoActionActive !== null}
                      className="p-2 bg-[#FFFFFF] hover:bg-amber-50 border border-[#D9DEE3] hover:border-[#B77900] rounded text-left transition-colors cursor-pointer"
                    >
                      <div className="font-bold text-[#B77900] text-[11px]">2. Freight Congestion</div>
                      <div className="text-[10px] text-[#8A949C]">+8m speed drop</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleExecuteDemoAction("speed_restriction")}
                      disabled={demoActionActive !== null}
                      className="p-2 bg-[#FFFFFF] hover:bg-blue-50 border border-[#D9DEE3] hover:border-[#2563A8] rounded text-left transition-colors cursor-pointer"
                    >
                      <div className="font-bold text-[#2563A8] text-[11px]">3. Speed Caution</div>
                      <div className="text-[10px] text-[#8A949C]">+15m track work</div>
                    </button>
                  </div>
                </div>

                {/* Action Feedback */}
                {demoResult && (
                  <div className="p-2.5 bg-[#EBF7EE] border border-[#B4E2C1] rounded text-[11px] text-[#168A55] font-mono">
                    ✓ {demoResult.action_name} executed &bull; New delay: +{(demoResult.new_delay_minutes ?? 0).toFixed(1)}m &bull; ML ETA: {formatTime(demoResult.ml_eta)}
                  </div>
                )}
                {demoError && (
                  <div className="p-2.5 bg-[#FDF2F2] border border-[#F5C2C2] rounded text-[11px] text-[#D64545]">
                    {demoError}
                  </div>
                )}
              </div>

              {/* SECTION 2: Custom Disruption Event Injection */}
              <form onSubmit={handleInjectEvent} className="p-3.5 bg-[#FFFFFF] border border-[#D9DEE3] rounded-lg space-y-3">
                <span className="font-bold text-[#172026] uppercase tracking-wider text-[11px] block">
                  Custom Event Injection
                </span>

                <div>
                  <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                    Select Target Train
                  </label>
                  <select
                    value={selectedTrain}
                    onChange={(e) => setSelectedTrain(e.target.value)}
                    className="w-full px-2.5 py-1.5 bg-[#F5F7F8] border border-[#D9DEE3] rounded text-xs font-mono text-[#172026]"
                  >
                    {trains.map((t) => (
                      <option key={t.id} value={t.train_number}>
                        {t.train_number} - {t.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <div>
                    <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                      Event Type
                    </label>
                    <select
                      value={eventType}
                      onChange={(e) => setEventType(e.target.value)}
                      className="w-full px-2 py-1.5 bg-[#F5F7F8] border border-[#D9DEE3] rounded text-xs font-mono text-[#172026]"
                    >
                      <option value="SIGNAL_HALT">Signal Halt</option>
                      <option value="SPEED_RESTRICTION">Speed Restriction</option>
                      <option value="WEATHER">Adverse Weather</option>
                      <option value="CONGESTION">Platform Congestion</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                      Severity
                    </label>
                    <select
                      value={severity}
                      onChange={(e) => setSeverity(e.target.value)}
                      className="w-full px-2 py-1.5 bg-[#F5F7F8] border border-[#D9DEE3] rounded text-xs font-mono text-[#172026]"
                    >
                      <option value="LOW">Low</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="HIGH">High</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                      Delay (Mins)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="180"
                      value={delayMinutes}
                      onChange={(e) => setDelayMinutes(Number(e.target.value))}
                      className="w-full px-2 py-1.5 bg-[#F5F7F8] border border-[#D9DEE3] rounded text-xs font-mono text-[#172026]"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                    Operational Note
                  </label>
                  <input
                    type="text"
                    value={locationNote}
                    onChange={(e) => setLocationNote(e.target.value)}
                    className="w-full px-2.5 py-1.5 bg-[#F5F7F8] border border-[#D9DEE3] rounded text-xs text-[#172026]"
                  />
                </div>

                <button
                  type="submit"
                  disabled={submittingEvent}
                  className="w-full py-2 bg-[#172026] hover:bg-[#2563A8] text-white font-semibold rounded text-xs transition-colors cursor-pointer"
                >
                  {submittingEvent ? "Injecting..." : "Simulate Event"}
                </button>

                {injectionResult && (
                  <div className="p-2.5 bg-[#EBF7EE] border border-[#B4E2C1] rounded text-[11px] text-[#168A55] font-mono">
                    ✓ {injectionResult.message} &bull; Updated ML ETA: {formatTime(injectionResult.ml_eta)}
                  </div>
                )}
                {injectionError && (
                  <div className="p-2.5 bg-[#FDF2F2] border border-[#F5C2C2] rounded text-[11px] text-[#D64545]">
                    {injectionError}
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
