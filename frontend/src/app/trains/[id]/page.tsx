"use client";

import { useEffect, useState, useCallback, use } from "react";
import Link from "next/link";
import { api, formatDateTime, formatDelay } from "@/lib/api";
import {
  TrainDetailResponse,
  RouteStationInfo,
  UpcomingStationETA,
  EventDetails,
  EventInjectionResponse,
} from "@/types/api";
import UncertaintyRangeBar from "@/components/UncertaintyRangeBar";
import { StatusBadge } from "@/components/StatusBadge";
import EtaComparisonBadge from "@/components/EtaComparisonBadge";
import { Skeleton, CardSkeleton, TableRowSkeleton } from "@/components/LoadingSkeleton";

interface PageProps {
  params: Promise<{ id: string }>;
}

// Operational Sparkline Visualization for Delay Trend
function DelayTrendSparkline({
  history,
  trend,
  width = 110,
  height = 28,
}: {
  history: number[];
  trend: number;
  width?: number;
  height?: number;
}) {
  const dataPoints = history && history.length > 0 ? history : [0];
  const minVal = Math.min(...dataPoints);
  const maxVal = Math.max(...dataPoints);
  const range = maxVal - minVal > 0 ? maxVal - minVal : 1;

  const points = dataPoints
    .map((val, idx) => {
      const x =
        dataPoints.length > 1
          ? (idx / (dataPoints.length - 1)) * (width - 10) + 5
          : width / 2;
      const y = height - 5 - ((val - minVal) / range) * (height - 10);
      return `${x},${y}`;
    })
    .join(" ");

  const strokeColor =
    trend > 0 ? "#D64545" : trend < 0 ? "#168A55" : "#66717A";

  return (
    <div className="flex items-center space-x-2">
      <svg
        width={width}
        height={height}
        className="overflow-visible bg-[#F0F3F5] border border-[#D9DEE3] rounded px-1"
      >
        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
        {dataPoints.map((val, idx) => {
          const x =
            dataPoints.length > 1
              ? (idx / (dataPoints.length - 1)) * (width - 10) + 5
              : width / 2;
          const y = height - 5 - ((val - minVal) / range) * (height - 10);
          return (
            <circle
              key={idx}
              cx={x}
              cy={y}
              r={idx === dataPoints.length - 1 ? 3 : 1.5}
              fill={strokeColor}
            />
          );
        })}
      </svg>
      <div className="text-xs font-mono whitespace-nowrap">
        {trend > 0 ? (
          <span className="text-[#D64545] font-semibold" title="Delay increasing">
            ↗ +{trend.toFixed(1)}m
          </span>
        ) : trend < 0 ? (
          <span className="text-[#168A55] font-semibold" title="Recovering time">
            ↘ {trend.toFixed(1)}m
          </span>
        ) : (
          <span className="text-[#8A949C] font-medium" title="Delay stable">
            → 0.0m
          </span>
        )}
      </div>
    </div>
  );
}

// Clean Route Progress Bar Component
function RouteProgressBar({
  routeStations,
  currentStationCode,
  nextStationCode,
  segmentProgress = 0,
}: {
  routeStations: RouteStationInfo[];
  currentStationCode: string | null;
  nextStationCode: string | null;
  segmentProgress?: number;
}) {
  if (!routeStations || routeStations.length === 0) return null;

  const currentIdx = routeStations.findIndex(
    (s) => s.station_code === currentStationCode
  );
  const effectiveCurrentIdx = currentIdx >= 0 ? currentIdx : 0;
  const totalStops = routeStations.length;

  const totalDistance =
    routeStations[totalStops - 1]?.distance_from_source_km || 1;
  const currentStnDist =
    routeStations[effectiveCurrentIdx]?.distance_from_source_km || 0;
  const nextStnDist =
    effectiveCurrentIdx < totalStops - 1
      ? routeStations[effectiveCurrentIdx + 1]?.distance_from_source_km
      : currentStnDist;

  const estimatedCurrentKm =
    currentStnDist + (nextStnDist - currentStnDist) * (segmentProgress || 0);
  const progressPercent = Math.min(
    100,
    Math.max(0, (estimatedCurrentKm / totalDistance) * 100)
  );

  return (
    <div className="space-y-3">
      {/* Top progress metrics */}
      <div className="flex items-center justify-between text-xs text-[#66717A]">
        <div>
          <span className="font-semibold text-[#172026]">
            {routeStations[0].station_code}
          </span>{" "}
          &rarr;{" "}
          <span className="font-semibold text-[#172026]">
            {routeStations[totalStops - 1].station_code}
          </span>{" "}
          <span className="text-[#8A949C]">({totalDistance.toFixed(0)} km total)</span>
        </div>
        <div className="font-mono font-medium text-[#2563A8]">
          {progressPercent.toFixed(1)}% Completed ({estimatedCurrentKm.toFixed(0)} km)
        </div>
      </div>

      {/* Progress Track Bar */}
      <div className="relative w-full h-2 bg-[#F0F3F5] rounded-full overflow-hidden border border-[#D9DEE3]">
        <div
          className="h-full bg-[#2563A8] transition-all duration-500 rounded-full"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Milestone Checkpoint Nodes */}
      <div className="flex items-center justify-between pt-1 overflow-x-auto scrollbar-none gap-2">
        {routeStations.map((stn, idx) => {
          const isPassed = idx < effectiveCurrentIdx;
          const isCurrent = idx === effectiveCurrentIdx;
          const isNext =
            stn.station_code === nextStationCode ||
            idx === effectiveCurrentIdx + 1;

          return (
            <div
              key={stn.station_code}
              className="flex flex-col items-center min-w-[48px] text-center"
            >
              <div
                className={`w-2.5 h-2.5 rounded-full border transition-all ${
                  isCurrent
                    ? "bg-[#2563A8] border-[#2563A8] ring-2 ring-[#BFDBFE]"
                    : isPassed
                    ? "bg-[#168A55] border-[#168A55]"
                    : isNext
                    ? "bg-[#FFFFFF] border-[#2563A8] ring-1 ring-[#2563A8]"
                    : "bg-[#FFFFFF] border-[#D9DEE3]"
                }`}
              />
              <span
                className={`text-[10px] font-mono mt-1 ${
                  isCurrent
                    ? "font-bold text-[#2563A8]"
                    : isPassed
                    ? "text-[#66717A]"
                    : "text-[#8A949C]"
                }`}
              >
                {stn.station_code}
              </span>
              <span className="text-[9px] text-[#8A949C] font-mono">
                {stn.scheduled_arrival || stn.scheduled_departure || ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function TrainDetailPage({ params }: PageProps) {
  const resolvedParams = use(params);
  const resolvedId = resolvedParams.id;

  const [train, setTrain] = useState<TrainDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Simulation controls state
  const [selectedDelay, setSelectedDelay] = useState<number>(15);
  const [customDelay, setCustomDelay] = useState<string>("15");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("HIGH");
  const [injecting, setInjecting] = useState<boolean>(false);
  const [injectSuccess, setInjectSuccess] = useState<string | null>(null);
  const [injectError, setInjectError] = useState<string | null>(null);

  const fetchTrainDetails = useCallback(
    async (isManualRefresh = false) => {
      if (isManualRefresh) setRefreshing(true);
      try {
        const data = await api.getTrainById(resolvedId);
        setTrain(data);
        setError(null);
      } catch (err: unknown) {
        const msg =
          err instanceof Error
            ? err.message
            : `Failed to load details for train '${resolvedId}'`;
        setError(msg);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [resolvedId]
  );

  useEffect(() => {
    fetchTrainDetails();
  }, [fetchTrainDetails]);

  // Periodic polling every 15s
  useEffect(() => {
    const timer = setInterval(() => {
      fetchTrainDetails(false);
    }, 15000);
    return () => clearInterval(timer);
  }, [fetchTrainDetails]);

  // Handle Disruption Injection
  const handleInjectDisruption = async (
    eventType: string,
    delayMins?: number
  ) => {
    if (!train) return;

    setInjecting(true);
    setInjectSuccess(null);
    setInjectError(null);

    const delayToUse =
      delayMins !== undefined
        ? delayMins
        : Number(customDelay) || selectedDelay;

    try {
      const payload = {
        train_id: train.train_number,
        event_type: eventType,
        delay_minutes: delayToUse,
        severity: selectedSeverity,
        metadata: {
          source: "TrainDetailPage",
          current_station: train.current_station,
          timestamp: new Date().toISOString(),
        },
      };

      const result: EventInjectionResponse = await api.simulateEvent(payload);

      setInjectSuccess(
        `Disruption '${eventType}' injected: +${delayToUse} min delay applied. Updated status: ${result.train_state.status}. Recalculating ML ETAs...`
      );

      await fetchTrainDetails(true);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to inject operational disruption event";
      setInjectError(msg);
    } finally {
      setInjecting(false);
    }
  };

  // Demo Handlers
  const handleResetDemo = async () => {
    setInjecting(true);
    setInjectSuccess(null);
    setInjectError(null);
    try {
      const res = await api.resetDemoScenario();
      setInjectSuccess(
        `Demo Scenario Reset: Train at ${res.current_station} &rarr; ${res.next_station}, delay: +${res.current_delay_minutes}m, speed: ${res.current_speed_kmh} km/h.`
      );
      await fetchTrainDetails(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to reset demo scenario";
      setInjectError(msg);
    } finally {
      setInjecting(false);
    }
  };

  const handleExecuteDemoAction = async (actionId: string) => {
    setInjecting(true);
    setInjectSuccess(null);
    setInjectError(null);
    try {
      const res = await api.executeDemoAction(actionId);
      setInjectSuccess(
        `Demo Action '${res.action_name}' applied: Delay +${res.new_delay_minutes.toFixed(1)}m, Status: ${res.new_status}, Speed: ${res.speed_kmh.toFixed(1)} km/h.`
      );
      await fetchTrainDetails(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : `Failed to execute demo action ${actionId}`;
      setInjectError(msg);
    } finally {
      setInjecting(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="space-y-2 pb-4 border-b border-[#D9DEE3]">
          <Skeleton className="h-4 w-36" />
          <div className="flex justify-between items-center">
            <Skeleton className="h-8 w-72" />
            <Skeleton className="h-7 w-28 rounded" />
          </div>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
        </div>
        <div className="bg-[#FFFFFF] p-5 rounded-lg border border-[#D9DEE3] space-y-4">
          <Skeleton className="h-5 w-48" />
          <Skeleton className="h-2 w-full rounded-full" />
        </div>
        <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] overflow-hidden p-4">
          <Skeleton className="h-6 w-56 mb-4" />
          <table className="w-full">
            <tbody className="divide-y divide-[#D9DEE3]/70">
              <TableRowSkeleton cols={7} />
              <TableRowSkeleton cols={7} />
              <TableRowSkeleton cols={7} />
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (error || !train) {
    return (
      <div className="space-y-4 max-w-2xl">
        <Link
          href="/"
          className="inline-flex items-center text-xs font-semibold text-[#2563A8] hover:underline"
        >
          &larr; Back to Control Room
        </Link>
        <div className="p-5 bg-rose-50 border border-rose-200 text-[#D64545] rounded-lg text-sm space-y-2">
          <p className="font-bold text-base">Train Not Found or Backend Error</p>
          <p className="font-mono text-xs text-rose-700">{error}</p>
          <p className="text-xs text-[#66717A] pt-1">
            Please verify the train ID or ensure the backend is active on port 8000.
          </p>
          <button
            type="button"
            onClick={() => fetchTrainDetails(true)}
            className="mt-2 px-3 py-1.5 bg-[#D64545] hover:bg-rose-700 text-white font-medium text-xs rounded transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const state = train.current_state;
  const trainDisplayName =
    train.train_name || train.name || `Train ${train.train_number}`;
  const currentDelayMin =
    train.current_delay_minutes ?? train.current_delay ?? 0;
  const delayInfo = formatDelay(currentDelayMin);
  const upcomingCount =
    train.upcoming_stations?.length ?? train.total_upcoming_stations ?? 0;
  const activeEvents: EventDetails[] = (train.active_events || []) as EventDetails[];
  const delayTrendValue = train.delay_trend ?? 0;
  const delayHistoryValues = train.delay_history || [currentDelayMin];

  return (
    <div className="w-full max-w-[1560px] mx-auto space-y-5 pb-12">
      {/* 1. Header & Navigation */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="inline-flex items-center text-xs font-medium text-[#66717A] hover:text-[#172026] transition-colors"
          >
            &larr; Back to Control Room
          </Link>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={() => fetchTrainDetails(true)}
              disabled={refreshing}
              className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md border border-[#D9DEE3] bg-[#FFFFFF] hover:bg-[#F0F3F5] text-[#172026] shadow-2xs transition-colors disabled:opacity-60 cursor-pointer"
            >
              <span className={`mr-1.5 ${refreshing ? "animate-spin" : ""}`}>
                ↻
              </span>
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>

            <Link
              href={`/passenger?train=${encodeURIComponent(train.train_number)}`}
              className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md border border-[#2563A8] bg-[#EBF3FC] text-[#2563A8] hover:bg-blue-100 transition-colors"
            >
              Passenger View &rarr;
            </Link>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-3 border-b border-[#D9DEE3] gap-2">
          <div>
            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#172026]">
                Train {train.train_number} &mdash; {trainDisplayName}
              </h1>
              <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-[#172026] text-white tracking-wide">
                {train.train_type}
              </span>
            </div>
            <p className="text-xs text-[#66717A] mt-0.5">
              Route #{train.route_id} &bull; Telemetry Observed:{" "}
              <span className="font-mono text-[#172026]">
                {formatDateTime(train.current_timestamp)}
              </span>
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded text-xs font-bold border ${delayInfo.colorClass}`}
            >
              {delayInfo.text}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Key Telemetry Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Status */}
        <div className="bg-[#FFFFFF] p-3.5 sm:p-4 rounded-lg border border-[#D9DEE3] shadow-2xs">
          <div className="text-[10px] font-bold text-[#8A949C] uppercase tracking-wider">
            Running Status
          </div>
          <div className="mt-1.5 flex items-center space-x-2">
            <StatusBadge status={state?.status} size="md" />
          </div>
          <div className="text-xs text-[#66717A] mt-2 font-mono">
            Speed: <strong className="text-[#172026]">{state?.speed_kmh?.toFixed(0) ?? 0} km/h</strong>
          </div>
        </div>

        {/* Current Station */}
        <div className="bg-[#FFFFFF] p-3.5 sm:p-4 rounded-lg border border-[#D9DEE3] shadow-2xs">
          <div className="text-[10px] font-bold text-[#8A949C] uppercase tracking-wider">
            Current Station
          </div>
          <div className="mt-1 text-base font-bold font-mono text-[#172026]">
            {train.current_station || "Origin Station"}
          </div>
          <div className="text-xs text-[#66717A] mt-0.5 font-mono">
            Next:{" "}
            <span className="font-semibold text-[#172026]">
              {state?.next_station_code || "Terminus"}
            </span>{" "}
            {state?.next_station_distance_km !== null &&
            state?.next_station_distance_km !== undefined ? (
              <span className="text-[#8A949C]">
                ({state.next_station_distance_km.toFixed(1)} km)
              </span>
            ) : null}
          </div>
        </div>

        {/* Current Delay */}
        <div className="bg-[#FFFFFF] p-3.5 sm:p-4 rounded-lg border border-[#D9DEE3] shadow-2xs">
          <div className="text-[10px] font-bold text-[#8A949C] uppercase tracking-wider">
            Current Delay
          </div>
          <div className="mt-1">
            <span
              className={`inline-block px-2 py-0.5 text-xs font-bold border rounded font-mono ${delayInfo.colorClass}`}
            >
              {delayInfo.text}
            </span>
          </div>
          <div className="text-xs text-[#8A949C] mt-1 font-mono">
            At checkpoint {train.current_station || "--"}
          </div>
        </div>

        {/* Delay Trend */}
        <div className="bg-[#FFFFFF] p-3.5 sm:p-4 rounded-lg border border-[#D9DEE3] shadow-2xs">
          <div className="text-[10px] font-bold text-[#8A949C] uppercase tracking-wider mb-1">
            Delay Trend
          </div>
          <DelayTrendSparkline
            history={delayHistoryValues}
            trend={delayTrendValue}
          />
          <div className="text-[10px] text-[#8A949C] mt-1">
            Trajectory across recent checkpoints
          </div>
        </div>
      </div>

      {/* 3. Route Progress Visualization */}
      <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs p-4 sm:p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-2 border-b border-[#D9DEE3]/70 gap-2">
          <div>
            <h2 className="text-sm font-bold text-[#172026] flex items-center">
              <span className="mr-2">🛤️</span> Route Progress & Station Checkpoints
            </h2>
            <p className="text-xs text-[#66717A] mt-0.5">
              Live journey tracking across {train.route_stations?.length || 0} scheduled route stops.
            </p>
          </div>
          <div className="text-xs text-[#66717A] font-mono">
            {upcomingCount} stops remaining to terminus
          </div>
        </div>

        <RouteProgressBar
          routeStations={train.route_stations || []}
          currentStationCode={train.current_station}
          nextStationCode={state?.next_station_code || null}
          segmentProgress={state?.segment_progress || 0}
        />
      </div>

      {/* 4. Active Disruptions & Event Controls (Side by Side) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Active Events */}
        <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs p-4 sm:p-5 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-[#D9DEE3]/70">
            <div>
              <h2 className="text-sm font-bold text-[#172026] flex items-center">
                <span className="mr-2">🚨</span> Active Disruptions
              </h2>
              <p className="text-xs text-[#66717A] mt-0.5">
                Current operational events affecting train motion and delay.
              </p>
            </div>
            <span
              className={`text-[11px] font-bold px-2 py-0.5 rounded border ${
                activeEvents.length > 0
                  ? "bg-[#FDF2F2] text-[#D64545] border-[#F5C2C2]"
                  : "bg-[#EBF7EE] text-[#168A55] border-[#B4E2C1]"
              }`}
            >
              {activeEvents.length} Active
            </span>
          </div>

          {activeEvents.length === 0 ? (
            <div className="p-4 bg-[#F8FAFB] rounded border border-[#D9DEE3] text-center space-y-1">
              <div className="text-base">🟢</div>
              <div className="text-xs font-semibold text-[#172026]">
                No Active Operational Disruptions
              </div>
              <div className="text-[11px] text-[#8A949C]">
                Train running under nominal conditions without halts or speed restrictions.
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {activeEvents.map((evt, idx) => (
                <div
                  key={idx}
                  className="p-2.5 bg-[#FDF2F2] border border-[#F5C2C2] rounded space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-xs font-mono text-[#D64545] px-1.5 py-0.2 bg-rose-100 rounded">
                      {evt.event_type}
                    </span>
                    <span className="text-xs font-bold text-[#D64545]">
                      +{(evt.delay_minutes ?? 0).toFixed(0)} min delay
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-[#66717A]">
                    <span>
                      Severity: <strong className="text-[#172026]">{evt.severity}</strong>
                    </span>
                    <span className="text-[11px] text-[#8A949C]">
                      Injected: {formatDateTime(evt.timestamp)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Operational Event Controls */}
        <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs p-4 sm:p-5 space-y-3">
          <div className="pb-2 border-b border-[#D9DEE3]/70">
            <h2 className="text-sm font-bold text-[#172026] flex items-center">
              <span className="mr-2">⚡</span> Disruption Controls
            </h2>
            <p className="text-xs text-[#66717A] mt-0.5">
              Simulate operational events via backend <code>POST /simulate/event</code>.
            </p>
          </div>

          {/* Prepared Disruption Actions (Train 12302) */}
          {train.train_number === "12302" && (
            <div className="p-3 bg-[#F8FAFB] border border-[#D9DEE3] rounded space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[11px] text-[#172026] uppercase tracking-wider">
                  Prepared Test Actions (Seed 42)
                </span>
                <button
                  type="button"
                  onClick={handleResetDemo}
                  disabled={injecting}
                  className="px-2 py-0.5 text-[10px] font-bold text-[#172026] bg-[#FFFFFF] hover:bg-[#F0F3F5] border border-[#D9DEE3] rounded shadow-2xs transition-colors disabled:opacity-50 cursor-pointer"
                >
                  Reset Demo
                </button>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                <button
                  type="button"
                  onClick={() => handleExecuteDemoAction("signal_halt")}
                  disabled={injecting}
                  className="p-1.5 text-center text-[10px] font-bold bg-[#FFFFFF] hover:bg-rose-50 text-[#D64545] rounded border border-[#D9DEE3] hover:border-[#D64545] transition-colors disabled:opacity-50 cursor-pointer"
                >
                  1. Signal (+15m)
                </button>
                <button
                  type="button"
                  onClick={() => handleExecuteDemoAction("congestion")}
                  disabled={injecting}
                  className="p-1.5 text-center text-[10px] font-bold bg-[#FFFFFF] hover:bg-amber-50 text-[#B77900] rounded border border-[#D9DEE3] hover:border-[#B77900] transition-colors disabled:opacity-50 cursor-pointer"
                >
                  2. Congestion (+10m)
                </button>
                <button
                  type="button"
                  onClick={() => handleExecuteDemoAction("speed_restriction")}
                  disabled={injecting}
                  className="p-1.5 text-center text-[10px] font-bold bg-[#FFFFFF] hover:bg-blue-50 text-[#2563A8] rounded border border-[#D9DEE3] hover:border-[#2563A8] transition-colors disabled:opacity-50 cursor-pointer"
                >
                  3. Speed Restr (+8m)
                </button>
              </div>
            </div>
          )}

          {/* Configurable Delay & Severity Controls */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                Delay Duration:
              </label>
              <div className="flex items-center space-x-1">
                {[5, 10, 15, 25].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => {
                      setSelectedDelay(preset);
                      setCustomDelay(String(preset));
                    }}
                    className={`px-1.5 py-0.5 rounded text-[11px] font-mono border transition-colors ${
                      Number(customDelay) === preset
                        ? "bg-[#172026] text-white border-[#172026]"
                        : "bg-[#F0F3F5] text-[#66717A] border-[#D9DEE3] hover:bg-[#E2E8F0]"
                    }`}
                  >
                    +{preset}m
                  </button>
                ))}
              </div>
              <input
                type="number"
                min="1"
                max="180"
                value={customDelay}
                onChange={(e) => setCustomDelay(e.target.value)}
                className="mt-1 w-full px-2 py-1 text-xs border border-[#D9DEE3] rounded font-mono focus:border-[#2563A8] focus:outline-none"
                placeholder="Custom delay (min)"
              />
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-[#66717A] mb-1">
                Event Severity:
              </label>
              <select
                value={selectedSeverity}
                onChange={(e) => setSelectedSeverity(e.target.value)}
                className="w-full px-2 py-1 text-xs border border-[#D9DEE3] rounded bg-[#FFFFFF] focus:border-[#2563A8] focus:outline-none"
              >
                <option value="LOW">LOW</option>
                <option value="MEDIUM">MEDIUM</option>
                <option value="HIGH">HIGH</option>
              </select>
            </div>
          </div>

          {/* Event Trigger Action Buttons */}
          <div className="grid grid-cols-2 gap-2 pt-1">
            <button
              type="button"
              onClick={() => handleInjectDisruption("SIGNAL_HALT", 15)}
              disabled={injecting}
              className="flex items-center justify-center p-2 bg-[#FFFFFF] hover:bg-rose-50 border border-[#D9DEE3] text-[#D64545] font-semibold text-xs rounded shadow-2xs transition-colors disabled:opacity-50 cursor-pointer"
            >
              <span className="mr-1 text-sm">🔴</span>
              Signal Halt
            </button>

            <button
              type="button"
              onClick={() => handleInjectDisruption("CONGESTION", 10)}
              disabled={injecting}
              className="flex items-center justify-center p-2 bg-[#FFFFFF] hover:bg-amber-50 border border-[#D9DEE3] text-[#B77900] font-semibold text-xs rounded shadow-2xs transition-colors disabled:opacity-50 cursor-pointer"
            >
              <span className="mr-1 text-sm">🟠</span>
              Congestion
            </button>
          </div>

          {injecting && (
            <div className="text-xs text-[#2563A8] bg-[#EBF3FC] border border-[#BFDBFE] p-2 rounded flex items-center space-x-2">
              <span className="animate-spin text-sm">⚙️</span>
              <span>Sending disruption to backend simulator and recalculating ML ETAs...</span>
            </div>
          )}

          {injectSuccess && (
            <div className="text-xs text-[#168A55] bg-[#EBF7EE] border border-[#B4E2C1] p-2 rounded">
              <div className="font-bold">✓ Disruption Applied</div>
              <div>{injectSuccess}</div>
            </div>
          )}

          {injectError && (
            <div className="text-xs text-[#D64545] bg-[#FDF2F2] border border-[#F5C2C2] p-2 rounded">
              <div className="font-bold">Error Injecting Event</div>
              <div className="font-mono text-[11px]">{injectError}</div>
            </div>
          )}
        </div>
      </div>

      {/* 5. Multi-Station ETA Forecasts (XGBoost Chained Inference Table) */}
      <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs overflow-hidden">
        <div className="px-4 py-3 border-b border-[#D9DEE3] bg-[#FFFFFF] flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
              <span>🎯</span>
              <span>Upcoming Station ETA Forecasts</span>
            </h2>
            <p className="text-xs text-[#66717A] mt-0.5">
              Multi-segment chaining comparing Timetable Schedule, Baseline Heuristic, and XGBoost ML Predictions.
            </p>
          </div>
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3] self-start sm:self-auto font-mono">
            {upcomingCount} Upcoming Stops
          </span>
        </div>

        {/* ETA Tier Legend Strip */}
        <div className="px-4 py-2 bg-[#F8FAFB] border-b border-[#D9DEE3] flex flex-wrap items-center justify-between text-xs gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[#8A949C]">
            ETA Methodology:
          </span>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-1.5">
              <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3] uppercase">
                SCHED
              </span>
              <span className="text-[11px] text-[#66717A]">Fixed Timetable</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-[#F5F7F8] text-[#172026] border border-[#D9DEE3] uppercase">
                BASELINE
              </span>
              <span className="text-[11px] text-[#172026]">Speed Heuristic</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-[#EBF3FC] text-[#2563A8] border border-[#BFDBFE] uppercase">
                ML ETA
              </span>
              <span className="text-[11px] text-[#2563A8] font-medium">Chained XGBoost + Uncertainty Range</span>
            </div>
          </div>
        </div>

        {train.upcoming_stations.length === 0 ? (
          <div className="p-10 text-center text-sm text-[#66717A] space-y-1">
            <div className="text-2xl">🏁</div>
            <p className="font-semibold text-[#172026]">Train Reached Terminus</p>
            <p className="text-xs text-[#8A949C]">
              This train journey has completed all scheduled checkpoints on Route #{train.route_id}.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto custom-scrollbar">
            <table className="min-w-full divide-y divide-[#D9DEE3] text-xs sm:text-[13px]">
              <thead className="bg-[#FFFFFF] text-[#66717A] text-[11px] font-semibold uppercase tracking-wider text-left">
                <tr>
                  <th className="px-3.5 py-2.5">Stop</th>
                  <th className="px-3.5 py-2.5">Station</th>
                  <th className="px-3.5 py-2.5">Distance</th>
                  <th className="px-3.5 py-2.5">Scheduled</th>
                  <th className="px-3.5 py-2.5">Baseline</th>
                  <th className="px-3.5 py-2.5 min-w-[240px]">
                    ML Forecast & Uncertainty Window
                  </th>
                  <th className="px-3.5 py-2.5 text-right">Remaining Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9DEE3]/70 bg-[#FFFFFF]">
                {train.upcoming_stations.map((stn: UpcomingStationETA, idx: number) => {
                  const isImmediateNext = idx === 0;

                  return (
                    <tr
                      key={stn.station_code}
                      className={`hover:bg-[#F8FAFB] transition-colors ${
                        isImmediateNext ? "bg-[#EBF3FC]/20" : ""
                      }`}
                    >
                      <td className="px-3.5 py-2.5 text-xs text-[#8A949C] font-mono">
                        #{stn.station_sequence}
                        {isImmediateNext && (
                          <span className="block text-[10px] text-[#2563A8] font-bold">
                            NEXT
                          </span>
                        )}
                      </td>
                      <td className="px-3.5 py-2.5">
                        <div className="font-bold font-mono text-[#172026]">
                          {stn.station_code}
                        </div>
                        <div className="text-xs text-[#66717A]">
                          {stn.station_name || stn.station_code}
                        </div>
                      </td>
                      <td className="px-3.5 py-2.5 text-xs text-[#66717A] font-mono">
                        {(stn.distance_to_go_km ?? 0).toFixed(1)} km
                        <span className="text-[#8A949C] block text-[10px]">
                          +{stn.segments_ahead} seg ahead
                        </span>
                      </td>
                      <td className="px-3.5 py-2.5 whitespace-nowrap">
                        <EtaComparisonBadge type="scheduled" time={stn.scheduled_eta} size="sm" />
                      </td>
                      <td className="px-3.5 py-2.5 whitespace-nowrap">
                        <EtaComparisonBadge type="baseline" time={stn.baseline_eta} size="sm" />
                      </td>
                      <td className="px-3.5 py-2.5">
                        <UncertaintyRangeBar
                          mlEta={stn.ml_eta || stn.predicted_eta}
                          lowerBound={stn.confidence_lower_bound}
                          upperBound={stn.confidence_upper_bound}
                          marginMinutes={stn.confidence_range?.margin_minutes}
                          segmentsAhead={stn.segments_ahead}
                          variant="table"
                          theme="light"
                        />
                      </td>
                      <td className="px-3.5 py-2.5 text-right font-mono text-xs text-[#172026] font-semibold whitespace-nowrap">
                        <div>
                          {stn.predicted_remaining_minutes !== undefined &&
                          stn.predicted_remaining_minutes !== null
                            ? `${stn.predicted_remaining_minutes.toFixed(1)}m`
                            : "--"}
                        </div>
                        <div className="text-[10px] text-[#8A949C] font-sans">
                          {stn.segments_ahead} stop{stn.segments_ahead > 1 ? "s" : ""}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 6. Granular Segment Predictions (Inference Breakdown) */}
      {train.segment_predictions && train.segment_predictions.length > 0 && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs overflow-hidden">
          <div className="px-4 py-3 border-b border-[#D9DEE3] bg-[#FFFFFF] flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
            <div>
              <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
                <span>📊</span>
                <span>Segment-by-Segment ML Chaining Breakdown</span>
              </h2>
              <p className="text-xs text-[#66717A] mt-0.5">
                Granular transit times predicted by the XGBoost regression model for individual route links.
              </p>
            </div>
            <span className="text-[11px] font-mono text-[#66717A]">
              {train.segment_predictions.length} Inter-station Segments
            </span>
          </div>

          <div className="overflow-x-auto custom-scrollbar">
            <table className="min-w-full divide-y divide-[#D9DEE3] text-xs sm:text-[13px]">
              <thead className="bg-[#FFFFFF] text-[#66717A] text-[11px] font-semibold uppercase tracking-wider text-left">
                <tr>
                  <th className="px-3.5 py-2.5">Order</th>
                  <th className="px-3.5 py-2.5">Segment Section</th>
                  <th className="px-3.5 py-2.5">Distance</th>
                  <th className="px-3.5 py-2.5">Timetable</th>
                  <th className="px-3.5 py-2.5">Baseline</th>
                  <th className="px-3.5 py-2.5">Predicted ML</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9DEE3]/70 bg-[#FFFFFF]">
                {train.segment_predictions.map((seg, idx) => {
                  const segOrder = seg.segment_index ?? seg.segment_order ?? idx + 1;
                  const distKm = seg.segment_distance_km ?? seg.distance_km ?? 0;
                  const schedMin = seg.scheduled_transit_minutes ?? seg.scheduled_minutes ?? 0;
                  const baseMin = seg.baseline_transit_minutes ?? seg.baseline_minutes ?? schedMin;
                  const predMin = seg.predicted_transit_minutes ?? seg.predicted_minutes ?? 0;

                  return (
                    <tr key={segOrder} className="hover:bg-[#F8FAFB] transition-colors">
                      <td className="px-3.5 py-2.5 text-xs text-[#8A949C] font-mono">
                        Segment #{segOrder}
                      </td>
                      <td className="px-3.5 py-2.5 font-semibold font-mono text-[#172026]">
                        {seg.from_station_code} &rarr; {seg.to_station_code}
                        {seg.to_station_name && (
                          <span className="block text-[11px] font-normal text-[#8A949C]">
                            {seg.to_station_name}
                          </span>
                        )}
                      </td>
                      <td className="px-3.5 py-2.5 text-xs text-[#66717A] font-mono">
                        {distKm.toFixed(1)} km
                      </td>
                      <td className="px-3.5 py-2.5 text-xs text-[#66717A] font-mono">
                        {schedMin.toFixed(1)} min
                      </td>
                      <td className="px-3.5 py-2.5 text-xs text-[#172026] font-mono">
                        {baseMin.toFixed(1)} min
                      </td>
                      <td className="px-3.5 py-2.5 text-xs font-bold text-[#2563A8] font-mono bg-[#EBF3FC]/40">
                        {predMin.toFixed(1)} min
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 7. Full Route Timetable Schedule */}
      {train.route_stations && train.route_stations.length > 0 && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs overflow-hidden">
          <div className="px-4 py-3 border-b border-[#D9DEE3] bg-[#FFFFFF] flex items-center justify-between">
            <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
              <span>🗓️</span>
              <span>Full Route Timetable Schedule</span>
            </h2>
            <span className="text-xs text-[#66717A] font-mono">
              Route #{train.route_id} ({train.route_stations.length} Stops)
            </span>
          </div>

          <div className="overflow-x-auto custom-scrollbar">
            <table className="min-w-full divide-y divide-[#D9DEE3] text-xs">
              <thead className="bg-[#FFFFFF] text-[#66717A] text-[11px] font-semibold uppercase tracking-wider text-left">
                <tr>
                  <th className="px-3.5 py-2.5">Seq</th>
                  <th className="px-3.5 py-2.5">Station</th>
                  <th className="px-3.5 py-2.5">Name</th>
                  <th className="px-3.5 py-2.5">Distance</th>
                  <th className="px-3.5 py-2.5">Scheduled Arr</th>
                  <th className="px-3.5 py-2.5">Scheduled Dep</th>
                  <th className="px-3.5 py-2.5">Halt</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9DEE3]/70 bg-[#FFFFFF]">
                {train.route_stations.map((rs: RouteStationInfo) => (
                  <tr
                    key={rs.station_code}
                    className="hover:bg-[#F8FAFB] transition-colors"
                  >
                    <td className="px-3.5 py-2 text-xs font-mono text-[#8A949C]">
                      #{rs.sequence}
                    </td>
                    <td className="px-3.5 py-2 font-bold font-mono text-[#172026]">
                      {rs.station_code}
                    </td>
                    <td className="px-3.5 py-2 text-xs text-[#66717A]">
                      {rs.station_name || rs.station_code}
                    </td>
                    <td className="px-3.5 py-2 text-xs font-mono text-[#66717A]">
                      {(rs.distance_from_source_km ?? 0).toFixed(1)} km
                    </td>
                    <td className="px-3.5 py-2 text-xs font-mono text-[#66717A]">
                      {rs.scheduled_arrival || "--:--"}
                    </td>
                    <td className="px-3.5 py-2 text-xs font-mono text-[#66717A]">
                      {rs.scheduled_departure || "--:--"}
                    </td>
                    <td className="px-3.5 py-2 text-xs font-mono text-[#8A949C]">
                      {(rs.scheduled_stop_minutes || rs.dwell_time_minutes) ? `${rs.scheduled_stop_minutes ?? rs.dwell_time_minutes} min` : "Origin/Term"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
