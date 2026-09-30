"use client";

import { useState, useEffect, useCallback, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { api, formatTime, formatDelay } from "@/lib/api";
import {
  TrainListItem,
  TrainDetailResponse,
  UpcomingStationETA,
} from "@/types/api";
import UncertaintyRangeBar from "@/components/UncertaintyRangeBar";
import { StatusBadge } from "@/components/StatusBadge";
import { CardSkeleton } from "@/components/LoadingSkeleton";

function PassengerViewContent() {
  const searchParams = useSearchParams();
  const initialTrainParam = searchParams.get("train") || "12302";

  const [trainsList, setTrainsList] = useState<TrainListItem[]>([]);
  const [selectedTrainNumber, setSelectedTrainNumber] = useState<string>(initialTrainParam);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [trainDetails, setTrainDetails] = useState<TrainDetailResponse | null>(null);
  const [selectedStationCode, setSelectedStationCode] = useState<string>("");

  const [loadingList, setLoadingList] = useState<boolean>(true);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  // 1. Fetch available trains list
  useEffect(() => {
    async function loadTrains() {
      try {
        setLoadingList(true);
        const res = await api.getTrains();
        setTrainsList(res.trains);
      } catch (err: unknown) {
        console.error("Failed to load trains list", err);
      } finally {
        setLoadingList(false);
      }
    }
    loadTrains();
  }, []);

  // 2. Fetch train details for the selected train
  const fetchSelectedTrain = useCallback(
    async (trainNum: string, isSilent = false) => {
      if (!trainNum.trim()) return;
      if (!isSilent) setLoadingDetails(true);
      else setIsRefreshing(true);
      setError(null);
      try {
        const data = await api.getTrainDetails(trainNum.trim());
        setTrainDetails(data);
        setLastRefreshed(new Date());

        if (data.upcoming_stations && data.upcoming_stations.length > 0) {
          const exists = data.upcoming_stations.some(
            (s) => s.station_code === selectedStationCode
          );
          if (!exists) {
            setSelectedStationCode(data.upcoming_stations[0].station_code);
          }
        } else {
          setSelectedStationCode("");
        }
      } catch (err: unknown) {
        const msg =
          err instanceof Error
            ? err.message
            : "Unable to retrieve train running state";
        if (!isSilent) {
          setError(msg);
          setTrainDetails(null);
        }
      } finally {
        if (!isSilent) setLoadingDetails(false);
        else setIsRefreshing(false);
      }
    },
    [selectedStationCode]
  );

  useEffect(() => {
    if (selectedTrainNumber) {
      fetchSelectedTrain(selectedTrainNumber);
    }
  }, [selectedTrainNumber, fetchSelectedTrain]);

  // 3. Periodic Auto-refresh every 10 seconds for real-time passenger sync
  useEffect(() => {
    if (!selectedTrainNumber) return;
    const interval = setInterval(() => {
      fetchSelectedTrain(selectedTrainNumber, true);
    }, 10000);
    return () => clearInterval(interval);
  }, [selectedTrainNumber, fetchSelectedTrain]);

  // Filtered train search options
  const filteredTrains = useMemo(() => {
    if (!searchQuery.trim()) return trainsList;
    const q = searchQuery.toLowerCase().trim();
    return trainsList.filter(
      (t) =>
        t.train_number.toLowerCase().includes(q) ||
        t.name.toLowerCase().includes(q)
    );
  }, [trainsList, searchQuery]);

  // Target upcoming station forecast
  const targetStationETA: UpcomingStationETA | null = useMemo(() => {
    if (!trainDetails?.upcoming_stations || trainDetails.upcoming_stations.length === 0) {
      return null;
    }
    if (!selectedStationCode) {
      return trainDetails.upcoming_stations[0];
    }
    const found = trainDetails.upcoming_stations.find(
      (s) => s.station_code === selectedStationCode
    );
    return found || trainDetails.upcoming_stations[0];
  }, [trainDetails, selectedStationCode]);

  const trainName =
    trainDetails?.train_name ||
    trainDetails?.name ||
    trainsList.find((t) => t.train_number === selectedTrainNumber)?.name ||
    `Train ${selectedTrainNumber}`;

  const currentDelayMinutes =
    trainDetails?.current_delay_minutes ??
    trainDetails?.current_delay ??
    0;
  const delayInfo = formatDelay(currentDelayMinutes);
  const currentStation = trainDetails?.current_station || "--";

  return (
    <div className="max-w-[1040px] mx-auto w-full space-y-4 pb-12">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-1 border-b border-[#D9DEE3]">
        <div className="flex items-center space-x-2">
          <Link
            href="/"
            className="text-xs font-semibold text-[#66717A] hover:text-[#172026] flex items-center transition-colors"
          >
            &larr; Control Room
          </Link>
          <span className="text-[#8A949C]">&bull;</span>
          <h1 className="text-sm font-bold text-[#172026]">Passenger ETA Portal</h1>
        </div>

        <div className="flex items-center space-x-2.5">
          {isRefreshing ? (
            <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-[#EBF3FC] text-[#2563A8] border border-[#BFDBFE]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2563A8] animate-ping" />
              <span>Updating Telemetry...</span>
            </span>
          ) : trainDetails?.current_state?.source === "external_api" ? (
            <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-[#EBF7EE] text-[#168A55] border border-[#B4E2C1]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#168A55] animate-pulse" />
              <span>Live RailRadar Stream</span>
            </span>
          ) : (
            <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#168A55]" />
              <span>Simulator Mode</span>
            </span>
          )}
          <button
            type="button"
            onClick={() => fetchSelectedTrain(selectedTrainNumber)}
            className="text-[11px] text-[#66717A] hover:text-[#172026] border border-[#D9DEE3] rounded px-2 py-0.5 bg-[#FFFFFF] hover:bg-[#F0F3F5] transition-colors cursor-pointer"
            title="Refresh now"
          >
            ↻ Sync
          </button>
          <span className="text-[11px] text-[#8A949C] font-mono">
            IST {formatTime(lastRefreshed.toISOString())}
          </span>
        </div>
      </div>

      {/* ======================================================== */}
      {/* 1. TRAIN SEARCH PANEL */}
      {/* ======================================================== */}
      <div className="bg-[#FFFFFF] p-4 sm:p-5 rounded-lg border border-[#D9DEE3] shadow-2xs space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-[11px] font-bold uppercase tracking-wider text-[#66717A]">
            1. Search / Select Train
          </label>
          <span className="text-[11px] text-[#8A949C] font-mono">
            {trainsList.length} Active Trains Available
          </span>
        </div>

        {/* Dual Search & Direct Select Row */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
          {/* Quick Search Input */}
          <div className="md:col-span-6 relative">
            <input
              type="text"
              placeholder="Search train by number or name (e.g. 12302, 12951)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && searchQuery.trim()) {
                  setSelectedTrainNumber(searchQuery.trim());
                  setSearchQuery("");
                }
              }}
              className="w-full pl-8 pr-3 py-2 text-xs border border-[#D9DEE3] rounded-md bg-[#F8FAFB] focus:bg-[#FFFFFF] focus:border-[#2563A8] focus:outline-none transition-all font-mono"
            />
            <span className="absolute left-2.5 top-2.5 text-xs text-[#8A949C]">
              🔍
            </span>
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-2.5 text-xs text-[#8A949C] hover:text-[#172026]"
              >
                ✕
              </button>
            )}
          </div>

          {/* Direct Dropdown Select */}
          <div className="md:col-span-6">
            <select
              value={selectedTrainNumber}
              onChange={(e) => {
                setSelectedTrainNumber(e.target.value);
                setSearchQuery("");
              }}
              className="w-full px-3 py-2 bg-[#F8FAFB] border border-[#D9DEE3] rounded-md text-xs sm:text-[13px] font-semibold text-[#172026] focus:bg-[#FFFFFF] focus:border-[#2563A8] focus:outline-none transition-all font-mono"
            >
              {loadingList ? (
                <option>Loading active trains...</option>
              ) : (
                trainsList.map((t) => (
                  <option key={t.id} value={t.train_number}>
                    {t.train_number} &mdash; {t.name} ({t.train_type})
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        {/* Filtered suggestions list when typing */}
        {searchQuery.trim().length > 0 && (
          <div className="max-h-48 overflow-y-auto divide-y divide-[#D9DEE3]/70 border border-[#D9DEE3] rounded-md text-xs bg-[#FFFFFF] shadow-sm">
            {/* Live Search Option */}
            <button
              type="button"
              onClick={() => {
                setSelectedTrainNumber(searchQuery.trim());
                setSearchQuery("");
              }}
              className="w-full text-left px-3.5 py-2 bg-[#EBF3FC]/60 hover:bg-[#EBF3FC] text-xs font-mono font-semibold text-[#2563A8] flex items-center justify-between transition-colors"
            >
              <span>📡 Track Live Train #{searchQuery.trim()}</span>
              <span className="text-[10px] text-[#2563A8] font-bold uppercase bg-white px-2 py-0.5 rounded border border-[#BFDBFE]">
                Live Radar &rarr;
              </span>
            </button>

            {filteredTrains.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  setSelectedTrainNumber(t.train_number);
                  setSearchQuery("");
                }}
                className="w-full text-left px-3.5 py-2 hover:bg-[#F0F3F5] flex items-center justify-between transition-colors"
              >
                <span className="font-semibold text-[#172026]">
                  {t.train_number} &mdash; {t.name}
                </span>
                <span className="text-[11px] text-[#8A949C] font-mono">
                  {t.current_station ? `At ${t.current_station}` : "Active"}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Fast Switch Preset Buttons */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-[#D9DEE3]/70 text-xs">
          <span className="text-[10px] text-[#8A949C] uppercase font-bold mr-1">
            Quick Select:
          </span>
          {[
            { num: "12302", label: "12302 (HWH Rajdhani)" },
            { num: "12952", label: "12952 (Mumbai Rajdhani)" },
            { num: "12028", label: "12028 (Shatabdi Exp)" },
            { num: "12260", label: "12260 (Duronto Exp)" },
          ].map((item) => {
            const isCurrent = selectedTrainNumber === item.num;
            return (
              <button
                key={item.num}
                type="button"
                onClick={() => setSelectedTrainNumber(item.num)}
                className={`px-2.5 py-1 text-xs rounded border transition-all font-mono ${
                  isCurrent
                    ? "bg-[#172026] text-white border-[#172026] font-semibold shadow-2xs"
                    : "bg-[#F0F3F5] hover:bg-[#E2E8F0] text-[#66717A] border-[#D9DEE3]"
                }`}
              >
                #{item.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Loading Details State */}
      {loadingDetails && (
        <div className="space-y-3">
          <CardSkeleton />
        </div>
      )}

      {/* Error State */}
      {error && !loadingDetails && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-[#D64545] rounded-lg text-xs space-y-1">
          <div className="font-bold">Error Loading Train Information</div>
          <div>{error}</div>
          <button
            type="button"
            onClick={() => fetchSelectedTrain(selectedTrainNumber)}
            className="mt-1 text-xs font-semibold text-rose-900 underline cursor-pointer"
          >
            Tap to retry
          </button>
        </div>
      )}

      {/* Main Passenger Flow Content */}
      {trainDetails && !loadingDetails && (
        <div className="space-y-4">
          {/* ======================================================== */}
          {/* 2. TRAIN STATUS HEADER */}
          {/* ======================================================== */}
          <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 shadow-2xs">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <div className="flex items-center space-x-2.5">
                  <span className="font-mono font-bold text-lg text-[#172026]">
                    {trainDetails.train_number}
                  </span>
                  <span className="font-bold text-base text-[#172026]">
                    &mdash; {trainName}
                  </span>
                </div>
                <div className="text-xs text-[#66717A] mt-1 flex flex-wrap items-center gap-2">
                  <span>{trainDetails.train_type}</span>
                  <span>&bull;</span>
                  <span>Route #{trainDetails.route_id}</span>
                  <span>&bull;</span>
                  {trainDetails.current_state?.source === "external_api" ? (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#EBF7EE] text-[#168A55] border border-[#B4E2C1]">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#168A55] animate-pulse" />
                      <span>LIVE RADAR</span>
                    </span>
                  ) : (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
                      <span>SIMULATOR</span>
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2 self-start sm:self-auto">
                <StatusBadge status={trainDetails.current_state?.status} size="md" />
                <span
                  className={`inline-block px-2.5 py-1 text-xs font-mono font-bold border rounded ${delayInfo.colorClass}`}
                >
                  {delayInfo.text}
                </span>
              </div>
            </div>
          </div>

          {/* ======================================================== */}
          {/* 3. CURRENT LOCATION & 4. DESTINATION CORRIDOR STRIP */}
          {/* ======================================================== */}
          <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 sm:p-5 shadow-2xs space-y-3">
            <div className="text-[11px] font-bold uppercase tracking-wider text-[#66717A]">
              2. Journey Corridor &amp; Destination Selection
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center p-3.5 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3]">
              {/* CURRENT LOCATION */}
              <div className="md:col-span-5 space-y-1">
                <div className="text-[10px] font-bold uppercase tracking-wider text-[#8A949C]">
                  Current Location
                </div>
                <div className="text-sm sm:text-base font-mono font-bold text-[#172026]">
                  {currentStation}
                </div>
                <div className="text-xs text-[#66717A] font-mono flex items-center space-x-2">
                  <span>Speed: <strong>{trainDetails.current_state?.speed_kmh?.toFixed(0) ?? 0} km/h</strong></span>
                  <span>&bull;</span>
                  <span className="text-[#168A55] font-semibold">Active In Transit</span>
                </div>
              </div>

              {/* TRANSIT CONNECTOR */}
              <div className="hidden md:flex md:col-span-2 flex-col items-center justify-center text-center">
                <span className="text-xs text-[#2563A8] font-bold font-mono">
                  &rarr;&rarr;&rarr;
                </span>
                <span className="text-[10px] text-[#8A949C] font-mono mt-0.5">
                  {targetStationETA?.segments_ahead ?? 1} stop(s) ahead
                </span>
              </div>

              {/* DESTINATION SELECTION */}
              <div className="md:col-span-5 space-y-1">
                <div className="flex items-center justify-between">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#2563A8]">
                    Your Destination Station
                  </div>
                  {trainDetails.upcoming_stations && trainDetails.upcoming_stations.length > 1 && (
                    <span className="text-[10px] text-[#66717A]">
                      Select stop &darr;
                    </span>
                  )}
                </div>

                {trainDetails.upcoming_stations && trainDetails.upcoming_stations.length > 0 ? (
                  <select
                    value={selectedStationCode}
                    onChange={(e) => setSelectedStationCode(e.target.value)}
                    className="w-full px-3 py-2 bg-[#FFFFFF] border border-[#2563A8]/40 rounded-md text-xs sm:text-[13px] font-bold font-mono text-[#172026] focus:border-[#2563A8] focus:outline-none shadow-2xs"
                  >
                    {trainDetails.upcoming_stations.map((stn, idx) => (
                      <option key={stn.station_code} value={stn.station_code}>
                        {idx === 0 ? "👉 NEXT: " : "Upcoming: "}
                        {stn.station_code} &mdash; {stn.station_name || stn.station_code}{" "}
                        ({(stn.distance_to_go_km ?? 0).toFixed(0)} km away)
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="p-2 bg-[#FFFFFF] rounded border border-[#D9DEE3] text-xs text-[#66717A] text-center font-medium">
                    Train has reached its final destination.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ======================================================== */}
          {/* 5. EXPECTED ARRIVAL WINDOW (VISUAL CENTERPIECE) */}
          {/* ======================================================== */}
          {targetStationETA ? (
            <div className="bg-[#FFFFFF] rounded-lg border-2 border-[#2563A8]/30 shadow-xs p-5 sm:p-6 space-y-5">
              {/* Centerpiece Header Tag */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-3 border-b border-[#D9DEE3]">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#2563A8]">
                    Dynamic ETA Forecast
                  </div>
                  <h2 className="text-base sm:text-lg font-bold text-[#172026] mt-0.5">
                    Arrival at {targetStationETA.station_name || targetStationETA.station_code} ({targetStationETA.station_code})
                  </h2>
                </div>

                <div className="flex items-center space-x-2">
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-[#EBF3FC] text-[#2563A8] border border-[#BFDBFE]">
                    ML Chained Forecast
                  </span>
                  <span className="text-xs font-mono text-[#66717A]">
                    {(targetStationETA.distance_to_go_km ?? 0).toFixed(1)} km to go
                  </span>
                </div>
              </div>

              {/* Hero Stats Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                {/* 1. Expected Arrival Time */}
                <div className="p-4 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] text-center sm:text-left">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#66717A]">
                    Estimated Arrival (ML)
                  </div>
                  <div className="text-2xl sm:text-3xl font-extrabold font-mono text-[#2563A8] mt-1">
                    {formatTime(targetStationETA.ml_eta || targetStationETA.predicted_eta)}
                  </div>
                  <div className="text-[11px] text-[#66717A] mt-1 font-mono">
                    Indian Standard Time (IST)
                  </div>
                </div>

                {/* 2. Remaining Transit Countdown */}
                <div className="p-4 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] text-center sm:text-left">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#66717A]">
                    Estimated Transit Remaining
                  </div>
                  <div className="text-2xl sm:text-3xl font-extrabold font-mono text-[#172026] mt-1">
                    ~{(targetStationETA.predicted_remaining_minutes ?? 0).toFixed(0)} min
                  </div>
                  <div className="text-[11px] text-[#66717A] mt-1 font-mono">
                    {targetStationETA.segments_ahead} intermediate checkpoint(s)
                  </div>
                </div>

                {/* 3. Prediction Uncertainty Bounds */}
                <div className="p-4 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] text-center sm:text-left">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#66717A]">
                    Prediction Range
                  </div>
                  <div className="text-lg sm:text-xl font-bold font-mono text-[#172026] mt-1.5">
                    {formatTime(targetStationETA.confidence_lower_bound)} &ndash; {formatTime(targetStationETA.confidence_upper_bound)}
                  </div>
                  <div className="text-[11px] text-[#66717A] mt-1 font-mono">
                    Window: &plusmn;{(targetStationETA.confidence_range?.margin_minutes ?? 0).toFixed(1)} min
                  </div>
                </div>
              </div>

              {/* Visual Uncertainty Range Bar */}
              <div className="pt-2">
                <UncertaintyRangeBar
                  mlEta={targetStationETA.ml_eta || targetStationETA.predicted_eta}
                  lowerBound={targetStationETA.confidence_lower_bound}
                  upperBound={targetStationETA.confidence_upper_bound}
                  marginMinutes={targetStationETA.confidence_range?.margin_minutes}
                  segmentsAhead={targetStationETA.segments_ahead}
                  variant="card"
                />
              </div>
            </div>
          ) : null}

          {/* ======================================================== */}
          {/* 6. SCHEDULED / BASELINE / ML FORECAST COMPARISON */}
          {/* ======================================================== */}
          {targetStationETA && (
            <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 sm:p-5 shadow-2xs space-y-3">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#66717A]">
                3. Forecast Comparison: Scheduled vs Baseline vs Machine Learning
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* 1. Timetable Scheduled */}
                <div className="p-3.5 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]">
                      SCHED
                    </span>
                    <span className="text-[11px] text-[#8A949C]">Timetable</span>
                  </div>
                  <div className="text-lg font-bold font-mono text-[#172026]">
                    {formatTime(targetStationETA.scheduled_eta)}
                  </div>
                  <p className="text-[11px] text-[#66717A] leading-relaxed">
                    Official published Indian Railways timetable arrival. Assumes zero line disruption or operational halts.
                  </p>
                </div>

                {/* 2. Baseline Extrapolation */}
                <div className="p-3.5 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#F5F7F8] text-[#172026] border border-[#D9DEE3]">
                      BASELINE
                    </span>
                    <span className="text-[11px] text-[#8A949C]">Heuristic</span>
                  </div>
                  <div className="text-lg font-bold font-mono text-[#172026]">
                    {targetStationETA.baseline_eta ? formatTime(targetStationETA.baseline_eta) : "--:--"}
                  </div>
                  <p className="text-[11px] text-[#66717A] leading-relaxed">
                    Linear extrapolation based on current recorded delay (+{currentDelayMinutes.toFixed(1)}m) and sectional speed.
                  </p>
                </div>

                {/* 3. ML Dynamic Prediction */}
                <div className="p-3.5 bg-[#EBF3FC] rounded-lg border border-[#BFDBFE] space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-[#2563A8] text-white">
                      ML DYNAMIC
                    </span>
                    <span className="text-[11px] text-[#2563A8] font-semibold">Recommended</span>
                  </div>
                  <div className="text-lg font-bold font-mono text-[#2563A8]">
                    {formatTime(targetStationETA.ml_eta || targetStationETA.predicted_eta)}
                  </div>
                  <p className="text-[11px] text-[#2563A8]/80 leading-relaxed">
                    Chained XGBoost forecast trained on historical track congestion, halt dwell variability, and recovery margins.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ======================================================== */}
          {/* 7. REFRESH / FULL ROUTE ACTIONS */}
          {/* ======================================================== */}
          <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 shadow-2xs flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="text-xs text-[#66717A] text-center sm:text-left">
              <span>Looking for complete station schedule, intermediate halt durations, or segment speeds?</span>
            </div>

            <div className="flex items-center space-x-3 w-full sm:w-auto">
              <button
                type="button"
                onClick={() => fetchSelectedTrain(selectedTrainNumber)}
                className="flex-1 sm:flex-initial py-2 px-4 rounded-md bg-[#F0F3F5] hover:bg-[#E2E8F0] border border-[#D9DEE3] text-[#172026] font-semibold text-xs transition-colors flex items-center justify-center space-x-1.5 cursor-pointer shadow-2xs"
              >
                <span>↻ Refresh Status</span>
              </button>

              <Link
                href={`/trains/${encodeURIComponent(trainDetails.train_number)}`}
                className="flex-1 sm:flex-initial py-2 px-4 rounded-md bg-[#172026] hover:bg-[#2563A8] text-white font-semibold text-xs transition-colors flex items-center justify-center space-x-1 shadow-2xs"
              >
                <span>Full Route &amp; All Stops &rarr;</span>
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PassengerLookupPage() {
  return (
    <Suspense
      fallback={
        <div className="py-16 text-center text-xs text-[#8A949C]">
          Loading Passenger Lookup...
        </div>
      }
    >
      <PassengerViewContent />
    </Suspense>
  );
}
