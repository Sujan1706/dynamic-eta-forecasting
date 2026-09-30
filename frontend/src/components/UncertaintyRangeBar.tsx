"use client";

import { formatTime } from "@/lib/api";

interface UncertaintyRangeBarProps {
  mlEta: string | null | undefined;
  lowerBound: string | null | undefined;
  upperBound: string | null | undefined;
  marginMinutes?: number | null;
  segmentsAhead?: number | null;
  variant?: "table" | "card" | "inline";
  theme?: "light" | "dark";
  showHorizonLabel?: boolean;
}

/**
 * UncertaintyRangeBar Component
 *
 * Visually communicates prediction uncertainty window [Lower Bound | ML ETA | Upper Bound].
 * Accurately demonstrates how prediction uncertainty widens as lookahead horizon increases
 * (short-horizon predictions have narrower ranges; longer-horizon predictions have wider ranges).
 *
 * Clean, restrained, operational light control room styling.
 */
export default function UncertaintyRangeBar({
  mlEta,
  lowerBound,
  upperBound,
  marginMinutes,
  segmentsAhead = 1,
  variant = "table",
  theme = "light",
  showHorizonLabel = true,
}: UncertaintyRangeBarProps) {
  if (!mlEta) {
    return <span className={`text-xs font-mono ${theme === "dark" ? "text-slate-500" : "text-[#8A949C]"}`}>--:--</span>;
  }

  const segs = Math.max(1, segmentsAhead || 1);
  const margin = marginMinutes ?? 5.0 * Math.sqrt(segs);

  // Proportional bar width representing the uncertainty window size
  const visualSpanPercent = Math.min(100, Math.max(26, 20 + segs * 13));

  // Horizon descriptive classification
  const horizonBadge =
    segs === 1
      ? {
          text: "1 Stop Ahead (Narrow)",
          lightColor: "text-[#168A55] bg-[#EBF7EE] border-[#B4E2C1]",
          barColor: "bg-[#168A55]",
        }
      : segs <= 3
      ? {
          text: `${segs} Stops Ahead (Moderate)`,
          lightColor: "text-[#2563A8] bg-[#EBF3FC] border-[#BFDBFE]",
          barColor: "bg-[#2563A8]",
        }
      : {
          text: `${segs} Stops Ahead (Wide Window)`,
          lightColor: "text-[#B77900] bg-amber-50 border-amber-200",
          barColor: "bg-[#B77900]",
        };

  const lowerTime = lowerBound ? formatTime(lowerBound) : "--:--";
  const mlTime = formatTime(mlEta);
  const upperTime = upperBound ? formatTime(upperBound) : "--:--";

  // ---------------------------------------------------------------------------
  // VARIANT: CARD (Passenger View & Detail Page)
  // ---------------------------------------------------------------------------
  if (variant === "card") {
    return (
      <div className="bg-[#FFFFFF] p-4 rounded-lg border border-[#D9DEE3] space-y-3 shadow-2xs">
        {/* Header with Uncertainty Disclaimer */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 pb-2 border-b border-[#D9DEE3]/70">
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-[#172026] flex items-center space-x-1.5">
              <span>🎯</span>
              <span>Expected Arrival Window</span>
            </div>
            <p className="text-[11px] text-[#66717A] mt-0.5">
              Prediction uncertainty range &bull; Widens with lookahead distance
            </p>
          </div>
          <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${horizonBadge.lightColor}`}>
            {horizonBadge.text} (&plusmn;{margin.toFixed(1)}m)
          </span>
        </div>

        {/* 3 Prominent Arrival Timestamps */}
        <div className="grid grid-cols-3 gap-2 text-center pt-1 font-mono">
          {/* Lower Bound */}
          <div className="p-2 bg-[#F8FAFB] rounded border border-[#D9DEE3]">
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8A949C]">
              Lower Bound
            </div>
            <div className="text-sm sm:text-base font-bold text-[#172026] mt-0.5">
              {lowerTime}
            </div>
            <div className="text-[9px] text-[#8A949C] font-sans">Earliest expected</div>
          </div>

          {/* ML ETA (Highlighted Center) */}
          <div className="p-2 bg-[#EBF3FC] text-[#2563A8] rounded border border-[#BFDBFE]">
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#2563A8] flex items-center justify-center space-x-1">
              <span>★</span>
              <span>ML ETA</span>
            </div>
            <div className="text-base sm:text-lg font-bold text-[#2563A8] mt-0.5 tracking-tight">
              {mlTime}
            </div>
            <div className="text-[9px] text-[#2563A8] font-sans font-medium">Most probable</div>
          </div>

          {/* Upper Bound */}
          <div className="p-2 bg-[#F8FAFB] rounded border border-[#D9DEE3]">
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8A949C]">
              Upper Bound
            </div>
            <div className="text-sm sm:text-base font-bold text-[#172026] mt-0.5">
              {upperTime}
            </div>
            <div className="text-[9px] text-[#8A949C] font-sans">Latest expected</div>
          </div>
        </div>

        {/* Visual Uncertainty Range Track */}
        <div className="space-y-1 pt-1">
          <div className="flex justify-between text-[10px] font-mono text-[#66717A]">
            <span>&larr; Lower ({lowerTime})</span>
            <span className="font-sans font-semibold text-[#2563A8]">
              Span: &plusmn;{margin.toFixed(1)}m window
            </span>
            <span>Upper ({upperTime}) &rarr;</span>
          </div>

          {/* The Visual Horizon Bar */}
          <div className="relative w-full h-2.5 bg-[#F0F3F5] rounded-full flex items-center justify-center p-0.5 border border-[#D9DEE3]">
            <div
              className={`h-full ${horizonBadge.barColor} rounded-full flex items-center justify-between px-1 transition-all duration-500`}
              style={{ width: `${visualSpanPercent}%` }}
              title={`Uncertainty window: ${lowerTime} to ${upperTime} (±${margin.toFixed(1)}m, ${segs} stop${segs > 1 ? "s" : ""} ahead)`}
            >
              <div className="w-1 h-1 rounded-full bg-white" />
              <div className="w-1.5 h-1.5 rounded-full bg-white ring-1 ring-[#172026]" />
              <div className="w-1 h-1 rounded-full bg-white" />
            </div>
          </div>

          <div className="text-[9px] text-[#8A949C] text-center italic pt-0.5">
            * Uncalibrated prediction uncertainty heuristic based on route distance and multi-station horizon.
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // VARIANT: TABLE ROW (Compact representation in operational tables)
  // ---------------------------------------------------------------------------
  return (
    <div className="space-y-1 py-0.5 min-w-[200px] max-w-[250px]">
      {/* 3 Explicit Numerical Timestamps */}
      <div className="flex items-center justify-between text-xs font-mono">
        <span
          className="text-[11px] text-[#8A949C]"
          title={`Lower bound: ${lowerTime}`}
        >
          {lowerTime}
        </span>
        <span
          className="font-bold px-1.5 py-0.2 rounded text-xs text-[#2563A8] bg-[#EBF3FC] border border-[#BFDBFE]"
          title={`ML Forecast: ${mlTime}`}
        >
          {mlTime}
        </span>
        <span
          className="text-[11px] text-[#8A949C]"
          title={`Upper bound: ${upperTime}`}
        >
          {upperTime}
        </span>
      </div>

      {/* Visual Uncertainty Span Bar */}
      <div
        className="relative w-full h-1.5 rounded-full flex items-center justify-center border overflow-hidden bg-[#F0F3F5] border-[#D9DEE3]"
        title={`Uncertainty window: ${lowerTime} to ${upperTime} (±${margin.toFixed(1)}m, ${segs} stop${segs > 1 ? "s" : ""} ahead)`}
      >
        <div
          className={`h-full rounded-full transition-all duration-300 ${horizonBadge.barColor}`}
          style={{ width: `${visualSpanPercent}%` }}
        />
      </div>

      {/* Horizon Label & Margin */}
      {showHorizonLabel && (
        <div className="flex justify-between items-center text-[10px] font-mono text-[#8A949C]">
          <span>&plusmn;{margin.toFixed(1)}m window</span>
          <span className="text-[9px] text-[#66717A]">
            {segs === 1 ? "1 stop (narrow)" : `${segs} stops`}
          </span>
        </div>
      )}
    </div>
  );
}
