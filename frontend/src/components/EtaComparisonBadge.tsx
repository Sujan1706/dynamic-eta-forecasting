"use client";

import { formatTime, formatDateTime } from "@/lib/api";

export type EtaType = "scheduled" | "baseline" | "ml";

interface EtaComparisonBadgeProps {
  type: EtaType;
  time: string | null | undefined;
  showDate?: boolean;
  uncertaintyMargin?: number | null;
  size?: "sm" | "md" | "lg";
  className?: string;
}

/**
 * EtaComparisonBadge Component
 *
 * Implements strict operational visual distinction between the 3 ETA tiers:
 * 1. Scheduled ETA  -> Neutral muted border (Static published timetable)
 * 2. Baseline ETA   -> Neutral font (Speed & distance extrapolation)
 * 3. ML ETA         -> Intelligent ML Forecast with uncertainty window
 */
export default function EtaComparisonBadge({
  type,
  time,
  showDate = false,
  uncertaintyMargin,
  size = "md",
  className = "",
}: EtaComparisonBadgeProps) {
  if (!time) {
    return <span className="text-[#8A949C] font-mono text-xs">--:--</span>;
  }

  const formattedTime = formatTime(time);
  const formattedDate = showDate ? formatDateTime(time).split(",")[0] : null;

  const configs = {
    scheduled: {
      tag: "SCHED",
      fullName: "Timetable Schedule",
      color: "bg-[#F0F3F5] text-[#66717A] border-[#D9DEE3]",
      timeColor: "text-[#66717A]",
      tagColor: "bg-[#FFFFFF] text-[#66717A] border-[#D9DEE3]",
    },
    baseline: {
      tag: "BASELINE",
      fullName: "Speed Heuristic",
      color: "bg-[#F5F7F8] text-[#172026] border-[#D9DEE3]",
      timeColor: "text-[#172026] font-medium",
      tagColor: "bg-[#FFFFFF] text-[#66717A] border-[#D9DEE3]",
    },
    ml: {
      tag: "ML ETA",
      fullName: "XGBoost Forecast",
      color: "bg-[#EBF3FC] text-[#2563A8] border-[#BFDBFE]",
      timeColor: "text-[#2563A8] font-bold",
      tagColor: "bg-[#2563A8] text-white font-semibold",
    },
  };

  const cfg = configs[type];

  if (size === "sm") {
    return (
      <div className={`inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border text-xs ${cfg.color} ${className}`}>
        <span className={`px-1 py-0.2 rounded text-[9px] font-mono font-bold uppercase tracking-wider border ${cfg.tagColor}`}>
          {cfg.tag}
        </span>
        <span className={`font-mono text-xs ${cfg.timeColor}`}>{formattedTime}</span>
        {uncertaintyMargin !== undefined && uncertaintyMargin !== null && (
          <span className="text-[10px] text-[#2563A8] font-sans">(&plusmn;{uncertaintyMargin.toFixed(0)}m)</span>
        )}
      </div>
    );
  }

  if (size === "lg") {
    return (
      <div className={`p-3 rounded-lg border ${cfg.color} space-y-1.5 ${className}`}>
        <div className="flex items-center justify-between text-xs font-semibold">
          <span className="uppercase tracking-wider text-[#66717A]">{cfg.fullName}</span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${cfg.tagColor}`}>
            {cfg.tag}
          </span>
        </div>
        <div className={`text-2xl font-mono tracking-tight ${cfg.timeColor}`}>
          {formattedTime}
        </div>
        {formattedDate && (
          <div className="text-[11px] text-[#8A949C] font-sans">{formattedDate}</div>
        )}
        {uncertaintyMargin !== undefined && uncertaintyMargin !== null && (
          <div className="text-xs text-[#2563A8] font-medium">
            Uncertainty: &plusmn;{uncertaintyMargin.toFixed(1)} min
          </div>
        )}
      </div>
    );
  }

  // Default 'md' size
  return (
    <div className={`inline-block ${className}`}>
      <div className="flex items-center space-x-1.5">
        <span className={`px-1.5 py-0.2 rounded text-[9px] font-mono font-bold uppercase tracking-wider border ${cfg.tagColor}`}>
          {cfg.tag}
        </span>
        <span className={`font-mono text-sm ${cfg.timeColor}`}>{formattedTime}</span>
      </div>
      {formattedDate && (
        <div className="text-[10px] text-[#8A949C] font-mono mt-0.5">{formattedDate}</div>
      )}
      {uncertaintyMargin !== undefined && uncertaintyMargin !== null && (
        <div className="text-[10px] text-[#2563A8] font-medium mt-0.5">
          &plusmn;{uncertaintyMargin.toFixed(0)}m window
        </div>
      )}
    </div>
  );
}
