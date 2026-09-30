"use client";

interface StatusBadgeProps {
  status: string | null | undefined;
  size?: "sm" | "md";
}

/**
 * StatusBadge Component
 *
 * Operational railway status indicator with restrained semantic coloring:
 * - Running: #168A55
 * - Halted: #D64545
 * - Arrived/Other: #66717A
 */
export function StatusBadge({ status = "RUNNING", size = "md" }: StatusBadgeProps) {
  const normStatus = (status || "RUNNING").toUpperCase();

  const isRunning = normStatus === "RUNNING";
  const isHalted = normStatus === "HALTED";
  const isCompleted = normStatus === "COMPLETED";

  const paddingClass = size === "sm" ? "px-1.5 py-0.2 text-[10px]" : "px-2 py-0.5 text-xs";

  if (isHalted) {
    return (
      <span
        className={`inline-flex items-center space-x-1.5 font-semibold uppercase tracking-wider rounded border bg-[#FDF2F2] text-[#D64545] border-[#F5C2C2] ${paddingClass}`}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#D64545]" />
        <span>Halted</span>
      </span>
    );
  }

  if (isRunning) {
    return (
      <span
        className={`inline-flex items-center space-x-1.5 font-semibold uppercase tracking-wider rounded border bg-[#EBF7EE] text-[#168A55] border-[#B4E2C1] ${paddingClass}`}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#168A55]" />
        <span>Running</span>
      </span>
    );
  }

  if (isCompleted) {
    return (
      <span
        className={`inline-flex items-center space-x-1.5 font-medium uppercase tracking-wider rounded border bg-[#F0F3F5] text-[#66717A] border-[#D9DEE3] ${paddingClass}`}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#8A949C]" />
        <span>Arrived</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center space-x-1.5 font-medium uppercase tracking-wider rounded border bg-[#F0F3F5] text-[#66717A] border-[#D9DEE3] ${paddingClass}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-[#8A949C]" />
      <span>{normStatus}</span>
    </span>
  );
}

interface DataSourceBadgeProps {
  source?: string | null;
  mode?: string | null;
  isFallback?: boolean;
}

/**
 * DataSourceBadge Component
 *
 * Displays telemetry origin (Live Railway API vs Simulator Engine).
 */
export function DataSourceBadge({ source, mode, isFallback }: DataSourceBadgeProps) {
  const isLive = mode === "LIVE_API" || source === "external_api";

  if (isLive) {
    return (
      <span
        className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium uppercase tracking-wider bg-[#F5F3FF] text-[#6D28D9] border border-[#DDD6FE]"
        title="Live telemetry from external RailRadar API"
      >
        <span>LIVE</span>
        {isFallback && (
          <span className="ml-1 text-[9px] text-[#B77900] font-bold" title="Fell back to simulator">
            (FB)
          </span>
        )}
      </span>
    );
  }

  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium uppercase tracking-wider bg-[#F0F3F5] text-[#66717A] border border-[#D9DEE3]"
      title="Synthesized operational trajectory from backend physics simulator"
    >
      <span>SIMULATOR</span>
      {isFallback && (
        <span className="ml-1 text-[9px] text-[#B77900] font-bold" title="Fell back to simulator">
          (FB)
        </span>
      )}
    </span>
  );
}
