"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api, formatDateTime } from "@/lib/api";
import { ModelMetricsResponse } from "@/types/api";
import { CardSkeleton } from "@/components/LoadingSkeleton";

function formatMins(val: number | null | undefined, precision = 2): string {
  if (val === null || val === undefined || isNaN(val)) return "--";
  return `${val.toFixed(precision)} min`;
}

/**
 * Metric Comparison Card comparing Baseline Heuristic vs ML Model
 */
function MetricComparisonCard({
  title,
  subtitle,
  baselineVal,
  mlVal,
  unit = "min",
  lowerIsBetter = true,
}: {
  title: string;
  subtitle: string;
  baselineVal: number | null | undefined;
  mlVal: number | null | undefined;
  unit?: string;
  lowerIsBetter?: boolean;
}) {
  const hasData = baselineVal !== null && baselineVal !== undefined && mlVal !== null && mlVal !== undefined;
  const pctImprovement = hasData && baselineVal! !== 0 ? ((baselineVal! - mlVal!) / baselineVal!) * 100 : 0;
  const isMlWinner = lowerIsBetter ? mlVal! < baselineVal! : mlVal! > baselineVal!;

  const maxBar = hasData ? Math.max(baselineVal!, mlVal!, 1) : 1;
  const baselineWidth = hasData ? (baselineVal! / maxBar) * 100 : 0;
  const mlWidth = hasData ? (mlVal! / maxBar) * 100 : 0;

  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 shadow-2xs space-y-3">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-sm font-bold text-[#172026] tracking-tight">{title}</h3>
          <p className="text-xs text-[#66717A] mt-0.5">{subtitle}</p>
        </div>
        {hasData && (
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${
              isMlWinner
                ? "bg-[#EBF7EE] text-[#168A55] border-[#B4E2C1]"
                : "bg-amber-50 text-[#B77900] border-amber-200"
            }`}
          >
            {isMlWinner ? `Lower Error: ML (-${Math.abs(pctImprovement).toFixed(1)}%)` : "Lower Error: Baseline"}
          </span>
        )}
      </div>

      {/* Numerical Values */}
      <div className="grid grid-cols-2 gap-2.5 pt-1">
        <div className="p-3 bg-[#F8FAFB] rounded border border-[#D9DEE3]">
          <div className="text-[10px] font-semibold text-[#66717A] uppercase tracking-wider">
            Baseline Heuristic
          </div>
          <div className="text-lg font-bold font-mono text-[#172026] mt-1">
            {formatMins(baselineVal)}
          </div>
          <div className="text-[10px] text-[#8A949C] mt-0.5">Static timetable + delay</div>
        </div>

        <div className="p-3 bg-[#EBF3FC]/60 rounded border border-[#BFDBFE]">
          <div className="text-[10px] font-semibold text-[#2563A8] uppercase tracking-wider flex items-center justify-between">
            <span>XGBoost ML</span>
            <span className="text-[10px] font-bold text-[#168A55] font-mono">
              {pctImprovement > 0 ? `-${pctImprovement.toFixed(1)}%` : ""}
            </span>
          </div>
          <div className="text-lg font-bold font-mono text-[#2563A8] mt-1">
            {formatMins(mlVal)}
          </div>
          <div className="text-[10px] text-[#2563A8] mt-0.5">Chained gradient boosting</div>
        </div>
      </div>

      {/* Visual Relative Bar Comparison */}
      {hasData && (
        <div className="space-y-1.5 pt-1 text-xs">
          <div>
            <div className="flex justify-between text-[10px] text-[#66717A] mb-0.5 font-mono">
              <span>Baseline error</span>
              <span>{baselineVal!.toFixed(2)} {unit}</span>
            </div>
            <div className="w-full h-1.5 bg-[#F0F3F5] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#8A949C] rounded-full"
                style={{ width: `${baselineWidth}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[10px] text-[#2563A8] font-semibold mb-0.5 font-mono">
              <span>ML forecast error</span>
              <span>{mlVal!.toFixed(2)} {unit}</span>
            </div>
            <div className="w-full h-1.5 bg-[#F0F3F5] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#2563A8] rounded-full"
                style={{ width: `${mlWidth}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ModelPerformancePage() {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const fetchMetrics = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const res = await api.getModelMetrics();
      setMetrics(res);
      setLastRefreshed(new Date());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load model metrics";
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const overall = metrics?.overall;
  const h1 = metrics?.metrics_by_horizon?.["1_station_ahead"];
  const h3 = metrics?.metrics_by_horizon?.["3_stations_ahead"];
  const h5 = metrics?.metrics_by_horizon?.["5_stations_ahead"];
  const dNone = metrics?.disruption_breakdown?.none ?? metrics?.metrics_by_disruption_status?.["no_disruption"] ?? metrics?.metrics_by_disruption_status?.["none"] ?? metrics?.metrics_by_disruption_status?.["nominal"];
  const dWith = metrics?.disruption_breakdown?.with_disruption ?? metrics?.metrics_by_disruption_status?.["with_disruption"] ?? metrics?.metrics_by_disruption_status?.["disrupted"];
  const datasetInfo = metrics?.dataset_info;
  const summary = metrics?.summary;

  return (
    <div className="w-full max-w-[1560px] mx-auto space-y-6 pb-12">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3 border-b border-[#D9DEE3]">
        <div>
          <div className="flex items-center space-x-2">
            <Link
              href="/"
              className="text-xs font-medium text-[#66717A] hover:text-[#172026] mr-2 transition-colors"
            >
              &larr; Control Room
            </Link>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#172026] mt-1">
            Model Evaluation & Performance Benchmarks
          </h1>
          <p className="text-xs text-[#66717A] mt-0.5">
            Empirical evaluation comparing XGBoost dynamic chained forecasts against baseline speed heuristic.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {lastRefreshed && (
            <span className="text-[11px] font-mono text-[#8A949C]">
              Refreshed: {formatDateTime(lastRefreshed.toISOString())}
            </span>
          )}
          <button
            type="button"
            onClick={() => fetchMetrics(true)}
            disabled={refreshing || loading}
            className="inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-md border border-[#D9DEE3] bg-[#FFFFFF] hover:bg-[#F0F3F5] text-[#172026] shadow-2xs transition-colors disabled:opacity-50 cursor-pointer"
          >
            <span className={`mr-1.5 ${refreshing ? "animate-spin" : ""}`}>
              ↻
            </span>
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Loading Skeleton */}
      {loading && !metrics && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <CardSkeleton />
            <CardSkeleton />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </div>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-[#D64545] rounded-lg text-xs space-y-1">
          <div className="font-bold">Error Loading Evaluation Metrics</div>
          <div>{error}</div>
          <div className="pt-1">
            <button
              type="button"
              onClick={() => fetchMetrics(false)}
              className="px-2.5 py-1 bg-[#D64545] hover:bg-rose-700 text-white rounded text-xs font-semibold"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Unavailable File Warning */}
      {metrics && !metrics.is_available && (
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-900 space-y-1">
          <div className="font-bold text-xs flex items-center space-x-1.5">
            <span>⚠️</span>
            <span>Evaluation Results File Not Found</span>
          </div>
          <p className="text-xs text-amber-800">
            {metrics.message || "Model metrics file is not present. Run evaluate_model.py in backend to generate evaluation results."}
          </p>
        </div>
      )}

      {metrics && metrics.is_available && (
        <>
          {/* SECTION 1: OVERALL BENCHMARKING */}
          <section className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 pb-1 border-b border-[#D9DEE3]">
              <div>
                <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
                  <span>🏆</span>
                  <span>1. Overall Model Performance</span>
                </h2>
                <p className="text-xs text-[#66717A]">
                  Aggregate error metrics computed over {(metrics.test_samples ?? overall?.sample_count)?.toLocaleString() || "--"} test horizon evaluations.
                </p>
              </div>

              {overall?.percentage_improvement !== undefined && (
                <div className="text-xs font-semibold text-[#168A55] bg-[#EBF7EE] px-2.5 py-0.5 rounded border border-[#B4E2C1] font-mono">
                  Overall Error Reduction: <strong>{overall.percentage_improvement.toFixed(1)}%</strong> ({formatMins(overall.absolute_diff_mae)} MAE)
                </div>
              )}
            </div>

            {/* 2 Key Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <MetricComparisonCard
                title="Mean Absolute Error (MAE)"
                subtitle="Average absolute magnitude of arrival time prediction error across test horizons"
                baselineVal={overall?.baseline_mae ?? metrics.baseline_mae}
                mlVal={overall?.ml_mae ?? metrics.ml_mae}
                unit="min"
                lowerIsBetter={true}
              />

              <MetricComparisonCard
                title="Root Mean Squared Error (RMSE)"
                subtitle="Error metric penalizing large tail forecasting errors and disruption outliers"
                baselineVal={overall?.baseline_rmse ?? metrics.baseline_rmse}
                mlVal={overall?.ml_rmse ?? metrics.ml_rmse}
                unit="min"
                lowerIsBetter={true}
              />
            </div>
          </section>

          {/* SECTION 2: PREDICTION HORIZON BREAKDOWN (1, 3, 5 Stations) */}
          <section className="space-y-3">
            <div className="pb-1 border-b border-[#D9DEE3]">
              <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
                <span>🎯</span>
                <span>2. Prediction Horizon Breakdown</span>
              </h2>
              <p className="text-xs text-[#66717A] mt-0.5">
                Evaluates forecasting accuracy as lookahead distance increases: 1 Station Ahead (Local), 3 Stations (Corridor), and 5 Stations (Terminus).
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* 1 Station Ahead */}
              <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-3.5 shadow-2xs space-y-2.5">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-1.5 py-0.2 text-[10px] font-bold rounded bg-[#F0F3F5] text-[#172026] font-mono border border-[#D9DEE3]">
                      HORIZON 1
                    </span>
                    <h3 className="font-bold text-xs sm:text-sm text-[#172026] mt-1">1 Station Ahead</h3>
                    <p className="text-[11px] text-[#66717A]">Immediate next upcoming stop</p>
                  </div>
                  <span className="text-[11px] font-mono text-[#8A949C]">
                    {h1?.sample_count?.toLocaleString() || "--"} samples
                  </span>
                </div>

                <div className="space-y-1.5 pt-2 border-t border-[#D9DEE3]/70 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-[#66717A]">Baseline MAE:</span>
                    <strong className="font-mono text-[#172026]">{formatMins(h1?.baseline_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML MAE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h1?.ml_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center pt-1 border-t border-[#D9DEE3]/50">
                    <span className="text-[#66717A]">Baseline RMSE:</span>
                    <strong className="font-mono text-[#66717A]">{formatMins(h1?.baseline_rmse)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML RMSE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h1?.ml_rmse)}</strong>
                  </div>
                </div>

                <div className="pt-2 border-t border-[#D9DEE3]/70 flex items-center justify-between text-xs">
                  <span className="text-[#66717A] font-medium">Error Reduction:</span>
                  <span className="font-bold text-[#168A55] font-mono bg-[#EBF7EE] px-2 py-0.5 rounded border border-[#B4E2C1]">
                    +{h1?.percentage_improvement?.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* 3 Stations Ahead */}
              <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-3.5 shadow-2xs space-y-2.5">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-1.5 py-0.2 text-[10px] font-bold rounded bg-[#F0F3F5] text-[#172026] font-mono border border-[#D9DEE3]">
                      HORIZON 3
                    </span>
                    <h3 className="font-bold text-xs sm:text-sm text-[#172026] mt-1">3 Stations Ahead</h3>
                    <p className="text-[11px] text-[#66717A]">Intermediate corridor checkpoints</p>
                  </div>
                  <span className="text-[11px] font-mono text-[#8A949C]">
                    {h3?.sample_count?.toLocaleString() || "--"} samples
                  </span>
                </div>

                <div className="space-y-1.5 pt-2 border-t border-[#D9DEE3]/70 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-[#66717A]">Baseline MAE:</span>
                    <strong className="font-mono text-[#172026]">{formatMins(h3?.baseline_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML MAE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h3?.ml_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center pt-1 border-t border-[#D9DEE3]/50">
                    <span className="text-[#66717A]">Baseline RMSE:</span>
                    <strong className="font-mono text-[#66717A]">{formatMins(h3?.baseline_rmse)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML RMSE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h3?.ml_rmse)}</strong>
                  </div>
                </div>

                <div className="pt-2 border-t border-[#D9DEE3]/70 flex items-center justify-between text-xs">
                  <span className="text-[#66717A] font-medium">Error Reduction:</span>
                  <span className="font-bold text-[#168A55] font-mono bg-[#EBF7EE] px-2 py-0.5 rounded border border-[#B4E2C1]">
                    +{h3?.percentage_improvement?.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* 5 Stations Ahead */}
              <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-3.5 shadow-2xs space-y-2.5">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-1.5 py-0.2 text-[10px] font-bold rounded bg-[#F0F3F5] text-[#172026] font-mono border border-[#D9DEE3]">
                      HORIZON 5
                    </span>
                    <h3 className="font-bold text-xs sm:text-sm text-[#172026] mt-1">5 Stations Ahead</h3>
                    <p className="text-[11px] text-[#66717A]">Distant terminus horizon</p>
                  </div>
                  <span className="text-[11px] font-mono text-[#8A949C]">
                    {h5?.sample_count?.toLocaleString() || "--"} samples
                  </span>
                </div>

                <div className="space-y-1.5 pt-2 border-t border-[#D9DEE3]/70 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-[#66717A]">Baseline MAE:</span>
                    <strong className="font-mono text-[#172026]">{formatMins(h5?.baseline_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML MAE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h5?.ml_mae)}</strong>
                  </div>
                  <div className="flex justify-between items-center pt-1 border-t border-[#D9DEE3]/50">
                    <span className="text-[#66717A]">Baseline RMSE:</span>
                    <strong className="font-mono text-[#66717A]">{formatMins(h5?.baseline_rmse)}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#2563A8] font-semibold">ML RMSE:</span>
                    <strong className="font-mono text-[#2563A8] font-bold">{formatMins(h5?.ml_rmse)}</strong>
                  </div>
                </div>

                <div className="pt-2 border-t border-[#D9DEE3]/70 flex items-center justify-between text-xs">
                  <span className="text-[#66717A] font-medium">Error Reduction:</span>
                  <span className="font-bold text-[#168A55] font-mono bg-[#EBF7EE] px-2 py-0.5 rounded border border-[#B4E2C1]">
                    +{h5?.percentage_improvement?.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Horizon Comparison Table */}
            <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs overflow-hidden">
              <div className="px-4 py-2.5 border-b border-[#D9DEE3] bg-[#FFFFFF] flex justify-between items-center">
                <span className="text-xs font-bold uppercase tracking-wider text-[#66717A]">
                  Horizon Error Progression Matrix
                </span>
                <span className="text-[11px] text-[#8A949C]">
                  Evaluation on test journeys
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#FFFFFF] text-[#66717A] font-semibold border-b border-[#D9DEE3]">
                    <tr>
                      <th className="px-3.5 py-2">Horizon</th>
                      <th className="px-3.5 py-2">Samples</th>
                      <th className="px-3.5 py-2">Baseline MAE</th>
                      <th className="px-3.5 py-2 text-[#2563A8]">ML MAE</th>
                      <th className="px-3.5 py-2">Baseline RMSE</th>
                      <th className="px-3.5 py-2 text-[#2563A8]">ML RMSE</th>
                      <th className="px-3.5 py-2 text-right">Error Reduction</th>
                      <th className="px-3.5 py-2 text-right">Lower Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#D9DEE3]/70">
                    {[
                      { name: "1 Station Ahead (Local)", data: h1 },
                      { name: "3 Stations Ahead (Corridor)", data: h3 },
                      { name: "5 Stations Ahead (Terminus)", data: h5 },
                    ].map((row, idx) => (
                      <tr key={idx} className="hover:bg-[#F8FAFB] transition-colors">
                        <td className="px-3.5 py-2.5 font-semibold text-[#172026]">{row.name}</td>
                        <td className="px-3.5 py-2.5 font-mono text-[#66717A]">{row.data?.sample_count?.toLocaleString() || "--"}</td>
                        <td className="px-3.5 py-2.5 font-mono text-[#172026]">{formatMins(row.data?.baseline_mae)}</td>
                        <td className="px-3.5 py-2.5 font-mono font-bold text-[#2563A8]">{formatMins(row.data?.ml_mae)}</td>
                        <td className="px-3.5 py-2.5 font-mono text-[#66717A]">{formatMins(row.data?.baseline_rmse)}</td>
                        <td className="px-3.5 py-2.5 font-mono font-bold text-[#2563A8]">{formatMins(row.data?.ml_rmse)}</td>
                        <td className="px-3.5 py-2.5 font-mono font-bold text-[#168A55] text-right">
                          {row.data?.percentage_improvement ? `+${row.data.percentage_improvement.toFixed(1)}%` : "--"}
                        </td>
                        <td className="px-3.5 py-2.5 text-right">
                          <span className="px-1.5 py-0.2 text-[10px] font-bold bg-[#EBF7EE] text-[#168A55] border border-[#B4E2C1] rounded">
                            {row.data?.winner || "ML"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          {/* SECTION 3: DISRUPTION OPERATIONAL BREAKDOWN */}
          <section className="space-y-3">
            <div className="pb-1 border-b border-[#D9DEE3]">
              <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
                <span>⚡</span>
                <span>3. Disruption Breakdown</span>
              </h2>
              <p className="text-xs text-[#66717A] mt-0.5">
                Measures predictive resilience under nominal running versus disrupted conditions (signal halts, speed restrictions, bottlenecks).
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* No Disruption Card */}
              <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 shadow-2xs space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center space-x-1.5">
                      <span className="w-2 h-2 rounded-full bg-[#168A55]" />
                      <span className="text-[10px] font-bold text-[#168A55] uppercase tracking-wide">
                        Normal Line Clear
                      </span>
                    </div>
                    <h3 className="font-bold text-sm text-[#172026] mt-0.5">No Disruption</h3>
                    <p className="text-xs text-[#66717A]">
                      Standard running conditions without active signal stops or caution orders.
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-[#66717A] bg-[#F0F3F5] px-2 py-0.5 rounded border border-[#D9DEE3]">
                    {dNone?.sample_count?.toLocaleString() || "--"} samples
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2.5 pt-1">
                  <div className="p-2.5 bg-[#F8FAFB] rounded border border-[#D9DEE3]">
                    <div className="text-[10px] text-[#66717A] font-medium">Baseline MAE</div>
                    <div className="text-base font-bold font-mono text-[#172026] mt-0.5">
                      {formatMins(dNone?.baseline_mae)}
                    </div>
                    <div className="text-[10px] text-[#8A949C] mt-0.5">
                      RMSE: {formatMins(dNone?.baseline_rmse)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-[#EBF7EE] rounded border border-[#B4E2C1]">
                    <div className="text-[10px] text-[#168A55] font-semibold flex justify-between">
                      <span>ML MAE</span>
                      <span>-{dNone?.percentage_improvement?.toFixed(1)}%</span>
                    </div>
                    <div className="text-base font-bold font-mono text-[#168A55] mt-0.5">
                      {formatMins(dNone?.ml_mae)}
                    </div>
                    <div className="text-[10px] text-[#168A55] mt-0.5">
                      RMSE: {formatMins(dNone?.ml_rmse)}
                    </div>
                  </div>
                </div>

                <div className="p-2.5 bg-[#F8FAFB] rounded text-xs text-[#66717A] border border-[#D9DEE3]">
                  <span className="font-semibold text-[#172026]">Result:</span> ML reduces prediction error by{" "}
                  <strong>{dNone?.percentage_improvement?.toFixed(1) ?? "--"}%</strong> under nominal line conditions.
                </div>
              </div>

              {/* With Disruption Card */}
              <div className="bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] p-4 shadow-2xs space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center space-x-1.5">
                      <span className="w-2 h-2 rounded-full bg-[#D64545]" />
                      <span className="text-[10px] font-bold text-[#D64545] uppercase tracking-wide">
                        Active Disruptions
                      </span>
                    </div>
                    <h3 className="font-bold text-sm text-[#172026] mt-0.5">With Disruption</h3>
                    <p className="text-xs text-[#66717A]">
                      Journeys experiencing active signal halts, track maintenance, or congestion bottlenecks.
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-[#66717A] bg-[#F0F3F5] px-2 py-0.5 rounded border border-[#D9DEE3]">
                    {dWith?.sample_count?.toLocaleString() || "--"} samples
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2.5 pt-1">
                  <div className="p-2.5 bg-[#F8FAFB] rounded border border-[#D9DEE3]">
                    <div className="text-[10px] text-[#66717A] font-medium">Baseline MAE</div>
                    <div className="text-base font-bold font-mono text-[#172026] mt-0.5">
                      {formatMins(dWith?.baseline_mae)}
                    </div>
                    <div className="text-[10px] text-[#8A949C] mt-0.5">
                      RMSE: {formatMins(dWith?.baseline_rmse)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-[#FDF2F2] rounded border border-[#F5C2C2]">
                    <div className="text-[10px] text-[#D64545] font-semibold flex justify-between">
                      <span>ML MAE</span>
                      <span>-{dWith?.percentage_improvement?.toFixed(1)}%</span>
                    </div>
                    <div className="text-base font-bold font-mono text-[#D64545] mt-0.5">
                      {formatMins(dWith?.ml_mae)}
                    </div>
                    <div className="text-[10px] text-[#D64545] mt-0.5">
                      RMSE: {formatMins(dWith?.ml_rmse)}
                    </div>
                  </div>
                </div>

                <div className="p-2.5 bg-[#F8FAFB] rounded text-xs text-[#66717A] border border-[#D9DEE3]">
                  <span className="font-semibold text-[#172026]">Result:</span> ML preserves a{" "}
                  <strong>{dWith?.percentage_improvement?.toFixed(1) ?? "--"}%</strong> error advantage during disruptions.
                </div>
              </div>
            </div>
          </section>

          {/* SECTION 4: DATASET INFORMATION */}
          <section className="space-y-3">
            <div className="pb-1 border-b border-[#D9DEE3]">
              <h2 className="text-sm font-bold text-[#172026] flex items-center space-x-1.5">
                <span>📁</span>
                <span>4. Dataset Information & Provenance</span>
              </h2>
              <p className="text-xs text-[#66717A] mt-0.5">
                Audit trail and training split metadata for evaluation reproducibility.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Journeys */}
              <div className="p-3.5 bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs space-y-1">
                <div className="text-[#66717A] font-semibold text-xs flex items-center space-x-1.5">
                  <span>🚆</span>
                  <span>Journeys</span>
                </div>
                <div className="text-lg font-bold font-mono text-[#172026]">
                  {summary?.total_test_journeys || metrics.test_journeys || "--"} Test
                </div>
                <div className="text-[11px] text-[#66717A]">
                  Out of {datasetInfo?.total_journeys || metrics.total_journeys || "--"} total corridor journeys.
                </div>
              </div>

              {/* Samples */}
              <div className="p-3.5 bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs space-y-1">
                <div className="text-[#66717A] font-semibold text-xs flex items-center space-x-1.5">
                  <span>📊</span>
                  <span>Evaluated Samples</span>
                </div>
                <div className="text-lg font-bold font-mono text-[#172026]">
                  {(overall?.sample_count || metrics.test_samples || 0).toLocaleString() || "--"}
                </div>
                <div className="text-[11px] text-[#66717A]">
                  From {datasetInfo?.total_samples?.toLocaleString() || "--"} total records.
                </div>
              </div>

              {/* Evaluation Timestamp */}
              <div className="p-3.5 bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs space-y-1">
                <div className="text-[#66717A] font-semibold text-xs flex items-center space-x-1.5">
                  <span>🕒</span>
                  <span>Evaluation Date</span>
                </div>
                <div className="text-xs font-bold font-mono text-[#172026] truncate">
                  {metrics.evaluation_timestamp ? formatDateTime(metrics.evaluation_timestamp) : "--"}
                </div>
                <div className="text-[10px] text-[#8A949C] font-mono truncate" title={metrics.evaluation_timestamp || ""}>
                  {metrics.evaluation_timestamp || "N/A"}
                </div>
              </div>

              {/* Model Version */}
              <div className="p-3.5 bg-[#FFFFFF] rounded-lg border border-[#D9DEE3] shadow-2xs space-y-1">
                <div className="text-[#66717A] font-semibold text-xs flex items-center space-x-1.5">
                  <span>🏷️</span>
                  <span>Model Version</span>
                </div>
                <div className="text-lg font-bold font-mono text-[#2563A8]">
                  v{metrics.model_version || "1.0.0"}
                </div>
                <div className="text-[11px] text-[#66717A] truncate" title={metrics.model_name || ""}>
                  {metrics.model_name || "XGBoost Regressor"}
                </div>
              </div>
            </div>

            {/* Technical Metadata Table */}
            <div className="p-3.5 bg-[#F8FAFB] rounded-lg border border-[#D9DEE3] text-xs space-y-1.5 text-[#66717A]">
              <div className="font-bold text-[#172026] flex items-center space-x-1.5">
                <span>🔒</span>
                <span>Dataset Integrity & Synthetic Disclaimer</span>
              </div>
              <p className="text-[11px] text-[#66717A] leading-relaxed">
                {metrics.disclaimer || "SYNTHETIC SIMULATED DATASET ONLY. Timings, routes, disruptions, and model predictions are generated for MVP machine learning evaluation and DO NOT represent real historical Indian Railways operational logs."}
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[#D9DEE3]/70 text-[10px] font-mono text-[#8A949C]">
                <div>Source: <strong>data/processed/model_metrics.json</strong></div>
                <div>Random Seed: <strong>{summary?.random_seed || datasetInfo?.random_seed || 42}</strong></div>
                <div>Split: <strong>GroupKFold</strong></div>
                <div>Endpoint: <strong>GET /model/metrics</strong></div>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
