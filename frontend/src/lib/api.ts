/**
 * Reusable FastAPI Client for Dynamic Train ETA Forecasting Engine.
 * Single source of truth communicating directly with the backend.
 * No business or prediction logic is duplicated on the client.
 */

import {
  HealthResponse,
  TrainListResponse,
  TrainDetailResponse,
  SingleStationETAResponse,
  StationArrivalsResponse,
  EventInjectionRequest,
  EventInjectionResponse,
  ModelMetricsResponse,
  DataSourceConfigResponse,
  DemoScenarioResponse,
  DemoActionResult,
  LiveTrainResult,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

export class ApiError extends Error {
  status: number;
  errorCode?: string;
  detail?: string;
  suggestedAction?: string;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;

    if (typeof data === "object" && data !== null) {
      const d = data as Record<string, unknown>;
      if (typeof d.error === "string") this.errorCode = d.error;
      if (typeof d.detail === "string") this.detail = d.detail;
      if (typeof d.suggested_action === "string") this.suggestedAction = d.suggested_action;
    }
  }
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...options.headers,
  };

  const timeoutMs = options.timeoutMs ?? 10000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      headers,
      signal: options.signal || controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let errorBody: unknown;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }

      let errorMessage = `HTTP error ${res.status}: ${res.statusText}`;
      if (typeof errorBody === "object" && errorBody !== null) {
        const eb = errorBody as Record<string, unknown>;
        if (typeof eb.message === "string" && eb.message.trim()) {
          errorMessage = eb.message.trim();
        } else if (typeof eb.detail === "string" && eb.detail.trim()) {
          errorMessage = eb.detail.trim();
        } else if (typeof eb.error === "string" && eb.error.trim()) {
          errorMessage = eb.error.trim();
        }
      } else if (typeof errorBody === "string" && errorBody.trim()) {
        errorMessage = errorBody.trim().slice(0, 200);
      }

      throw new ApiError(errorMessage, res.status, errorBody);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        `Connection timed out after ${(timeoutMs / 1000).toFixed(0)}s connecting to ${endpoint}. Please verify FastAPI is running.`,
        408
      );
    }
    const message = err instanceof Error ? err.message : "Network unreachable";
    throw new ApiError(
      `Unable to reach backend service (${message}). Please ensure FastAPI is running on http://localhost:8000.`,
      0,
      err
    );
  }
}

export const api = {
  /**
   * Health check to test backend connection.
   */
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  /**
   * List all trains and active simulated states.
   */
  async getTrains(): Promise<TrainListResponse> {
    return request<TrainListResponse>("/trains");
  },

  /**
   * Fetch complete train running details, route stops, and multi-station ETA predictions.
   */
  async getTrainDetails(trainId: string | number): Promise<TrainDetailResponse> {
    return request<TrainDetailResponse>(`/train/${encodeURIComponent(String(trainId))}`);
  },

  /**
   * Alias for getTrainDetails.
   */
  async getTrainById(trainId: string | number): Promise<TrainDetailResponse> {
    return this.getTrainDetails(trainId);
  },

  /**
   * Fetch live train status and predicted ETAs via RailRadar integration.
   */
  async getLiveTrain(trainNumber: string): Promise<LiveTrainResult> {
    return request<LiveTrainResult>(`/live/train/${encodeURIComponent(trainNumber.trim())}`);
  },

  /**
   * Fetch single-station ETA prediction for a specific train and station.
   */
  async getStationETA(
    trainId: string | number,
    stationCode: string
  ): Promise<SingleStationETAResponse> {
    return request<SingleStationETAResponse>(
      `/train/${encodeURIComponent(String(trainId))}/eta/${encodeURIComponent(stationCode.toUpperCase().trim())}`
    );
  },

  /**
   * Fetch upcoming train arrivals for a specific station.
   */
  async getStationArrivals(
    stationCode: string,
    windowHours?: number
  ): Promise<StationArrivalsResponse> {
    const query = windowHours !== undefined ? `?window_hours=${windowHours}` : "";
    return request<StationArrivalsResponse>(
      `/station/${encodeURIComponent(stationCode.toUpperCase().trim())}/arrivals${query}`
    );
  },

  /**
   * Inject operational disruption event into the simulator.
   */
  async simulateEvent(payload: EventInjectionRequest): Promise<EventInjectionResponse> {
    return request<EventInjectionResponse>("/simulate/event", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch latest evaluation metrics comparing Baseline heuristic vs XGBoost ML.
   */
  async getModelMetrics(): Promise<ModelMetricsResponse> {
    return request<ModelMetricsResponse>("/model/metrics");
  },

  /**
   * Fetch current system data-source telemetry mode and cache stats.
   */
  async getDataSourceConfig(): Promise<DataSourceConfigResponse> {
    return request<DataSourceConfigResponse>("/system/data-source");
  },

  /**
   * Reset deterministic hackathon demo scenario (Seed 42).
   */
  async resetDemoScenario(): Promise<DemoScenarioResponse> {
    return request<DemoScenarioResponse>("/demo/reset", {
      method: "POST",
    });
  },

  /**
   * Fetch current state of the deterministic hackathon demo scenario.
   */
  async getDemoScenario(): Promise<DemoScenarioResponse> {
    return request<DemoScenarioResponse>("/demo/scenario");
  },

  /**
   * Execute prepared hackathon demo action deterministically.
   */
  async executeDemoAction(actionId: string): Promise<DemoActionResult> {
    return request<DemoActionResult>(`/demo/action/${encodeURIComponent(actionId)}`, {
      method: "POST",
    });
  },
};

/**
 * Utility display formatters
 */
export function formatTime(isoString: string | null | undefined): string {
  if (!isoString) return "--:--";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  } catch {
    return isoString;
  }
}

export function formatDateTime(isoString: string | null | undefined): string {
  if (!isoString) return "N/A";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return isoString;
  }
}

export function formatDelay(delayMinutes: number | null | undefined): {
  text: string;
  isDelayed: boolean;
  colorClass: string;
} {
  const d = delayMinutes ?? 0;
  if (d <= 0.5) {
    return {
      text: "On Time",
      isDelayed: false,
      colorClass: "bg-emerald-100 text-emerald-800 border-emerald-300",
    };
  }
  return {
    text: `+${d.toFixed(1)} min`,
    isDelayed: true,
    colorClass:
      d > 15
        ? "bg-rose-100 text-rose-800 border-rose-300 font-semibold"
        : "bg-amber-100 text-amber-800 border-amber-300",
  };
}
