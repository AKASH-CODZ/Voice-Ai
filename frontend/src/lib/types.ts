export type Mode = "casual" | "teaching" | "observation";
export type EnginePreference = "auto" | "local" | "cloud";
export type Role = "user" | "assistant";

export interface GPUInfo {
  index: number;
  name: string;
  total_vram_gb: number;
  free_vram_gb: number;
  used_vram_gb: number;
  temperature_c: number | null;
  utilization_pct: number | null;
  driver_version: string | null;
}

export interface HardwareReport {
  cuda_available: boolean;
  gpus: GPUInfo[];
  cpu_model: string;
  cpu_physical_cores: number;
  cpu_logical_cores: number;
  total_ram_gb: number;
  available_ram_gb: number;
  platform: string;
  apple_silicon: boolean;
  ollama_reachable: boolean;
  cloud_credentials: boolean;
  recommended_engine: string;
  reason: string;
}

export interface HealthResponse {
  status: "ok";
  version: string;
  engine_mode: string;
  recommended_engine: string;
  reason: string;
  hardware: HardwareReport;
}

export interface LatencySample {
  turn_id: number;
  stt_ms: number;
  llm_ttft_ms: number;
  llm_ms: number;
  tts_ms: number;
  e2e_ms: number;
  engine: string;
}

export interface TranscriptTurn {
  id: string;
  role: Role;
  text: string;
  hesitation?: boolean;
  correctionMade?: boolean;
  streaming?: boolean;
}

/** Server → client events, discriminated on `type`. Mirrors backend schemas.py. */
export type ServerEvent =
  | { type: "ready"; session_id: string; mode: Mode; engine: string; engine_reason: string;
      audio_in_sample_rate: number; audio_out_sample_rate: number; frame_samples: number;
      hardware: HardwareReport }
  | { type: "vad"; speaking: boolean }
  | { type: "partial_transcript"; text: string }
  | { type: "transcript"; turn_id: number; role: Role; text: string;
      hesitation: boolean; correction_made: boolean }
  | { type: "agent_delta"; text: string }
  | { type: "speaking"; active: boolean }
  | { type: "stall"; silence_ms: number }
  | { type: "latency" } & LatencySample
  | { type: "error"; message: string; fatal: boolean }
  | { type: "engine_switched"; engine: string; reason: string };

export type ConnectionStatus =
  | "idle" | "requesting-mic" | "connecting" | "live" | "closed" | "error";
