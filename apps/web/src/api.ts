/** Typed API client helpers for upload, pipeline execution, and report access. */

import { z } from "zod";
import {
  extractResponseSchema,
  mapRoomsResponseSchema,
  matchResponseSchema,
  renderReportResponseSchema,
  runIdSchema,
  uploadRunResponseSchema,
  type ExtractResponse,
  type MapRoomsResponse,
  type MatchResponse,
  type RenderReportResponse,
  type UploadRunResponse,
} from "./validation";

const API_BASE =
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_BASE_URL ??
  "http://localhost:8000";

const apiBaseSchema = z.string().url("Invalid VITE_API_BASE / VITE_API_BASE_URL.");
const apiBaseParsed = apiBaseSchema.safeParse(API_BASE);
if (!apiBaseParsed.success) {
  throw new Error(apiBaseParsed.error.issues[0]?.message ?? "Invalid API base URL.");
}
const NORMALIZED_API_BASE = apiBaseParsed.data.replace(/\/+$/, "");

function validateRunIdOrThrow(runId: string): string {
  const parsed = runIdSchema.safeParse(runId);
  if (!parsed.success) {
    throw new Error(parsed.error.issues[0]?.message ?? "Invalid run id.");
  }
  return parsed.data;
}

async function parseJsonOrThrow<T>(res: Response, path: string, schema: z.ZodType<T>): Promise<T> {
  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    throw new Error(`API ${path} returned invalid JSON.`);
  }
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const location = issue?.path?.length ? issue.path.join(".") : "response";
    throw new Error(`API ${path} response validation failed at ${location}: ${issue?.message ?? "invalid data"}`);
  }
  return parsed.data;
}

async function apiRequest<T>(path: string, schema: z.ZodType<T>, init?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${NORMALIZED_API_BASE}${path}`, init);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(body || `Request failed with status ${res.status}`);
    }
    return parseJsonOrThrow(res, path, schema);
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error(
        `Could not reach API at ${NORMALIZED_API_BASE}. Make sure the backend is running and your localhost origin is allowed by CORS.`,
      );
    }
    throw err;
  }
}

export async function uploadRun(proposalA: File, proposalB: File): Promise<UploadRunResponse> {
  const fd = new FormData();
  fd.append("proposal_a", proposalA);
  fd.append("proposal_b", proposalB);

  return apiRequest("/uploads", uploadRunResponseSchema, { method: "POST", body: fd });
}

export async function extract(runId: string): Promise<ExtractResponse> {
  const validRunId = validateRunIdOrThrow(runId);
  return apiRequest(`/runs/${validRunId}/extract`, extractResponseSchema, { method: "POST" });
}

export async function mapRooms(runId: string): Promise<MapRoomsResponse> {
  const validRunId = validateRunIdOrThrow(runId);
  return apiRequest(`/runs/${validRunId}/map-rooms`, mapRoomsResponseSchema, { method: "POST" });
}

export async function match(runId: string): Promise<MatchResponse> {
  const validRunId = validateRunIdOrThrow(runId);
  return apiRequest(`/runs/${validRunId}/match`, matchResponseSchema, { method: "POST" });
}

export async function renderReport(runId: string): Promise<RenderReportResponse> {
  const validRunId = validateRunIdOrThrow(runId);
  return apiRequest(`/runs/${validRunId}/render`, renderReportResponseSchema, { method: "POST" });
}

export function reportUrl(runId: string) {
  const validRunId = validateRunIdOrThrow(runId);
  return `${NORMALIZED_API_BASE}/runs/${validRunId}/report`;
}

export type PipelineEvent = {
  step: string;
  status: string;
  msg?: string;
  data?: Record<string, unknown>;
};

/** Open an SSE stream to run the full pipeline and stream events back. Returns a cleanup fn. */
export function streamPipeline(
  runId: string,
  onEvent: (event: PipelineEvent) => void,
  onError: (err: Error) => void,
): () => void {
  const validRunId = validateRunIdOrThrow(runId);
  const url = `${NORMALIZED_API_BASE}/runs/${validRunId}/pipeline/stream`;
  const source = new EventSource(url);

  source.onmessage = (e) => {
    try {
      const parsed: PipelineEvent = JSON.parse(e.data as string);
      onEvent(parsed);
      if (parsed.step === "done" || parsed.status === "error") {
        source.close();
      }
    } catch {
      // ignore parse errors on individual events
    }
  };

  source.onerror = () => {
    source.close();
    onError(new Error("Pipeline stream connection lost."));
  };

  return () => source.close();
}
