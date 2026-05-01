import { API_BASE_URL } from "./config";

export type EvalRunRecord = {
  run_id: string;
  completed_at?: string | null;
  configs?: Record<string, Record<string, Record<string, number>>>;
  baseline_delta?: Record<string, Record<string, Record<string, number>>>;
  [key: string]: unknown;
};

export type EvalRunListResponse = {
  run_ids: string[];
};

export type EvalRunSummaryResponse = {
  run_id: string;
  configs: Record<string, Record<string, number>>;
};

export type EvalRunAblationResponse = {
  run_id: string;
  configs: Record<string, Record<string, Record<string, number>>>;
  baseline_delta: Record<string, Record<string, Record<string, number>>>;
};

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function listEvalRuns(): Promise<EvalRunListResponse> {
  const response = await fetch(`${API_BASE_URL}/eval/runs`);
  return parseJson(response);
}

export async function getLatestEvalRun(): Promise<EvalRunRecord> {
  const response = await fetch(`${API_BASE_URL}/eval/runs/latest`);
  return parseJson(response);
}

export async function getEvalRun(runId: string): Promise<EvalRunRecord> {
  const response = await fetch(`${API_BASE_URL}/eval/runs/${encodeURIComponent(runId)}`);
  return parseJson(response);
}

export async function getEvalRunSummary(runId: string): Promise<EvalRunSummaryResponse> {
  const response = await fetch(`${API_BASE_URL}/eval/runs/${encodeURIComponent(runId)}/summary`);
  return parseJson(response);
}

export async function getEvalRunAblation(runId: string): Promise<EvalRunAblationResponse> {
  const response = await fetch(`${API_BASE_URL}/eval/runs/${encodeURIComponent(runId)}/ablation`);
  return parseJson(response);
}

export async function triggerEvalRun(): Promise<{ run_id: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/eval/trigger`, {
    method: "POST",
  });

  return parseJson(response);
}
