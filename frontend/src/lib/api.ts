import { API_BASE_URL } from "./config";

export type QueryPayload = {
  question: string;
  chunking_strategy?: string;
  similarity_threshold?: number;
  use_cache?: boolean;
  tenant_id?: string;
  user_id?: string;
};

export type QueryResult = {
  final_answer?: string;
  final_route?: string;
  cache_status?: string;
  response_source?: string;
  citations?: string[];
  retrieved_contexts?: string[];
  retrieved_context_count?: number;
  plan_type?: string;
  [key: string]: unknown;
};

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function runQuery(payload: QueryPayload): Promise<QueryResult> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      chunking_strategy: "hierarchical",
      similarity_threshold: 0.92,
      use_cache: true,
      tenant_id: "default",
      user_id: "frontend-user",
      ...payload,
    }),
  });

  return parseJson<QueryResult>(response);
}

export async function clearCache(): Promise<{ status: string; cleared: Record<string, number> }> {
  const response = await fetch(`${API_BASE_URL}/cache/clear`, {
    method: "POST",
  });

  return parseJson(response);
}

export async function getCacheStats(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/cache/stats`);
  return parseJson(response);
}
