/**
 * Incident Remediation AI — TypeScript Client SDK
 * ------------------------------------------------
 * Typed async functions for every API endpoint.
 * Works in Node.js 18+ and modern browsers (native fetch).
 *
 * Usage:
 *   import { setApiKey, submitIncident, resumeHITL, submitFeedback } from './sdk';
 *
 *   setApiKey('sk-live-abc123');  // required once API_KEYS is configured server-side
 *
 *   const result = await submitIncident({
 *     incident_id: 'INC-001',
 *     incident_title: 'DB connection pool exhausted',
 *     severity: 'P1',
 *     affected_service: 'payments-api',
 *     raw_logs: ['FATAL: too many connections'],
 *   });
 *
 *   if (result.status === 'pending_review') {
 *     // reviewer identity comes from the API key, not a client-supplied field
 *     await resumeHITL({ thread_id: result.thread_id, approved: true });
 *   }
 */
import type {
  IncidentRequest,
  IncidentResponse,
  HITLResumeRequest,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  ApiError,
} from "./types";

// ── Config ────────────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL =
  typeof process !== "undefined"
    ? process.env.INCIDENT_AI_BASE_URL ?? "http://localhost:8000"
    : "http://localhost:8000";

// Sent as the X-API-Key header on every request once set. Node picks up
// INCIDENT_AI_API_KEY automatically; browsers have no env vars, so call
// setApiKey() once at startup instead.
let apiKey: string | undefined =
  typeof process !== "undefined" ? process.env.INCIDENT_AI_API_KEY : undefined;

export function setApiKey(key: string | undefined): void {
  apiKey = key;
}

const DEFAULT_TIMEOUT_MS = 120_000; // 2 min — LLM calls can take time
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1_000;

// ── Internal helpers ──────────────────────────────────────────────────────────

class IncidentAIError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly detail?: string
  ) {
    super(message);
    this.name = "IncidentAIError";
  }
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  retries = MAX_RETRIES
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    // Retry on 429 (rate limit) or 5xx server errors
    if (!response.ok && retries > 0 && (response.status === 429 || response.status >= 500)) {
      const retryAfter = response.headers.get("Retry-After");
      const delay = retryAfter ? parseInt(retryAfter) * 1000 : RETRY_DELAY_MS * (MAX_RETRIES - retries + 1);
      await sleep(delay);
      return fetchWithRetry(url, options, retries - 1);
    }

    return response;
  } finally {
    clearTimeout(timer);
  }
}

function authHeaders(): Record<string, string> {
  return apiKey ? { "X-API-Key": apiKey } : {};
}

async function post<TBody, TResponse>(
  path: string,
  body: TBody,
  baseUrl = DEFAULT_BASE_URL
): Promise<TResponse> {
  const url = `${baseUrl}${path}`;
  const response = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err: ApiError = await response.json();
      detail = err.detail ?? detail;
    } catch {
      // ignore parse error
    }
    throw new IncidentAIError(`Request to ${path} failed`, response.status, detail);
  }

  return response.json() as Promise<TResponse>;
}

async function get<TResponse>(
  path: string,
  baseUrl = DEFAULT_BASE_URL
): Promise<TResponse> {
  const url = `${baseUrl}${path}`;
  const response = await fetchWithRetry(url, { method: "GET", headers: authHeaders() });

  if (!response.ok) {
    throw new IncidentAIError(`GET ${path} failed`, response.status);
  }

  return response.json() as Promise<TResponse>;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Submit an incident for AI-driven analysis and remediation.
 * Returns immediately — if status is 'pending_review', call resumeHITL() after engineer approval.
 */
export async function submitIncident(
  incident: IncidentRequest,
  baseUrl?: string
): Promise<IncidentResponse> {
  return post<IncidentRequest, IncidentResponse>(
    "/api/v1/incidents/",
    incident,
    baseUrl
  );
}

/**
 * Resume a paused high-risk incident workflow after human review.
 */
export async function resumeHITL(
  request: HITLResumeRequest,
  baseUrl?: string
): Promise<IncidentResponse> {
  return post<HITLResumeRequest, IncidentResponse>(
    "/api/v1/hitl/resume",
    request,
    baseUrl
  );
}

/**
 * Submit engineer feedback on a completed remediation.
 * Ratings are 1 (poor) to 5 (excellent).
 */
export async function submitFeedback(
  feedback: FeedbackRequest,
  baseUrl?: string
): Promise<FeedbackResponse> {
  return post<FeedbackRequest, FeedbackResponse>(
    "/api/v1/feedback",
    feedback,
    baseUrl
  );
}

/**
 * Submit an incident via the voice/IVR channel.
 * Pass raw speech-to-text transcript — the API normalises it into an IncidentRequest.
 */
export async function submitVoiceIncident(
  payload: { transcript: string; caller_id?: string; service_hint?: string },
  baseUrl?: string
): Promise<IncidentResponse> {
  return post<typeof payload, IncidentResponse>(
    "/api/v1/voice/incident",
    payload,
    baseUrl
  );
}

/**
 * Check API health — useful for liveness monitoring.
 */
export async function getHealth(baseUrl?: string): Promise<HealthResponse> {
  return get<HealthResponse>("/health", baseUrl);
}

// Re-export types for convenience
export type {
  IncidentRequest,
  IncidentResponse,
  HITLResumeRequest,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  Severity,
  IncidentStatus,
} from "./types";

export { IncidentAIError };
