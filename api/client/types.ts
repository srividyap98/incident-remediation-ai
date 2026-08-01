/**
 * TypeScript interfaces mirroring the Python Pydantic schemas.
 * Keep in sync with api/schemas/models.py
 */

export interface TimelineEvent {
  timestamp: string;
  event: string;
  source?: string;
}

export type Severity = "P1" | "P2" | "P3" | "P4";
export type IncidentStatus = "completed" | "pending_review" | "error";

export interface IncidentRequest {
  incident_id: string;
  incident_title: string;
  incident_description: string;
  severity: Severity;
  affected_service: string;
  timeline_events?: TimelineEvent[];
  raw_logs?: string[];
}

export interface IncidentResponse {
  incident_id: string;
  thread_id: string;
  status: IncidentStatus;
  risk_score: number | null;
  requires_human_review: boolean;
  root_cause_analysis: string | null;
  remediation_steps: string[];
  confidence_score: number | null;
  final_response: string | null;
  agent_messages: string[];
  risk_factors: string[];
  grounding_warnings?: string[];
  error: string | null;
}

export interface HITLResumeRequest {
  thread_id: string;
  approved: boolean;
  reviewer_id?: string;
  review_notes?: string;
}

export interface FeedbackRequest {
  incident_id: string;
  rca_accuracy: 1 | 2 | 3 | 4 | 5;
  remediation_usefulness: 1 | 2 | 3 | 4 | 5;
  correction_notes?: string;
  reviewer_id?: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  incident_id: string;
  status: "recorded";
  average_rating: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  vector_store_backend: string;
  bedrock_model: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
