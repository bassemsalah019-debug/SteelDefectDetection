import { API_BASE } from "./config";

export interface User { id: string; email: string; full_name: string; created_at: string; }
export interface Token { access_token: string; refresh_token: string; token_type: string; user: User; }
export interface Detection { id: string; cls_name: string; confidence: number; x1: number; y1: number; x2: number; y2: number; }
export interface ImageOut {
  id: string; filename: string;
  original_url: string; annotated_url: string; cam_url: string;
  width: number; height: number; n_defects: number;
  brightness: number | null; quality: number | null; detections: Detection[];
}
export interface Inspection {
  id: string; title: string; mode: string; conf: number; imgsz: number;
  status: string; n_images: number; n_defects: number; created_at: string;
}
export interface InspectionDetail extends Inspection { images: ImageOut[]; class_counts: Record<string, number>; }
export interface Page<T> { items: T[]; total: number; page: number; page_size: number; }
export interface TimePoint { date: string; inspections: number; defects: number; }
export interface DashboardStats {
  total_inspections: number; total_images: number; total_defects: number;
  avg_defects_per_image: number; class_counts: Record<string, number>;
  mode_split: Record<string, number>; over_time: TimePoint[]; recent: Inspection[];
}
export interface ReportOut { id: string; lang: string; text: string; used_llm: boolean; created_at: string; }

let tokenGetter: () => string | null = () => null;
export const setTokenGetter = (fn: () => string | null) => { tokenGetter = fn; };

async function req<T>(path: string, opts: RequestInit = {}, auth = true): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  if (opts.body && !(opts.body instanceof FormData)) headers["Content-Type"] = "application/json";
  if (auth) { const t = tokenGetter(); if (t) headers["Authorization"] = `Bearer ${t}`; }
  const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
  if (!res.ok) {
    let message = res.statusText || `Request failed (${res.status})`;
    try { const j = await res.json(); message = j?.error?.message || message; } catch { /* ignore */ }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"];
export const CLASS_COLORS: Record<string, string> = {
  crazing: "#ff5d5d", inclusion: "#ff9f43", patches: "#feca57",
  pitted_surface: "#1dd1a1", "rolled-in_scale": "#54a0ff", scratches: "#c56cf0",
};

export const api = {
  signup: (b: { email: string; password: string; full_name: string }) =>
    req<Token>("/auth/signup", { method: "POST", body: JSON.stringify(b) }, false),
  login: (b: { email: string; password: string }) =>
    req<Token>("/auth/login", { method: "POST", body: JSON.stringify(b) }, false),
  me: () => req<User>("/auth/me"),
  dashboard: () => req<DashboardStats>("/api/dashboard/stats"),
  listInspections: (page = 1, pageSize = 12, mode?: string, cls?: string) => {
    const q = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (mode) q.set("mode", mode);
    if (cls) q.set("cls", cls);
    return req<Page<Inspection>>(`/api/inspections?${q.toString()}`);
  },
  getInspection: (id: string) => req<InspectionDetail>(`/api/inspections/${id}`),
  deleteInspection: (id: string) => req<void>(`/api/inspections/${id}`, { method: "DELETE" }),
  createInspection: (fd: FormData) => req<InspectionDetail>("/api/inspections", { method: "POST", body: fd }),
  report: (id: string, lang: string) =>
    req<ReportOut>(`/api/inspections/${id}/report`, { method: "POST", body: JSON.stringify({ lang }) }),
};
