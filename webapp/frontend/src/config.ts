// Backend API base. Override at build time with VITE_API_BASE.
//  - unset (local dev)      -> http://localhost:8000
//  - "" (Docker/nginx)      -> same-origin (nginx proxies /api, /auth, /media)
const _raw = (import.meta as any).env?.VITE_API_BASE;
export const API_BASE: string = _raw === undefined ? "http://localhost:8000" : _raw;

export const mediaUrl = (p?: string | null): string => (p ? `${API_BASE}${p}` : "");
