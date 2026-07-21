import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server fixed to 5173 so it matches the backend CORS allow-list.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, host: true },
});
