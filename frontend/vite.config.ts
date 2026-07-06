import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";

// Vite is our dev server + build tool. In dev it serves the app on :5173 with
// hot-module-reload; `vite build` produces a static bundle for prod.
//
// Plugin ORDER matters here:
//   1. TanStackRouterVite runs FIRST. It scans `src/routes/**` and (re)generates
//      `src/routeTree.gen.ts` — the typed map of every route. It must run before
//      the React plugin so the generated file exists when React compiles.
//   2. react() gives us JSX + Fast Refresh (component state survives edits).
//   3. tailwindcss() is Tailwind v4's official Vite plugin. It scans our files
//      for class names and injects the CSS. No `tailwind.config.js` or PostCSS
//      config needed in v4 — configuration is done in CSS (`src/styles.css`).
export default defineConfig({
  plugins: [
    TanStackRouterVite({ target: "react", autoCodeSplitting: true }),
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    // --- The dev proxy (web-app-plan §13) ---
    // The browser will call same-origin URLs like `/api/jobs`. Vite intercepts
    // anything starting with `/api` and forwards it to the FastAPI process on
    // :8000, rewriting `/api/jobs` -> `/jobs` (the API has no `/api` prefix).
    //
    // Why bother instead of calling http://localhost:8000 directly?
    //   - No CORS: the browser thinks every request is same-origin (:5173), so
    //     there is no cross-origin preflight to configure.
    //   - It mirrors production, where nginx sits in front and routes `/api/*`
    //     to the API container. Same mental model in dev and prod.
    // Note: the `gen:api` script below still hits :8000 directly — that runs in
    // Node (not the browser), so CORS never applies to it.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
