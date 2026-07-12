import createClient from "openapi-fetch";
import type { paths } from "./types";

// A single typed HTTP client for the whole app.
//
// `createClient<paths>` binds the client to the generated OpenAPI `paths` type.
// After `pnpm gen:api`, calls are fully checked:
//   const { data, error } = await api.GET("/jobs", { params: { query: { limit: 50 } } });
//   // `data` is typed as the 200 response; `error` as the problem+json body.
//
// baseUrl is "/api" so requests go to the SAME origin as the app (:5173) and hit
// the Vite proxy, which forwards to the FastAPI process (see vite.config.ts). In
// prod, nginx maps "/api" to the API container — same path, no code change.
export const api = createClient<paths>({ baseUrl: "/api" });

// openapi-fetch never throws on HTTP errors. Instead every call returns
// `{ data, error }`: on 2xx `data` is set, otherwise `error` holds the parsed
// body (our RFC 7807 problem+json — see backend/app/errors.py). This helper
// turns that into the throw-on-error style TanStack Query expects in its
// queryFn, while giving us a typed Error carrying the problem detail.
export class ApiError extends Error {
  constructor(
    public status: number,
    public title: string,
    detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

type Problem = { status?: number; title?: string; detail?: string };

export function unwrap<T>(result: { data?: T; error?: unknown }): T {
  if (result.error !== undefined) {
    const p = (result.error ?? {}) as Problem;
    throw new ApiError(
      p.status ?? 0,
      p.title ?? "Error",
      p.detail ?? "Request failed",
    );
  }
  // If there's no error, data is present (openapi-fetch guarantees this for 2xx).
  return result.data as T;
}
