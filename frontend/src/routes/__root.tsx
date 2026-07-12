import { createRootRouteWithContext } from "@tanstack/react-router";
import { Link, Outlet } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";

// __root.tsx is special: it's the ONE layout that wraps every page. Its
// <Outlet /> is where the matched child route renders. File-based routing maps
// files in src/routes/ to URLs:
//   __root.tsx        -> layout for everything
//   index.tsx         -> "/"
//   jobs.new.tsx      -> "/jobs/new"     (dot = path separator)
//   jobs.$jobId.tsx   -> "/jobs/:jobId"
//   runs.$runId.tsx   -> "/runs/:runId"  ($ = dynamic segment)
//
// createRootRouteWithContext types the router `context` we set in main.tsx, so
// route loaders can reach the shared QueryClient with full type-safety.
export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="border-b bg-white">
        <nav className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <Link to="/" className="text-lg font-semibold">
            ClipScribe
          </Link>
          {/* activeProps styles the link when its route is active. `to` is
              type-checked against the generated route tree — a bad path fails
              to compile. */}
          <Link
            to="/"
            activeProps={{ className: "font-medium text-blue-600" }}
            className="text-sm text-neutral-600 hover:text-neutral-900"
          >
            Jobs
          </Link>
          <Link
            to="/jobs/new"
            activeProps={{ className: "font-medium text-blue-600" }}
            className="text-sm text-neutral-600 hover:text-neutral-900"
          >
            New job
          </Link>
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
