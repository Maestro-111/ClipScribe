"""Cooperative-cancellation seam for the extraction / parse pipeline.

The counterpart to :mod:`src.utils.progress`. Where the progress seam lets the
pipeline *push* state out at fixed points, this seam lets it *pull* one bit of
state in at those same points: "has this job been asked to stop?".

The core — extractor, parser, engine — depends only on the abstract
:class:`CancellationToken`, never on Redis or the web layer. Web execution paths
inject ``app.events.RedisCancellationToken`` (backed by a per-job Redis flag the
cancel endpoint sets), while the CLI (``main.py``), tests, and the Redis-down
fallback use :class:`NullCancellationToken`, which is never canceled.

Cancellation is *cooperative*: the running work is not killed from the outside
(which would leak GPU memory / half-written files and, under a ``solo`` worker,
take down the whole process). Instead the pipeline calls :meth:`check` at safe
checkpoints — between shots, between frame batches, between parser criteria —
and :class:`JobCanceled` unwinds the stack normally, so per-run resources are
released by ordinary teardown while the long-lived, model-loaded process
survives for the next job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class JobCanceled(Exception):
    """Raised by :meth:`CancellationToken.check` when the job must stop.

    Caught by :func:`app.job_execution.run_job_core` (recorded as ``canceled``)
    and by :class:`~src.clip_scribe.engine.ClipScribeEngine` (so a cancel is not
    mistaken for a failure). It is an ordinary exception so it unwinds through
    the pipeline's existing ``finally`` blocks (e.g. the extractor's
    ``cleanup()``), freeing per-run resources on the way out.
    """


class CancellationToken(ABC):
    """A read-only view of one job's cancel signal.

    Implementations must be cheap (checks sit inside hot loops) and must never
    raise from :meth:`is_canceled` — a transport hiccup must not crash a job;
    degrade to "not canceled" instead.
    """

    @abstractmethod
    def is_canceled(self) -> bool:
        """Return True if this job has been asked to stop."""

    def check(self) -> None:
        """Raise :class:`JobCanceled` if the job has been asked to stop."""
        if self.is_canceled():
            raise JobCanceled()


class NullCancellationToken(CancellationToken):
    """No-op token used by the CLI (``main.py``) and tests; never canceled."""

    def is_canceled(self) -> bool:
        return False
