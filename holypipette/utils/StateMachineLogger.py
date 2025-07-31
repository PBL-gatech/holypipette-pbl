"""
Light‑weight state machine logger for Autopatcher attempts.
• Creates a date‑stamped *session* folder the first time it is imported.
• Creates one sub‑folder per attempt   →   session/attempt_<n>/
• Writes one *.pickle* file per state  →   <n>_<state>_<started>.pickle
"""

import os, pickle, functools, threading
from datetime import datetime
from typing import Dict


class StateMachineLogger:
    """
    A *very* thin, thread‑safe file logger.
    No video / movement batching – those belong to your FileLogger.
    """

    # -------- outcome codes -------- #
    SUCCESS = 0
    FAILURE = 1
    ABORTED = 2

    # -------- constructor -------- #
    def __init__(self,
                 base_path: str = "experiments/Data/state_recorder_data/",
                 attempt_id: int = 1):
        """
        Parameters
        ----------
        base_path   : str
            Root folder **without** the date stamp.
        attempt_id  : int
            Numeric counter supplied by the caller (AutoPatcher).
        """
        self._session_time = datetime.now()
        self.session_folder = os.path.join(
            base_path,
            self._session_time.strftime("%Y_%m_%d-%H_%M")
        )
        self.attempt_id   = attempt_id
        self.attempt_path = os.path.join(self.session_folder,
                                         f"attempt_{attempt_id}")
        os.makedirs(self.attempt_path, exist_ok=True)

        self._states: Dict[str, Dict] = {}
        self._lock   = threading.Lock()          # make file writes thread‑safe

    # ------------------------------------------------------------------ #
    # public API – called by the decorator
    # ------------------------------------------------------------------ #
    def start(self, state: str) -> None:
        """Stamp the *started* time (first call only)."""
        with self._lock:
            if state not in self._states:
                self._states[state] = {
                    "started":  datetime.now()
                                .isoformat(timespec="milliseconds"),
                    "finished": None,
                    "outcome":  None
                }

    def finish(self, state: str, outcome: int) -> None:
        """Stamp *finished*, set outcome, and write the pickle file."""
        with self._lock:
            rec = self._states[state]                    # must exist
            rec["finished"] = datetime.now() \
                              .isoformat(timespec="milliseconds")
            rec["outcome"]  = outcome
            self._save(state, rec)

    # ------------------------------------------------------------------ #
    # internal helper
    # ------------------------------------------------------------------ #
    def _save(self, state: str, record: Dict) -> None:
        ts    = record["started"].replace(":", "-")      # file‑safe
        fname = f"{self.attempt_id}_{state}_{ts}.pickle"
        path  = os.path.join(self.attempt_path, fname)
        with open(path, "wb") as f:
            pickle.dump(record, f)


# ---------------------------------------------------------------------- #
# decorator – attach to each top‑level sub‑method
# ---------------------------------------------------------------------- #
def record_state(state_name: str):
    """
    Usage
    -----
    >>> @record_state("gigaseal")
    >>> def gigaseal(self): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            logger = self._get_state_recorder()      # lazy / singleton
            logger.start(state_name)

            try:
                result   = fn(self, *args, **kwargs)
                aborted  = bool(getattr(self, "abort_requested", False))
                outcome  = (StateMachineLogger.ABORTED if aborted
                            else StateMachineLogger.SUCCESS)
                logger.finish(state_name, outcome)
                return result

            except Exception as exc:
                # treat *abort* messages as outcome = ABORTED
                msg     = str(exc).lower()
                aborted = "abort" in msg
                outcome = (StateMachineLogger.ABORTED if aborted
                           else StateMachineLogger.FAILURE)
                logger.finish(state_name, outcome)
                raise

            finally:
                # If NOT inside patch(), reset logger so the next call
                # becomes a fresh *attempt*.
                if not getattr(self, "_in_patch", False):
                    self._state_recorder = None
        return wrapper
    return decorator
