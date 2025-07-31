"""
Light-weight state machine logger for Autopatcher attempts.
• Creates ONE date-stamped *session* folder the first time it is used.
• Creates one sub-folder per attempt   →   session/attempt_<n>/
• Writes one *.pickle* per state       →   <n>_<state>_<started>.pickle
• Uses Unix-epoch milliseconds for all timestamps (ints).
"""

import os, pickle, functools, threading, time
from typing import Dict


class StateMachineLogger:
    # ---------- outcome codes ---------- #
    SUCCESS = 0
    FAILURE = 1
    ABORTED = 2

    # ---------- statetype codes ---------- #
    CLASSIC = 0
    MANUAL = 1
    AGENT = 2
    NOMODE = 3


    # ---------- class-level session storage ---------- #
    _session_folder: str | None = None          # set on first use

    # ---------- constructor ---------- #
    def __init__(self,
                 base_path: str = "experiments/Data/state_recorder_data/",
                 attempt_id: int = 1):
        """
        Parameters
        ----------
        base_path  : str
            Root folder **without** the date stamp. A single session folder
            (YYYY_MM_DD-HH_MM) is created once under this root.
        attempt_id : int
            Numeric counter supplied by the caller (AutoPatcher).
        """
        # Initialise the session folder only once
        if StateMachineLogger._session_folder is None:
            date_stamp = time.strftime("%Y_%m_%d-%H_%M")
            StateMachineLogger._session_folder = os.path.join(base_path,
                                                              date_stamp)
            os.makedirs(StateMachineLogger._session_folder, exist_ok=True)

        self.session_folder = StateMachineLogger._session_folder
        self.attempt_id     = attempt_id
        self.attempt_path   = os.path.join(self.session_folder,
                                           f"attempt_{attempt_id}")
        os.makedirs(self.attempt_path, exist_ok=True)

        self._states: Dict[str, Dict] = {}
        self._lock = threading.Lock()           # thread-safe writes

    # ------------------------------------------------------------------ #
    # public API – called by the decorator
    # ------------------------------------------------------------------ #
    def start(self, state: str, system_mode: int | None = None) -> None:
        """
        Stamp the *started* time (first call only) and record the
        system-mode code supplied by the caller.

        Parameters
        ----------
        state        : str
            Name of the state being entered.
        system_mode  : int | None
            One of the CLASSIC / MANUAL / AGENT / NOMODE codes.  If None,
            NOMODE is stored.
        """
        if system_mode is None:
            system_mode = StateMachineLogger.NOMODE

        with self._lock:
            if state not in self._states:
                epoch_ms = int(time.time() * 1_000)            # milliseconds
                self._states[state] = {
                    "started":     epoch_ms,
                    "finished":    None,
                    "outcome":     None,
                    "system_mode": system_mode
                }

    def finish(self, state: str, outcome: int) -> None:
        """Stamp *finished*, set outcome, and write the pickle file."""
        with self._lock:
            rec = self._states[state]                      # must exist
            rec["finished"] = int(time.time() * 1_000)
            rec["outcome"]  = outcome
            self._save(state, rec)

    # ------------------------------------------------------------------ #
    # internal helper
    # ------------------------------------------------------------------ #
    def _save(self, state: str, record: Dict) -> None:
        ts    = record["started"]                          # epoch ms int
        fname = f"{self.attempt_id}_{state}_{ts}.pickle"
        path  = os.path.join(self.attempt_path, fname)
        with open(path, "wb") as f:
            pickle.dump(record, f)


# ---------------------------------------------------------------------- #
# decorator – attach to each top-level sub-method
# ---------------------------------------------------------------------- #
def record_state(state_name: str):
    """
    Decorator to log state transitions with StateMachineLogger.

    >>> @record_state("gigaseal")
    >>> def gigaseal(self): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            logger = self._get_state_recorder()    # lazy / singleton

            # --- map self.config.mode → numeric code ---
            mode_str = getattr(getattr(self, "config", None), "mode", None)
            mode_map = {
                "Classic": StateMachineLogger.CLASSIC,
                "Manual":  StateMachineLogger.MANUAL,
                "Agent":   StateMachineLogger.AGENT,
            }
            mode_code = mode_map.get(mode_str, StateMachineLogger.NOMODE)

            logger.start(state_name, mode_code)

            try:
                result   = fn(self, *args, **kwargs)
                aborted  = bool(getattr(self, "abort_requested", False))
                outcome  = StateMachineLogger.ABORTED if aborted else StateMachineLogger.SUCCESS
                logger.finish(state_name, outcome)
                return result

            except Exception as exc:
                msg     = str(exc).lower()
                aborted = "abort" in msg
                outcome = StateMachineLogger.ABORTED if aborted else StateMachineLogger.FAILURE
                logger.finish(state_name, outcome)
                raise

            finally:
                # If NOT inside patch(), reset logger so the next call
                # becomes a fresh *attempt*.
                if not getattr(self, "_in_patch", False):
                    self._state_recorder = None
        return wrapper
    return decorator
