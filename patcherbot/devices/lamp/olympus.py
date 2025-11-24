import serial, threading, time
from .lamp import Lamp

__all__ = ["OlympusLamp"]


class OlympusLamp(Lamp):
    """
    RS-232 driver for Olympus BX/IX/MX-UCB fluorescence shutters + cube wheels.

    ✔  Same public API as Lamp base-class  
    ✔  Command set and edge-case handling taken verbatim from OlympusTester.py  
    ✔  Thread-safe serial writes (mutex)  
    ✔  On-demand queries only – no background poll thread  
    """

    # ──────────────────── protocol constants ────────────────────
    _PREFIX          = "1"
    _SHUTTER_CMD     = "SHUTTER"
    _MAX_CUBE_TIME   = 10.0
    _BUSY_ERR_PREFIX = "Cube error: !,E01120"

    # ----- NEW: tester-style manual-override + defaults -----
    _MANUAL_CUBE_PREFIX: str | None = "1"
    _MANUAL_CUBE_CMD:    str | None = "CUBE"
    _DEFAULT_SLOTS:      int        = 6

    # ─────────────────────── construction ───────────────────────
    def __init__(self, port: str = "COM21", baud: int = 19200, cube_slots: int | None = None):
        """
        Parameters
        ----------
        port        : serial port name (e.g. "COM21" or "/dev/ttyUSB0")
        baud        : baud rate (default 19200, 8E2)
        cube_slots  : override number of wheel slots (default = 6)
        """
        # open & configure the port before any Lamp logic fires
        self._com = serial.Serial(
            port,
            baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_TWO,
            timeout=1.0,
            rtscts=True,
            write_timeout=1.0
        )

        self._lock                = threading.Lock()
        self._cube_slots_override = cube_slots
        self._cube_slots          = None if cube_slots is None else cube_slots  
        self._cube_cmd            = None
        self._cube_prefix         = OlympusLamp._PREFIX
        self._current_filter      = None
        self.shutter_state        = "closed"

        super().__init__()  # triggers self._initialize()


    # ───────────────────── low‑level helpers ─────────────────────
    def _send_cmd(self, line: str, tries: int = 3) -> str:
        """Thread‑safe I/O with automatic <CR><LF>, returns *decoded* reply."""
        raw = (line + "\r\n").encode()
        for _ in range(tries):
            with self._lock:
                self._com.write(raw)
                ans = self._com.readline()
            if ans:
                return ans.strip().decode(errors="replace")
        raise TimeoutError(f"No reply to {line!r}")

    def _cmd(self, tail: str) -> str:
        """Convenience: prefix the controller address to *tail*."""
        return self._send_cmd(f"{self._PREFIX}{tail}")

    # ──────────────────────── start‑up ──────────────────────────
    def _probe_var(self, pref: str, name: str) -> bool:
        """Return *True* if controller recognises variable *name* under *pref*."""
        for q in (f"{name} ?", f"{name}?"):
            try:
                if self._send_cmd(f"{pref}{q}").startswith(name):
                    return True
            except TimeoutError:
                pass
        return False

    # ─────────────────────── cube detection ──────────────────────
    def _detect_cube(self) -> None:
        """
        Replicates OlympusTester.detect_cube(): manual override first,
        otherwise probe (“CUBE” / “MU”, both prefixes).
        """
        # ---- manual override exactly like tester ----
        if OlympusLamp._MANUAL_CUBE_CMD:
            self._cube_cmd    = OlympusLamp._MANUAL_CUBE_CMD
            self._cube_prefix = (OlympusLamp._MANUAL_CUBE_PREFIX
                                 or OlympusLamp._PREFIX)
            self._cube_slots  = (self._cube_slots_override
                                 or OlympusLamp._DEFAULT_SLOTS)
            return

        # ---- auto-detect: (name, prefix) cartesian probe ----
        for name in ("CUBE", "MU"):
            for pref in (OlympusLamp._PREFIX,
                         "" if OlympusLamp._PREFIX == "1" else "1"):
                if self._probe_var(pref, name):
                    self._cube_cmd    = name
                    self._cube_prefix = pref
                    self._cube_slots  = (self._cube_slots_override
                                         or OlympusLamp._DEFAULT_SLOTS)
                    return

        # ---- nothing found ----
        self._cube_cmd = None   # leave prefix / slots indeterminate

    def _initialize(self):
        """Login + auto‑detect cube wheel."""
        # ---------- login ----------
        if not self._cmd("LOG?").endswith("IN"):
            if not self._cmd("LOG IN").endswith("+"):
                raise RuntimeError("OlympusLamp: LOGIN failed")

        # ---------- cube detection ----------
        self._detect_cube()

    # ───────────────────── shutter control ──────────────────────
    def _query_shutter_state(self) -> str:
        """Return 'IN' (closed) or 'OUT' (open)."""
        rep = self._cmd(f"{self._SHUTTER_CMD}?")
        return rep.split()[-1]

    def open_shutter(self):
        if self._query_shutter_state() != "OUT":
            self._cmd(f"{self._SHUTTER_CMD} OUT")
        self.shutter_state = "open"

    def close_shutter(self):
        if self._query_shutter_state() != "IN":
            self._cmd(f"{self._SHUTTER_CMD} IN")
        self.shutter_state = "closed"

    def get_shutter_state(self):
        state = self._query_shutter_state()
        self.shutter_state = "open" if state == "OUT" else "closed"
        return self.shutter_state

    # ───────────────────── cube / filter wheel ──────────────────
    @staticmethod
    def _parse_slot(reply: str) -> int | None:
        """Extract first integer token from reply line."""
        for tok in reply.replace(",", " ").split():
            if tok.isdigit():
                return int(tok)
        return None

    def _cube_pos(self) -> int | None:
        """Return current cube slot (1‑based) or *None* if controller is busy."""
        rep = self._send_cmd(f"{self._cube_prefix}{self._cube_cmd}?")
        while rep.strip(" +") in {"", "CUBE", "MU"}:   # skip empty / ack
            rep = self._com.readline().strip().decode(errors="replace")
        if rep.startswith("!,E"):
            raise RuntimeError(f"Cube error: {rep}")
        return self._parse_slot(rep)
    
    def _set_cube(self,pos):
        """Move the cube wheel to *pos* (1‑based slot)."""
        if self._cube_cmd is None:
            raise RuntimeError("OlympusLamp: Cube wheel not detected")
        if self._cube_slots and not (1 <= pos <= self._cube_slots):
            raise ValueError(f"Filter slot must be 1‑{self._cube_slots}")

        # Command move
        self._send_cmd(f"{self._cube_prefix}{self._cube_cmd} {pos}")

        # Wait until reached or timeout
        t0, last_rep = time.time(), ""
        while time.time() - t0 < self._MAX_CUBE_TIME:
            try:
                slot = self._cube_pos()
                if slot == pos:
                    self._current_filter = pos
                    return
            except RuntimeError as e:
                last_rep = str(e)
            time.sleep(0.3)

        # Graceful exit if only complaint was "busy +"
        if last_rep.startswith(self._BUSY_ERR_PREFIX):
            self._current_filter = pos
            return
        raise RuntimeError("Cube move timed out")


    # --------------- public filter‑wheel API --------------------
    def set_filter(self, filter: int | None = None):
        """
        Move the cube wheel to *filter* (numeric slot).
        Raises `RuntimeError` on time-out or protocol error.
        """
        if filter is None:
            self.info("OlympusLamp: No filter specified, skipping set_filter.")
            return

        # Determine number of slots available
        slots = (self._cube_slots
                 or self._cube_slots_override
                 or OlympusLamp._DEFAULT_SLOTS)

        # Wrap the request into the valid 1…slots range
        filter = ((filter - 1) % slots) + 1

        self._set_cube(filter)


    def get_filter(self) -> int | None:
        """Return current numeric cube slot (or *None* if indeterminate)."""
        if self._cube_cmd is None:
            return None
        try:
            slot = self._cube_pos()
            self._current_filter = slot
            return slot
        except RuntimeError:
            return self._current_filter   # last known – better than raising
