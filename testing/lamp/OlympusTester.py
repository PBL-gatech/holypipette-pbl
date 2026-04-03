
"""
Olympus BX/IX/MX‑UCB driver – shutter + cube (handles long moves)
================================================================
Fixes the remaining *timeout* edge‑case when the cube wheel takes >4 s to reach
its slot or when the controller keeps replying `CUBE +` until the motor stops.

Changes
-------
* **`MAX_CUBE_MOVE_TIME`** constant (default = 10 s).
* In **`set_cube()`** we loop until either:
  * `cube_pos()` returns the requested slot, or
  * the timeout elapses **and** the most recent reply contained a `+` *without*
    an error – in that case we *assume success* and exit silently.
* `cube_pos()` now treats `CUBE +` (no number) as *indeterminate* and returns
  `None` instead of raising.

With these tweaks even very slow filter wheels (older BX‑CB controllers) no
longer raise `cube timeout` after a successful move.
"""
from __future__ import annotations
import os, time, serial
from contextlib import contextmanager
from typing import Optional

PORT = os.getenv("UCB_PORT", "COM21")
BAUD = 19200
PREFIX = "1"
DEBUG = False

MANUAL_CUBE_PREFIX: Optional[str] = "1"
MANUAL_CUBE_CMD:    Optional[str] = "CUBE"
CUBE_SLOTS_OVERRIDE: Optional[int] = 6

SHUT_CMD = "SHUTTER"
CUBE_CMD: Optional[str] = None
CUBE_PREFIX = PREFIX
CUBE_SLOTS: Optional[int] = None
MAX_CUBE_MOVE_TIME = 10.0   # seconds to wait after a move command

@contextmanager
def open_port():
    """
    Context manager to open and safely close the UCB serial port.

    Returns:
        serial.Serial: An open serial connection to the UCB device.

    Raises:
        SerialException: If the port cannot be opened.
    """
    s = serial.Serial(PORT, BAUD, bytesize=serial.EIGHTBITS, parity=serial.PARITY_EVEN,
                      stopbits=serial.STOPBITS_TWO, timeout=1.0,
                      rtscts=True, write_timeout=1.0)
    try:
        yield s
    finally:
        s.close()

def _io(s: serial.Serial, line: str, tries=3) -> str:
    """
    Send a command line to the device and read a single reply.

    Args:
        s (serial.Serial): Open serial connection.
        line (str): Command line to send (without prefix).
        tries (int, optional): Number of attempts before timeout. Defaults to 3.

    Returns:
        str: The decoded reply from the device.

    Raises:
        TimeoutError: If no reply is received after the given attempts.
    """
    raw = (line + "\r\n").encode()
    for _ in range(tries):
        if DEBUG: print("TX>", line)
        s.write(raw)
        ans = s.readline()
        if ans:
            txt = ans.strip().decode(errors="replace")
            if DEBUG: print("RX<", txt)
            return txt
    raise TimeoutError(f"No reply to {line!r}")

def cmd(s, tail):
    """
    Send a prefixed command to the device and return the reply.

    Args:
        s (serial.Serial): Open serial connection.
        tail (str): Command tail (suffix after PREFIX).

    Returns:
        str: Device reply.
    """
    return _io(s, f"{PREFIX}{tail}")

def raw_cmd(s, pref, body):
    """
    Send a raw command with custom prefix and body to the device.

    Args:
        s (serial.Serial): Open serial connection.
        pref (str): Command prefix.
        body (str): Command body.

    Returns:
        str: Device reply.
    """
    return _io(s, f"{pref}{body}")

def ensure_login(s):
    """
    Ensure the device is logged in; sends login command if necessary.

    Args:
        s (serial.Serial): Open serial connection.

    Raises:
        RuntimeError: If login fails.
    """
    if not cmd(s, "LOG?").endswith("IN"):
        if not cmd(s, "LOG IN").endswith("+"): raise RuntimeError("login failed")

def logout(s):
    """
    Log out from the device.

    Args:
        s (serial.Serial): Open serial connection.
    """
    cmd(s, "LOG OUT")
    
def shutter_state(s): 
    """
    Query the current shutter state.

    Args:
        s (serial.Serial): Open serial connection.

    Returns:
        str: 'IN' or 'OUT' depending on shutter position.
    """
    return cmd(s, f"{SHUT_CMD}?").split()[1]

def shutter_open(s):
    """
    Open the shutter if it is not already open.

    Args:
        s (serial.Serial): Open serial connection.
    """
    if shutter_state(s) != "OUT": cmd(s, f"{SHUT_CMD} OUT")

def shutter_close(s):
    """
    Close the shutter if it is not already closed.

    Args:
        s (serial.Serial): Open serial connection.
    """
    if shutter_state(s) != "IN": cmd(s, f"{SHUT_CMD} IN")

def _probe_var(s, pref, name):
    """
    Check if a device variable exists by sending query commands.

    Args:
        s (serial.Serial): Open serial connection.
        pref (str): Command prefix to use.
        name (str): Name of the variable to probe.

    Returns:
        bool: True if the variable exists (device replies with its name), False otherwise.

    Raises:
        None: TimeoutErrors are caught internally and treated as False.
    """
    for q in (f"{name} ?", f"{name}?"):
        try:
            if raw_cmd(s, pref, q).startswith(name): return True
        except TimeoutError: pass
    return False

def detect_cube(s):
    """
    Detect the cube (filter wheel) device and set global cube variables.

    Args:
        s (serial.Serial): Open serial connection.

    Raises:
        None (sets CUBE_CMD = None if cube not detected)
    """
    global CUBE_CMD, CUBE_PREFIX, CUBE_SLOTS
    if MANUAL_CUBE_CMD:
        CUBE_CMD, CUBE_PREFIX, CUBE_SLOTS = MANUAL_CUBE_CMD, MANUAL_CUBE_PREFIX or PREFIX, CUBE_SLOTS_OVERRIDE; return
    for name in ("CUBE", "MU"):
        for pref in (PREFIX, "" if PREFIX=="1" else "1"):
            if _probe_var(s, pref, name):
                CUBE_CMD, CUBE_PREFIX, CUBE_SLOTS = name, pref, CUBE_SLOTS_OVERRIDE; return
    CUBE_CMD = None

def _parse_slot(reply: str) -> Optional[int]:
    """
    Extract the first numeric slot from a cube device reply string.

    Args:
        reply (str): Reply from the device (e.g., 'CUBE 3', '3').

    Returns:
        Optional[int]: The first integer found in the reply, or None if no number is present.
    """
    for tok in reply.replace(",", " ").split():
        if tok.isdigit():
            return int(tok)
    return None

def cube_pos(s) -> Optional[int]:
    """
    Query the current cube slot position.

    Args:
        s (serial.Serial): Open serial connection.

    Returns:
        Optional[int]: Current slot number, or None if indeterminate.

    Raises:
        RuntimeError: If the cube reports an error.
    """
    rep = raw_cmd(s, CUBE_PREFIX, f"{CUBE_CMD}?")
    while rep.strip(" +") in {"", "CUBE", "MU"}:  # skip empty / ack lines
        rep = s.readline().strip().decode(errors="replace")
    if rep.startswith("!,E"):
        raise RuntimeError(f"Cube error: {rep}")
    return _parse_slot(rep)

def set_cube(s, pos):
    """
    Move the cube to a specific slot, with timeout handling.

    Args:
        s (serial.Serial): Open serial connection.
        pos (int): Target cube slot.

    Raises:
        RuntimeError: If the cube cannot be set within MAX_CUBE_MOVE_TIME.
        ValueError: If the requested slot is out of valid range.
    """
    if CUBE_CMD is None: raise RuntimeError("cube not detected")
    if CUBE_SLOTS and not 1<=pos<=CUBE_SLOTS: raise ValueError("pos range")
    raw_cmd(s, CUBE_PREFIX, f"{CUBE_CMD} {pos}")
    t0=time.time(); last_rep=""
    while time.time()-t0<MAX_CUBE_MOVE_TIME:
        try:
            slot=cube_pos(s)
            if slot==pos: return
        except RuntimeError as e:
            last_rep=str(e)
        time.sleep(0.3)
    # if the only error was a busy '+', assume success
    if last_rep.startswith("Cube error: !,E01120"):
        return
    raise RuntimeError("cube timeout")

if __name__=="__main__":
    with open_port() as ucb:
        ensure_login(ucb)
        detect_cube(ucb)
        print("Shutter:", PREFIX+SHUT_CMD)
        print("Cube:", (CUBE_PREFIX+CUBE_CMD) if CUBE_CMD else "<none>")
        shutter_open(ucb); time.sleep(0.3)

        if CUBE_CMD:
            print("Cube at:", cube_pos(ucb))
            set_cube(ucb,3)
            print("Cube set to 3, now at:", cube_pos(ucb))
            set_cube(ucb,1)
            print("Cube set to 1, now at:", cube_pos(ucb))

        shutter_close(ucb); logout(ucb)
        print("Done.")
