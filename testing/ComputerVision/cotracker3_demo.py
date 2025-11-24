# cotracker3_folder_player_auto_lazy.py
# Opens fast: lazy frame decoding + lazy model load. Auto-runs after clicks.
# Controls: left-click add (auto), right-click remove (auto), trackbar/arrows scrub, 'q' quit.

import glob, os
from functools import lru_cache
from natsort import natsorted
import numpy as np
import cv2
import torch
import imageio.v3 as iio
from collections import defaultdict

# ---------------------------
# Lazy .webp sequence (RGB)
# ---------------------------

class WebpSequence:
    def __init__(self, folder, resize_long_edge=None):
        self.paths = natsorted([p for p in glob.glob(os.path.join(folder, "*.webp"))])
        if not self.paths:
            raise FileNotFoundError(f"No .webp images found in: {folder}")
        self.resize_long_edge = resize_long_edge
        f0 = self.get(0)  # decode first frame only
        self.H, self.W = f0.shape[:2]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _resize(img, resize_long_edge):
        if not resize_long_edge:
            return img
        h, w = img.shape[:2]
        s = resize_long_edge / max(h, w)
        if s == 1.0:
            return img
        return cv2.resize(img, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)

    @lru_cache(maxsize=128)  # keep last ~128 frames in RAM
    def _decode_index(self, idx):
        img = iio.imread(self.paths[idx])  # RGB uint8
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = self._resize(img, self.resize_long_edge)
        return img

    def get(self, idx):
        # idx: 0..len-1
        return self._decode_index(idx)

# ---------------------------
# CoTracker3 Online wrapper (streaming)
# ---------------------------

def to_video_tensor(frames, device):
    # frames: list of RGB uint8 arrays [H,W,3]
    arr = np.stack(frames, axis=0)                 # T,H,W,3
    ten = torch.from_numpy(arr).float()            # 0..255
    ten = ten.permute(0,3,1,2)[None].to(device)    # 1,T,3,H,W
    return ten

class OnlineCoTrackerRunner:
    def __init__(self, device="cuda"):
        self.device = device
        # Lazy download/load happens here; defer creating this class until needed.
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device).eval()
        torch.backends.cudnn.benchmark = True
        self.step = self.model.step

    @torch.inference_mode()
    def run_streaming(self, seq: WebpSequence, queries_txy, add_support_grid=True):
        """
        seq: WebpSequence (RGB, lazy)
        queries_txy: np.float32 [N,3] (t, x, y) at the sequence resolution
        returns tracks [T,N,2], vis [T,N]
        """
        if queries_txy is None or len(queries_txy) == 0:
            raise ValueError("No queries provided. Click points to track.")
        device = self.device
        T = len(seq)

        queries = torch.as_tensor(queries_txy, dtype=torch.float32, device=device)[None]
        N = int(queries.shape[1])

        # Warmup: first 2*step frames (or fewer)
        first_end = min(T, self.step * 2)
        vid0 = to_video_tensor([seq.get(i) for i in range(0, first_end)], device)
        _ = self.model(video_chunk=vid0, is_first_step=True,
                       grid_size=0, queries=queries, add_support_grid=add_support_grid)

        tracks_out = np.zeros((T, N, 2), dtype=np.float32)
        vis_out    = np.zeros((T, N), dtype=np.float32)

        # Slide over by `step`, feeding only the needed chunk each time
        for ind in range(0, T - self.step, self.step):
            end = min(ind + self.step * 2, T)
            vid = to_video_tensor([seq.get(i) for i in range(ind, end)], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                pred_tracks, pred_vis = self.model(
                    video_chunk=vid,
                    grid_size=0,
                    queries=queries,
                    add_support_grid=add_support_grid
                )
            chunk_len = pred_tracks.shape[1]
            pt = pred_tracks[0].float().detach().cpu().numpy()        # [t_chunk,N,2]
            pv = pred_vis[0, :, :, 0].float().detach().cpu().numpy()  # [t_chunk,N]
            tracks_out[ind:ind+chunk_len, :, :] = pt[:min(chunk_len, T-ind)]
            vis_out[ind:ind+chunk_len, :]       = pv[:min(chunk_len, T-ind)]

        return tracks_out, vis_out

# ---------------------------
# Interactive player (uses lazy sequence)
# ---------------------------

def draw_text(img, text, org=(10,20)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def distinct_color(i, n=20):
    hue = (i % n) / float(n)
    color = np.array(cv2.cvtColor(np.uint8([[[int(hue*180), 200, 255]]]), cv2.COLOR_HSV2BGR)[0,0])
    return (int(color[0]), int(color[1]), int(color[2]))  # BGR

class Player:
    def __init__(self, seq: WebpSequence, window="CoTracker3", add_support_grid=True):
        self.seq = seq
        self.T = len(seq)
        self.H, self.W = seq.H, seq.W
        self.window = window
        self.idx = 0
        self.add_support_grid = add_support_grid

        self.queries = []         # list[(t,x,y)] ; None if deleted
        self.colors = []          # per-query color
        self.queries_by_t = defaultdict(list)
        self.tracks = None        # [T,N,2]
        self.vis = None           # [T,N]

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.window, min(1280, self.W), min(720, self.H))
        cv2.createTrackbar("frame", self.window, 0, self.T-1, self._on_seek)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_seek(self, val):
        self.idx = int(val)
        self.refresh()

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            qi = len(self.queries)
            self.queries.append((float(self.idx), float(x), float(y)))
            self.colors.append(distinct_color(qi))
            self.queries_by_t[self.idx].append(qi)
            self.tracks = self.vis = None  # invalidate; main loop will auto-run
            self.refresh()
        elif event == cv2.EVENT_RBUTTONDOWN:
            ids = self.queries_by_t.get(self.idx, [])
            if not ids:
                return
            pts = np.array([self.queries[i][1:3] for i in ids], dtype=np.float32)
            d = np.sqrt(((pts - np.array([x, y], dtype=np.float32))**2).sum(axis=1))
            j = int(np.argmin(d))
            rem_qi = ids[j]
            self.queries[rem_qi] = None
            self.colors[rem_qi] = None
            self.queries_by_t[self.idx] = [i for i in ids if i != rem_qi]
            self.tracks = self.vis = None
            self.refresh()

    def _draw(self):
        frame_rgb = self.seq.get(self.idx)
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for qi in self.queries_by_t.get(self.idx, []):
            if self.queries[qi] is None: continue
            _, x, y = self.queries[qi]
            cv2.circle(img, (int(x), int(y)), 4, self.colors[qi], -1)
            cv2.putText(img, f"{qi}", (int(x)+6, int(y)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, f"{qi}", (int(x)+6, int(y)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        if self.tracks is not None:
            xy = self.tracks[self.idx]; vis = self.vis[self.idx]
            for qi in range(xy.shape[0]):
                if qi >= len(self.queries) or self.queries[qi] is None:
                    continue
                if float(vis[qi]) > 0.5:
                    cv2.circle(img, (int(xy[qi,0]), int(xy[qi,1])), 3, self.colors[qi], -1)
                else:
                    cv2.circle(img, (int(xy[qi,0]), int(xy[qi,1])), 3, (128,128,128), 1)
        draw_text(img, f"Frame {self.idx+1}/{self.T} | q: quit | left-click: add (auto) | right-click: remove (auto) | arrows/trackbar: scrub")
        return img

    def refresh(self):
        img = self._draw()
        cv2.imshow(self.window, img)
        cv2.setTrackbarPos("frame", self.window, self.idx)

    def collect_queries_txy(self):
        compact, new_colors = [], []
        self.queries_by_t = defaultdict(list)
        for i,q in enumerate(self.queries):
            if q is None: continue
            compact.append(q); new_colors.append(self.colors[i])
        self.queries = [tuple(q) for q in compact]
        self.colors = new_colors
        for i,q in enumerate(self.queries):
            t = int(round(q[0])); self.queries_by_t[t].append(i)
        return np.array(self.queries, dtype=np.float32)

# ---------------------------
# Main (hardcoded defaults)
# ---------------------------

def main():
    # >>> set your defaults here <<<
    folder = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\rig_recorder_data\2025_07_31-17_11\camera_frames"   # use an absolute path
    resize_long_edge = 256               # 0 = keep original resolution
    add_support_grid = True
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    seq = WebpSequence(folder, resize_long_edge=None if resize_long_edge==0 else resize_long_edge)
    player = Player(seq, add_support_grid=add_support_grid)

    runner = None  # lazy model load
    player.refresh()
    while True:
        # Auto-run tracking whenever points exist and results are invalidated
        if player.tracks is None and any(q is not None for q in player.queries):
            if runner is None:
                print("[cotracker] Loading CoTracker3 (first time may download)…")
                runner = OnlineCoTrackerRunner(device="cuda" if torch.cuda.is_available() else "cpu")
                print("[cotracker] Model ready.")
            try:
                queries = player.collect_queries_txy()
                tracks, vis = runner.run_streaming(player.seq, queries, add_support_grid=player.add_support_grid)
                player.tracks, player.vis = tracks, vis
                player.refresh()
            except Exception as e:
                print("Tracking error:", e)
                player.tracks = player.vis = None
                player.refresh()

        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        elif k == 81:  # Left arrow
            player.idx = max(0, player.idx-1); player.refresh()
        elif k == 83:  # Right arrow
            player.idx = min(player.T-1, player.idx+1); player.refresh()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

