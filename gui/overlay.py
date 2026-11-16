"""
Overlay: a small always-on-top window in the top-left corner of the screen
that reads the Super Auto Pets shop and shows the moves a model suggests,
each with the model's confidence. It only suggests: you make the moves.

    python -m gui.overlay                  # pick the model in the window
    python -m gui.overlay --model mlp_19.3M --every 2

It reads the whole screen, so play with the game in a window (maximized is
fine) rather than macOS full screen, where other windows can't stay on top.
The first time, macOS asks to let Terminal record the screen (System
Settings > Privacy & Security > Screen Recording); allow it and restart.

Before it can read anything, set it up and teach it in the browser GUI's
Shop Reader tab (python -m gui.server, then the overlay's Set up button):
press Read here on the shop screen, which saves what the overlay saw for
the set-up page to load.

Needs Pillow (pip install pillow) for screen capture, and Tk (on a Mac with
Homebrew Python: brew install python-tk).
"""

import argparse
import os
import threading
import time
import webbrowser

import numpy as np

from gui import lab, shopread

TOP = 3
WIDTH = 340


################################################################################
#### Screen and advice (no window: these are tested on their own)
################################################################################


def grab_screen():
    """The whole screen as an RGB array"""
    from PIL import ImageGrab

    return np.asarray(ImageGrab.grab().convert("RGB"))


def pct(p):
    if p < 0.001:
        return "<0.1%"
    if p > 0.999:
        return ">99.9%"
    return f"{p * 100:.1f}%"


def status_line(counters):
    c = counters or {}
    parts = []
    for key, word in (
        ("turn", "Turn"),
        ("gold", "gold"),
        ("lives", "lives"),
        ("wins", "trophies"),
    ):
        value = c.get(key)
        value = "?" if value is None else value
        parts.append(f"{word} {value}" if key == "turn" else f"{value} {word}")
    return " · ".join(parts)


def summary(out, max_unsure=4):
    """What the window shows for one read: status, moves and a warning"""
    seen = out.get("seen") or {}
    unsure = seen.get("unsure") or []
    warning = ""
    if unsure:
        more = len(unsure) - max_unsure
        warning = "Not sure about: " + ", ".join(unsure[:max_unsure])
        if more > 0:
            warning += f" and {more} more"
        warning += ". Teach it in the Shop Reader."
    return {
        "status": status_line(seen.get("counters")),
        "moves": [(m["text"], pct(m["prob"]), m["prob"]) for m in out.get("moves", [])],
        "warning": warning,
    }


class Advisor:
    """Reads a screen with the saved layout and what has been learned, and
    asks the chosen model; picks up new lessons as they are saved"""

    def __init__(self, model):
        self.model = model
        self._memory = None
        self._memory_time = None

    def memory(self):
        path = lab.memory_path()
        stamp = os.path.getmtime(path) if os.path.exists(path) else None
        if self._memory is None or stamp != self._memory_time:
            self._memory = lab.Memory.load()
            self._memory_time = stamp
        return self._memory

    def advise(self, img, save=True):
        if save:
            shopread.save_capture(img)
        layout = shopread.load_layout()
        if layout is None:
            raise ValueError(
                "Not set up yet: press Set up, then follow the Shop Reader page"
            )
        return shopread.advise(img, self.model, TOP, layout, self.memory())


################################################################################
#### The window
################################################################################


class OverlayApp:
    def __init__(self, model=None, every=3.0, port=8405):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.models = shopread.list_models()
        if not self.models:
            raise SystemExit(f"No models found in {shopread.MODELS_DIR}")
        names = [n for n, _ in self.models]
        self.advisor = Advisor(model if model in names else names[0])
        self.every = every
        self.port = port
        self.busy = False

        root = self.root = tk.Tk()
        root.title("SAP advisor")
        root.geometry(f"{WIDTH}x250+0+0")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.93)
        except tk.TclError:
            pass
        root.resizable(True, True)

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        self.model_var = tk.StringVar(value=dict(self.models)[self.advisor.model])
        pick = ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=[label for _, label in self.models],
            state="readonly",
            width=30,
        )
        pick.pack(side="left", fill="x", expand=True)
        pick.bind("<<ComboboxSelected>>", self.on_model)

        row = ttk.Frame(root, padding=(6, 0))
        row.pack(fill="x")
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Auto", variable=self.auto_var).pack(side="left")
        ttk.Button(row, text="Read now", command=self.read).pack(side="left", padx=4)
        ttk.Button(row, text="Set up", command=self.setup).pack(side="left")

        self.status = ttk.Label(
            root, text="Press Read now on the shop screen", padding=(6, 4)
        )
        self.status.pack(fill="x")

        self.rows = []
        for k in range(TOP):
            f = ttk.Frame(root, padding=(6, 2))
            f.pack(fill="x")
            text = ttk.Label(f, text="", wraplength=WIDTH - 90, justify="left")
            text.pack(side="left", fill="x", expand=True)
            num = ttk.Label(f, text="", width=7, anchor="e")
            num.pack(side="right")
            bar = tk.Canvas(root, height=4, highlightthickness=0)
            bar.pack(fill="x", padx=6)
            self.rows.append((text, num, bar))

        self.warning = ttk.Label(
            root, text="", foreground="#b05a00", wraplength=WIDTH - 12, padding=(6, 4)
        )
        self.warning.pack(fill="x")

        root.after(500, self.tick)

    def on_model(self, _event=None):
        label = self.model_var.get()
        self.advisor.model = next(n for n, l in self.models if l == label)
        self.read()

    def setup(self):
        url = f"http://127.0.0.1:{self.port}/#shop"
        self.status.config(
            text=f"Set-up page: {url} (start python -m gui.server first)"
        )
        webbrowser.open(url)

    def tick(self):
        if self.auto_var.get() and not self.busy:
            self.read()
        self.root.after(int(self.every * 1000), self.tick)

    def capture(self):
        """The screen without this window on it (the window covers part of
        the game's top-left corner)"""
        try:
            self.root.attributes("-alpha", 0.0)
            self.root.update()
            time.sleep(0.05)
            return grab_screen()
        finally:
            self.root.attributes("-alpha", 0.93)

    def read(self):
        if self.busy:
            return
        self.busy = True
        try:
            img = self.capture()
        except Exception as e:
            self.busy = False
            self.show_error(f"Can't capture the screen: {e}")
            return

        def work():
            try:
                out = summary(self.advisor.advise(img))
                self.root.after(0, lambda: self.show(out))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda: self.show_error(msg))
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def show(self, s):
        self.status.config(text=s["status"])
        for k, (text, num, bar) in enumerate(self.rows):
            bar.delete("all")
            if k < len(s["moves"]):
                words, p_text, p = s["moves"][k]
                text.config(text=f"{k + 1}. {words}")
                num.config(text=p_text)
                bar.update_idletasks()
                w = max(bar.winfo_width(), 1)
                bar.create_rectangle(0, 0, int(w * p), 4, fill="#2e8b57", width=0)
            else:
                text.config(text="")
                num.config(text="")
        self.warning.config(text=s["warning"])

    def show_error(self, msg):
        self.status.config(text=msg)
        for text, num, bar in self.rows:
            text.config(text="")
            num.config(text="")
            bar.delete("all")
        self.warning.config(text="")

    def run(self):
        self.root.mainloop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", help="model to start with (a name in gui/models)")
    p.add_argument(
        "--every", type=float, default=3.0, help="seconds between reads when Auto is on"
    )
    p.add_argument(
        "--port",
        type=int,
        default=8405,
        help="port of python -m gui.server, for Set up",
    )
    args = p.parse_args(argv)
    OverlayApp(args.model, args.every, args.port).run()


if __name__ == "__main__":
    main()
