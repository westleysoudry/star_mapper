"""
Researcher Mapper — Windows GUI

Double-click  "Researcher Mapper.bat"  (or run this file directly) to open.
No command line needed.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ── Queue-based log handler so pipeline messages appear in the GUI ─────────────

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(self.format(record))


# ── Main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Researcher Mapper")
        self.resizable(True, True)
        self.minsize(620, 640)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._stop_event = threading.Event()

        self._build_ui()
        self._poll_log_queue()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ── Input frame ───────────────────────────────────────────────────────
        input_frame = ttk.LabelFrame(self, text="Researcher", padding=8)
        input_frame.pack(fill="x", **pad)
        input_frame.columnconfigure(1, weight=1)

        self._vars: dict[str, tk.Variable] = {}

        rows = [
            ("Name *",        "name",        "str",  ""),
            ("Institution",   "institution", "str",  ""),
            ("ORCID",         "orcid",       "str",  ""),
            ("Faculty page",  "faculty_page","str",  ""),
        ]
        for row_idx, (label, key, _type, default) in enumerate(rows):
            ttk.Label(input_frame, text=label).grid(
                row=row_idx, column=0, sticky="w", padx=4, pady=2
            )
            v = tk.StringVar(value=default)
            self._vars[key] = v
            ttk.Entry(input_frame, textvariable=v, width=55).grid(
                row=row_idx, column=1, sticky="ew", padx=4, pady=2
            )

        # CV row with browse button
        cv_row = len(rows)
        ttk.Label(input_frame, text="CV file").grid(
            row=cv_row, column=0, sticky="w", padx=4, pady=2
        )
        cv_frame = ttk.Frame(input_frame)
        cv_frame.grid(row=cv_row, column=1, sticky="ew", padx=4, pady=2)
        cv_frame.columnconfigure(0, weight=1)
        self._vars["cv"] = tk.StringVar()
        ttk.Entry(cv_frame, textvariable=self._vars["cv"]).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(cv_frame, text="Browse…", command=self._browse_cv).grid(
            row=0, column=1, padx=(4, 0)
        )

        # ── Settings frame ────────────────────────────────────────────────────
        settings_frame = ttk.LabelFrame(self, text="Settings", padding=8)
        settings_frame.pack(fill="x", **pad)
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        int_settings = [
            ("From year", "from_year", 2014, 0, 0, 1900, 9999),
            ("Max works", "max_works", 400, 0, 2, 1, 5000),
            ("Pool size (0=auto)", "pool_size", 0, 0, 4, 0, 5000),
            ("List size", "list_size", 20, 1, 0, 1, 100),
            ("Star topics", "star_map_topics", 5, 1, 2, 2, 10),
        ]
        for label, key, default, row, col, min_val, max_val in int_settings:
            ttk.Label(settings_frame, text=label).grid(
                row=row, column=col, sticky="w", padx=4, pady=2
            )
            v = tk.IntVar(value=default)
            self._vars[key] = v
            ttk.Spinbox(
                settings_frame, textvariable=v,
                from_=min_val, to=max_val, width=8
            ).grid(row=row, column=col + 1, sticky="w", padx=4, pady=2)

        # Output dir row
        out_row = 2
        ttk.Label(settings_frame, text="Output folder").grid(
            row=out_row, column=0, sticky="w", padx=4, pady=2
        )
        out_frame = ttk.Frame(settings_frame)
        out_frame.grid(row=out_row, column=1, columnspan=5, sticky="ew", padx=4, pady=2)
        out_frame.columnconfigure(0, weight=1)
        self._vars["output_dir"] = tk.StringVar(value="output")
        ttk.Entry(out_frame, textvariable=self._vars["output_dir"]).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).grid(
            row=0, column=1, padx=(4, 0)
        )

        # ── Button row ────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        ttk.Button(btn_frame, text="Load from file…", command=self._load_file).pack(
            side="left", padx=4
        )
        ttk.Button(btn_frame, text="Save to file…", command=self._save_file).pack(
            side="left", padx=4
        )
        self._run_btn = ttk.Button(
            btn_frame, text="▶  Run", command=self._run, style="Accent.TButton"
        )
        self._run_btn.pack(side="right", padx=4)
        self._stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self._stop, state="disabled"
        )
        self._stop_btn.pack(side="right", padx=4)

        # ── Log area ──────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True, **pad)

        self._log_text = scrolledtext.ScrolledText(
            log_frame, height=14, state="disabled",
            font=("Consolas", 9), wrap="word",
        )
        self._log_text.pack(fill="both", expand=True)

        self._status = ttk.Label(self, text="Ready", anchor="w")
        self._status.pack(fill="x", padx=8, pady=(0, 4))

    # ── Button handlers ───────────────────────────────────────────────────────

    def _browse_cv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CV file",
            filetypes=[
                ("CV files", "*.pdf *.txt"),
                ("PDF",      "*.pdf"),
                ("Text",     "*.txt"),
                ("All files","*.*"),
            ],
        )
        if path:
            self._vars["cv"].set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._vars["output_dir"].set(path)

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load researcher file",
            filetypes=[
                ("All supported", "*.txt *.yaml *.yml *.json"),
                ("Text file", "*.txt"),
                ("YAML file", "*.yaml *.yml"),
                ("JSON file", "*.json"),
            ],
        )
        if not path:
            return
        try:
            from researcher_mapper.pipelines.run_target import load_input_file
            cfg = load_input_file(path)
            for key, var in self._vars.items():
                if key in cfg:
                    val = cfg[key]
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(val))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(val))
                    else:
                        var.set(str(val))
            self._log(f"Loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _save_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save researcher file",
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"),
                ("YAML file", "*.yaml"),
                ("JSON file", "*.json"),
            ],
        )
        if not path:
            return
        cfg = self._collect_config()
        try:
            _write_config_file(cfg, Path(path))
            self._log(f"Saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _collect_config(self) -> dict:
        return {k: v.get() for k, v in self._vars.items()}

    def _run(self) -> None:
        if self._running:
            return
        cfg = self._collect_config()
        if not cfg.get("name", "").strip():
            messagebox.showwarning("Missing name", "Please enter the researcher's name.")
            return

        self._running = True
        self._stop_event.clear()
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._clear_log()
        self._status.config(text="Running…")

        # Install queue-based log handler
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = _QueueHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
        root_logger.addHandler(handler)
        self._log_handler = handler

        self._thread = threading.Thread(
            target=self._pipeline_thread, args=(cfg, self._stop_event), daemon=True
        )
        self._thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self._status.config(text="Stopping after current step…")

    def _pipeline_thread(self, cfg: dict, stop_event: threading.Event) -> None:
        try:
            from researcher_mapper.pipelines.run_target import run_target
            result = run_target(
                researcher_name=cfg["name"],
                institution=cfg.get("institution") or None,
                orcid=cfg.get("orcid") or None,
                output_dir=Path(cfg.get("output_dir") or "output"),
                from_year=int(cfg.get("from_year") or 2014),
                max_works=int(cfg.get("max_works") or 400),

                pool_size=int(cfg["pool_size"]) if int(cfg.get("pool_size") or 0) > 0 else None,
                faculty_page_url=cfg.get("faculty_page") or None,
                cv_path=cfg.get("cv") or None,
                list_size=int(cfg.get("list_size") or 20),
                star_map_topics=int(cfg.get("star_map_topics") or 5),
                stop_event=stop_event,
            )
            out_dir = result.get("graph_path", "output")
            self._log_queue.put(f"\n✓ Done.  Results saved to: {Path(out_dir).parent}")
            self.after(0, lambda: self._status.config(
                text=f"Done — results in {Path(out_dir).parent}"
            ))
        except InterruptedError:
            self._log_queue.put("Stopped by user.")
            self.after(0, lambda: self._status.config(text="Stopped."))
        except Exception as exc:
            _msg = str(exc)
            self._log_queue.put(f"ERROR: {_msg}")
            self.after(0, lambda m=_msg: self._status.config(text=f"Error: {m}"))
        finally:
            self._running = False
            self.after(0, self._reset_buttons)
            # Remove the log handler we added
            logging.getLogger().removeHandler(getattr(self, "_log_handler", None))

    def _reset_buttons(self) -> None:
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self) -> None:
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _poll_log_queue(self) -> None:
        """Drain the log queue and write messages to the text widget."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)


# ── Config file writer ────────────────────────────────────────────────────────

def _write_config_file(cfg: dict, path: Path) -> None:
    if path.suffix.lower() == ".json":
        import json
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        return

    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, allow_unicode=True, default_flow_style=False)
        return

    # Default: plain .txt
    lines = [
        "# Researcher Mapper — input file\n",
        "# Edit the values below, then open 'Researcher Mapper.bat'\n\n",
    ]
    labels = {
        "name":                       "name",
        "institution":                "institution",
        "orcid":                      "orcid",
        "faculty_page":               "faculty_page",
        "cv":                         "cv",
        "from_year":                  "from_year",
        "max_works":                  "max_works",
        "pool_size":                  "pool_size",
        "list_size":                  "list_size",
        "star_map_topics":            "star_map_topics",
        "output_dir":                 "output_dir",
    }
    for key, label in labels.items():
        val = cfg.get(key, "")
        lines.append(f"{label}: {val}\n")
    path.write_text("".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
