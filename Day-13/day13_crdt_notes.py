# Day 13 — CRDT Notes 📝
# Author: Stuart Abhishek
#
# Offline-first notes that automatically merge using an LWW-Register CRDT per field.
# - Lamport clock (no wall-clock dependence)
# - Per-note fields: title, body, deleted -> each is (value, (timestamp, node_id))
# - Merge is commutative/associative/idempotent -> eventual consistency
# - Tombstones (deleted=True) resolve properly by LWW semantics
# - Simple TCP sync: peers exchange full state as JSON and both converge
#
# Usage examples:
#   python day13_crdt_notes.py init --node-id A
#   python day13_crdt_notes.py new "Groceries" "Milk, eggs, bread"
#   python day13_crdt_notes.py list
#   python day13_crdt_notes.py edit <note_id> --title "New Title"
#   python day13_crdt_notes.py show <note_id>
#   python day13_crdt_notes.py delete <note_id>
#   python day13_crdt_notes.py export-md  # dumps all non-deleted notes to Day-13/notes.md
#   # sync two replicas (run server on one side)
#   python day13_crdt_notes.py serve --port 9500
#   python day13_crdt_notes.py sync --host 127.0.0.1 --port 9500
#
# Concepts: CRDTs, eventual consistency, Lamport timestamps, LWW registers, tombstones, idempotent merge.

import argparse
import json
import os
import socket
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

DATA_DIR = Path("Day-13")
STATE_FILE = DATA_DIR / "notes_state.json"
NODE_FILE = DATA_DIR / "node.json"
MD_EXPORT = DATA_DIR / "notes.md"

# ----------------------------- Utilities -----------------------------

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ----------------------------- Lamport Clock -----------------------------

@dataclass
class LamportClock:
    time: int
    node: str

    def tick(self):
        self.time += 1
        return self.time

    def observe(self, other_time: int):
        # Lamport: T := max(T, other) + 1
        self.time = max(self.time, other_time) + 1
        return self.time

# ----------------------------- LWW Register Ops -----------------------------
# We’ll store each field as: {"value": ..., "ts": int, "node": str}
# Ordering: (ts, node) lexicographic; larger wins.

def lww_new(value, ts: int, node: str):
    return {"value": value, "ts": ts, "node": node}

def lww_merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    ka = (a["ts"], a["node"])
    kb = (b["ts"], b["node"])
    return a if ka >= kb else b

# ----------------------------- CRDT State -----------------------------
# State schema:
# {
#   "meta": {"node": "A", "lamport": 42},
#   "notes": {
#       "<note_id>": {
#           "title":   {"value":"...", "ts":N, "node":"A"},
#           "body":    {"value":"...", "ts":M, "node":"A"},
#           "deleted": {"value":False, "ts":K, "node":"A"}
#       },
#       ...
#   }
# }

def empty_state(node_id: str) -> dict:
    return {
        "meta": {"node": node_id, "lamport": 0},
        "notes": {}
    }

def get_clock(state: dict) -> LamportClock:
    return LamportClock(time=state["meta"]["lamport"], node=state["meta"]["node"])

def set_clock(state: dict, clk: LamportClock):
    state["meta"]["lamport"] = clk.time
    state["meta"]["node"] = clk.node

def crdt_merge(local: dict, remote: dict) -> dict:
    # Merge metadata clock conservatively (max)
    node_local = local["meta"]["node"]
    node_remote = remote["meta"]["node"]
    lamport = max(local["meta"]["lamport"], remote["meta"]["lamport"])
    out = {
        "meta": {"node": node_local, "lamport": lamport},
        "notes": {}
    }
    # Merge note sets by key union
    all_ids = set(local["notes"].keys()) | set(remote["notes"].keys())
    for nid in all_ids:
        L = local["notes"].get(nid)
        R = remote["notes"].get(nid)
        if L is None:
            out["notes"][nid] = R
            continue
        if R is None:
            out["notes"][nid] = L
            continue
        # field-wise LWW
        title = lww_merge(L.get("title"), R.get("title"))
        body = lww_merge(L.get("body"), R.get("body"))
        deleted = lww_merge(L.get("deleted"), R.get("deleted"))
        out["notes"][nid] = {"title": title, "body": body, "deleted": deleted}
    return out

# ----------------------------- Local Ops -----------------------------

def load_state() -> dict:
    ensure_dirs()
    info = read_json(NODE_FILE, {})
    if not info:
        raise SystemExit("Not initialized. Run: python day13_crdt_notes.py init --node-id <ID>")
    node_id = info["node"]
    state = read_json(STATE_FILE, None)
    if state is None:
        state = empty_state(node_id)
        write_json(STATE_FILE, state)
    return state

def save_state(state: dict):
    write_json(STATE_FILE, state)

def init(node_id: str):
    ensure_dirs()
    if NODE_FILE.exists():
        print("Already initialized.")
        return
    write_json(NODE_FILE, {"node": node_id})
    write_json(STATE_FILE, empty_state(node_id))
    print(f"✅ Initialized CRDT Notes with node id '{node_id}'")

def new_note(title: str, body: str):
    state = load_state()
    clk = get_clock(state)
    clk.tick()
    nid = str(uuid.uuid4())
    state["notes"][nid] = {
        "title": lww_new(title, clk.time, clk.node),
        "body": lww_new(body, clk.time, clk.node),
        "deleted": lww_new(False, clk.time, clk.node),
    }
    set_clock(state, clk)
    save_state(state)
    print(f"🆕 Created note {nid}")

def edit_note(nid: str, title: Optional[str], body: Optional[str]):
    state = load_state()
    if nid not in state["notes"]:
        raise SystemExit("Note not found.")
    clk = get_clock(state)
    if title is not None:
        clk.tick()
        state["notes"][nid]["title"] = lww_new(title, clk.time, clk.node)
    if body is not None:
        clk.tick()
        state["notes"][nid]["body"] = lww_new(body, clk.time, clk.node)
    set_clock(state, clk)
    save_state(state)
    print(f"✏️  Edited note {nid}")

def delete_note(nid: str):
    state = load_state()
    if nid not in state["notes"]:
        raise SystemExit("Note not found.")
    clk = get_clock(state)
    clk.tick()
    state["notes"][nid]["deleted"] = lww_new(True, clk.time, clk.node)
    set_clock(state, clk)
    save_state(state)
    print(f"🗑️  Deleted note {nid} (tombstoned)")

def show_note(nid: str):
    state = load_state()
    note = state["notes"].get(nid)
    if not note:
        raise SystemExit("Note not found.")
    if note["deleted"]["value"]:
        print("(deleted)")
    print(f"# {note['title']['value']}\n\n{note['body']['value']}\n")

def list_notes(include_deleted=False):
    state = load_state()
    for nid, note in state["notes"].items():
        if not include_deleted and note["deleted"]["value"]:
            continue
        title = note["title"]["value"]
        flag = " (deleted)" if note["deleted"]["value"] else ""
        print(f"- {nid} :: {title}{flag}")

def export_md():
    state = load_state()
    lines = ["# CRDT Notes Export\n"]
    for nid, note in state["notes"].items():
        if note["deleted"]["value"]:
            continue
        lines.append(f"## {note['title']['value']}  \n_id: {nid}\n")
        lines.append(note["body"]["value"])
        lines.append("\n---\n")
    MD_EXPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"📤 Exported Markdown to {MD_EXPORT}")

# ----------------------------- Sync (TCP JSON) -----------------------------

def send_json(sock: socket.socket, obj: dict):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)

def recv_json(sock: socket.socket) -> Optional[dict]:
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            return None
        if b == b"\n":
            break
        buf += b
    return json.loads(buf.decode("utf-8"))

def serve(port: int):
    state = load_state()  # ensure initialized
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    print(f"🌐 CRDT Notes server on 0.0.0.0:{port}")
    try:
        while True:
            conn, addr = s.accept()
            try:
                req = recv_json(conn)
                if not req or req.get("type") != "SYNC":
                    send_json(conn, {"type":"ERROR","msg":"bad handshake"})
                    conn.close(); continue
                # send our state, receive theirs, merge both ways
                local = read_json(STATE_FILE, empty_state(read_json(NODE_FILE, {})["node"]))
                send_json(conn, {"type":"STATE","payload": local})
                remote = recv_json(conn)
                if not remote or remote.get("type") != "STATE":
                    send_json(conn, {"type":"ERROR","msg":"no state"})
                    conn.close(); continue
                merged = crdt_merge(local, remote["payload"])
                # Persist locally
                write_json(STATE_FILE, merged)
                # Return merged so the peer also converges
                send_json(conn, {"type":"MERGED","payload": merged})
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nServer shutting down.")

def sync(host: str, port: int):
    _ = load_state()  # ensure initialized
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((host, port))
    # handshake
    send_json(c, {"type":"SYNC"})
    # receive remote's STATE request reply, send ours
    hdr = recv_json(c)
    if not hdr or hdr.get("type") != "STATE":
        raise SystemExit("Protocol error (expected STATE)")
    remote_state = hdr["payload"]
    local_state = read_json(STATE_FILE, empty_state(read_json(NODE_FILE, {})["node"]))
    send_json(c, {"type":"STATE","payload": local_state})
    # receive merged state from server
    merged_msg = recv_json(c)
    if not merged_msg or merged_msg.get("type") != "MERGED":
        raise SystemExit("Protocol error (expected MERGED)")
    merged_state = merged_msg["payload"]
    write_json(STATE_FILE, merged_state)
    c.close()
    print("🔁 Sync complete. States converged.")

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Day 13 — CRDT Notes (LWW + Lamport + Sync)")
    sub = ap.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init")
    p_init.add_argument("--node-id", required=True, help="Unique replica id (e.g., A, phone1, laptop)")

    p_new = sub.add_parser("new")
    p_new.add_argument("title")
    p_new.add_argument("body")

    p_edit = sub.add_parser("edit")
    p_edit.add_argument("id")
    p_edit.add_argument("--title")
    p_edit.add_argument("--body")

    p_del = sub.add_parser("delete")
    p_del.add_argument("id")

    sub.add_parser("list")
    p_list_all = sub.add_parser("list-all")

    p_show = sub.add_parser("show")
    p_show.add_argument("id")

    sub.add_parser("export-md")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=9500)

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--host", required=True)
    p_sync.add_argument("--port", type=int, default=9500)

    args = ap.parse_args()
    if args.cmd == "init":
        init(args.node_id)
    elif args.cmd == "new":
        new_note(args.title, args.body)
    elif args.cmd == "edit":
        edit_note(args.id, args.title, args.body)
    elif args.cmd == "delete":
        delete_note(args.id)
    elif args.cmd == "list":
        list_notes(False)
    elif args.cmd == "list-all":
        list_notes(True)
    elif args.cmd == "show":
        show_note(args.id)
    elif args.cmd == "export-md":
        export_md()
    elif args.cmd == "serve":
        serve(args.port)
    elif args.cmd == "sync":
        sync(args.host, args.port)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()