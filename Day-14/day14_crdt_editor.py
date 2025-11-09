# Day 14 — CRDT Collaborative Editor ✍️
# Author: Stuart Abhishek
#
# A minimal sequence-CRDT (Logoot/LSEQ-style) for collaborative text editing:
# - Each character is assigned a *position identifier* (list of (digit, site) pairs)
# - Total order is by lexicographic comparison of identifiers
# - Concurrent inserts allocate positions *between* neighbors by random digits, growing depth as needed
# - Deletes add a tombstone (idempotent, causal-safe)
# - Operations carry Lamport timestamps and unique op_ids; merges are commutative/associative/idempotent
# - Tiny TCP JSON protocol for state sync (like Day 13)
#
# Usage quickstart:
#   python day14_crdt_editor.py init --site A
#   python day14_crdt_editor.py insert 0 "H"
#   python day14_crdt_editor.py insert 1 "i"
#   python day14_crdt_editor.py show
#   python day14_crdt_editor.py serve --port 9600
#   # on another replica:
#   python day14_crdt_editor.py init --site B
#   python day14_crdt_editor.py insert 0 "H"; python day14_crdt_editor.py insert 1 "!"
#   python day14_crdt_editor.py sync --host 127.0.0.1 --port 9600
#   python day14_crdt_editor.py show
#
# Concepts: CRDTs for sequences (Logoot/LSEQ), Lamport clocks, tombstones, idempotent op merge, P2P sync.

from __future__ import annotations
import argparse
import json
import os
import random
import socket
import string
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

DATA_DIR = Path("Day-14")
STATE_FILE = DATA_DIR / "editor_state.json"
SITE_FILE  = DATA_DIR / "site.json"
EXPORT_TXT = DATA_DIR / "document.txt"

# ---------------- Utilities ----------------

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path: Path, default):
    if not path.exists(): return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- Lamport Clock ----------------

@dataclass
class Lamport:
    time: int
    site: str
    def tick(self) -> int:
        self.time += 1; return self.time
    def observe(self, other: int) -> int:
        self.time = max(self.time, other) + 1; return self.time

# ---------------- Identifiers (LSEQ/Logoot style) ----------------

# An identifier is a list of components: [(digit:int, site:str), ...]
# Compare lexicographically by digit then site.

def cmp_id(a: List[Tuple[int,str]], b: List[Tuple[int,str]]) -> int:
    for (da,sa), (db,sb) in zip(a,b):
        if da != db: return -1 if da < db else 1
        if sa != sb: return -1 if sa < sb else 1
    if len(a) == len(b): return 0
    return -1 if len(a) < len(b) else 1

BASE = 2 ** 15  # digit space per level

def allocate_between(left: List[Tuple[int,str]], right: List[Tuple[int,str]], site: str, depth: int = 0) -> List[Tuple[int,str]]:
    """
    Pick a digit in (L, R) at current depth; if empty, recurse deeper.
    Based on LSEQ/Logoot random allocation to avoid dense growth.
    """
    # Extend with virtual 0/BASE-1 when shorter
    ldig = left[depth][0] if depth < len(left) else 0
    rdig = right[depth][0] if depth < len(right) else BASE - 1
    if rdig - ldig > 1:
        # choose a random digit strictly between
        new_digit = random.randint(ldig + 1, rdig - 1)
        new = (new_digit, site)
        return (left[:depth] + [new])
    else:
        # allocate deeper
        prefix = left[:depth] + [(ldig, left[depth][1] if depth < len(left) else site)]
        return allocate_between(prefix, right, site, depth + 1)

# Sentinels to bound the sequence
HEAD_ID: List[Tuple[int,str]] = [(-1, "_")]
TAIL_ID: List[Tuple[int,str]] = [(BASE, "_")]

# ---------------- CRDT Core ----------------

@dataclass
class Atom:
    id: List[Tuple[int,str]]        # position id
    ch: str                          # single character
    visible: bool                    # tombstone flag (True => present)
    created_ts: int                  # lamport time at creation
    creator: str                     # site id

@dataclass
class Operation:
    op_id: str
    kind: str          # "insert" or "delete"
    lamport: int
    site: str
    payload: Dict[str, Any]  # for insert: {"id": [...], "ch": "X"} ; delete: {"id":[...]}

def empty_state(site: str) -> dict:
    return {
        "meta": {"site": site, "lamport": 0},
        "ops_applied": {},   # op_id -> True
        "atoms": [],         # list of Atom dicts
    }

def get_clock(state: dict) -> Lamport:
    return Lamport(time=state["meta"]["lamport"], site=state["meta"]["site"])

def set_clock(state: dict, clock: Lamport):
    state["meta"]["lamport"] = clock.time

def atoms_sorted(atoms: List[dict]) -> List[dict]:
    return sorted(atoms, key=lambda a: a["id"])

def materialize(atoms: List[dict]) -> str:
    ordered = atoms_sorted(atoms)
    return "".join(a["ch"] for a in ordered if a["visible"])

def find_index_for_id(atoms: List[dict], pid: List[Tuple[int,str]]) -> int:
    # return index where pid would be inserted to keep order
    lo, hi = 0, len(atoms)
    while lo < hi:
        mid = (lo + hi) // 2
        if cmp_id(atoms[mid]["id"], pid) < 0: lo = mid + 1
        else: hi = mid
    return lo

def apply_insert(state: dict, op: Operation):
    pid = op.payload["id"]
    ch  = op.payload["ch"]
    # dedup
    for a in state["atoms"]:
        if a["id"] == pid:
            # if remote insert same id, ensure char/visibility
            a["visible"] = True
            a["ch"] = ch
            return
    idx = find_index_for_id(state["atoms"], pid)
    state["atoms"].insert(idx, asdict(Atom(id=pid, ch=ch, visible=True, created_ts=op.lamport, creator=op.site)))

def apply_delete(state: dict, op: Operation):
    pid = op.payload["id"]
    # find by id, tombstone
    idx = find_index_for_id(state["atoms"], pid)
    # check if exact match at idx or neighbors
    for j in (idx, idx-1, idx+1):
        if 0 <= j < len(state["atoms"]) and state["atoms"][j]["id"] == pid:
            state["atoms"][j]["visible"] = False
            return
    # if we didn't find the atom yet (out-of-order), create a hidden placeholder
    state["atoms"].insert(idx, asdict(Atom(id=pid, ch="", visible=False, created_ts=op.lamport, creator=op.site)))

def apply_operation(state: dict, op: Operation):
    if op.op_id in state["ops_applied"]:
        return
    # advance lamport clock by observe()
    clk = get_clock(state)
    clk.observe(op.lamport)
    if op.kind == "insert":
        apply_insert(state, op)
    elif op.kind == "delete":
        apply_delete(state, op)
    else:
        return
    state["ops_applied"][op.op_id] = True
    set_clock(state, clk)

# ---------------- Local editing API ----------------

def load_state() -> dict:
    ensure_dirs()
    site_info = read_json(SITE_FILE, {})
    if not site_info:
        raise SystemExit("Not initialized. Run: python day14_crdt_editor.py init --site <ID>")
    st = read_json(STATE_FILE, None)
    if st is None:
        st = empty_state(site_info["site"])
        write_json(STATE_FILE, st)
    return st

def save_state(st: dict):
    write_json(STATE_FILE, st)

def init(site: str):
    ensure_dirs()
    if SITE_FILE.exists():
        print("Already initialized.")
        return
    write_json(SITE_FILE, {"site": site})
    write_json(STATE_FILE, empty_state(site))
    print(f"✅ Initialized CRDT editor with site '{site}'")

def build_position_for_insert(state: dict, index: int) -> List[Tuple[int,str]]:
    atoms = atoms_sorted(state["atoms"])
    site  = state["meta"]["site"]
    left  = HEAD_ID if index <= 0 else atoms[index-1]["id"]
    right = TAIL_ID if index >= len(atoms) else atoms[index]["id"]
    return allocate_between(left, right, site)

def op_insert(state: dict, index: int, text: str) -> List[Operation]:
    clk = get_clock(state)
    ops = []
    # insert characters one by one to maintain relative order
    for ch in text:
        clk.tick()
        pid = build_position_for_insert(state, index)
        op = Operation(
            op_id=f"{state['meta']['site']}:{uuid.uuid4()}",
            kind="insert", lamport=clk.time, site=state["meta"]["site"],
            payload={"id": pid, "ch": ch}
        )
        apply_operation(state, op)
        index += 1
        ops.append(op)
    set_clock(state, clk)
    save_state(state)
    return ops

def op_delete(state: dict, index: int, count: int = 1) -> List[Operation]:
    atoms = atoms_sorted(state["atoms"])
    if index < 0 or index >= len(atoms):
        return []
    clk = get_clock(state)
    ops = []
    end = min(index + count, len(atoms))
    for i in range(index, end):
        if not atoms[i]["visible"]:
            continue
        clk.tick()
        pid = atoms[i]["id"]
        op = Operation(
            op_id=f"{state['meta']['site']}:{uuid.uuid4()}",
            kind="delete", lamport=clk.time, site=state["meta"]["site"],
            payload={"id": pid}
        )
        apply_operation(state, op)
        ops.append(op)
    set_clock(state, clk)
    save_state(state)
    return ops

def show(state: dict):
    doc = materialize(state["atoms"])
    print(doc)

def export_txt(state: dict):
    txt = materialize(state["atoms"])
    EXPORT_TXT.write_text(txt, encoding="utf-8")
    print(f"📤 Exported to {EXPORT_TXT}")

# ---------------- Sync (TCP JSON frames) ----------------

def send_json(sock: socket.socket, obj: dict):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))

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

def state_compact(state: dict) -> dict:
    # shrink atoms: tuple ids become lists for JSON; include ops_applied keys only
    return {
        "meta": state["meta"],
        "ops_applied_keys": list(state["ops_applied"].keys()),
        "atoms": state["atoms"],  # already JSON-friendly
    }

def merge_states(local: dict, remote: dict) -> dict:
    # 1) Merge ops_applied sets
    merged = local
    for k in remote["ops_applied"].keys():
        merged["ops_applied"][k] = True
    # 2) Re-apply any atoms from remote that we don't have
    #    We reconstruct operations from remote atoms that aren't in our doc.
    #    (For brevity we treat atoms as authoritative state here.)
    # Build map from id -> atom
    have = {json.dumps(a["id"]): a for a in merged["atoms"]}
    for a in remote["atoms"]:
        key = json.dumps(a["id"])
        if key not in have:
            merged["atoms"].append(a)
        else:
            # ensure visibility/char reflect a "max" of knowledge (idempotent)
            have[key]["ch"] = a["ch"] or have[key]["ch"]
            have[key]["visible"] = have[key]["visible"] and a["visible"]
    # 3) Lamport: take max
    merged["meta"]["lamport"] = max(local["meta"]["lamport"], remote["meta"]["lamport"])
    return merged

def serve(port: int):
    st = load_state()  # ensure init
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    print(f"🌐 CRDT Editor server on 0.0.0.0:{port}")
    try:
        while True:
            conn, addr = s.accept()
            try:
                req = recv_json(conn)
                if not req or req.get("type") != "SYNC":
                    send_json(conn, {"type":"ERROR","msg":"bad handshake"}); conn.close(); continue
                local = read_json(STATE_FILE, empty_state(read_json(SITE_FILE, {})["site"]))
                send_json(conn, {"type":"STATE","payload": local})
                remote = recv_json(conn)
                if not remote or remote.get("type") != "STATE":
                    send_json(conn, {"type":"ERROR","msg":"no state"}); conn.close(); continue
                merged = merge_states(local, remote["payload"])
                write_json(STATE_FILE, merged)
                send_json(conn, {"type":"MERGED","payload": merged})
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nServer exiting.")

def sync(host: str, port: int):
    _ = load_state()
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((host, port))
    send_json(c, {"type":"SYNC"})
    hdr = recv_json(c)
    if not hdr or hdr.get("type") != "STATE":
        raise SystemExit("Protocol error (expected STATE)")
    remote_state = hdr["payload"]
    local_state = read_json(STATE_FILE, empty_state(read_json(SITE_FILE, {})["site"]))
    send_json(c, {"type":"STATE","payload": local_state})
    merged_msg = recv_json(c)
    if not merged_msg or merged_msg.get("type") != "MERGED":
        raise SystemExit("Protocol error (expected MERGED)")
    merged_state = merged_msg["payload"]
    write_json(STATE_FILE, merged_state)
    c.close()
    print("🔁 Sync complete. States converged.")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Day 14 — CRDT Collaborative Editor (LSEQ/Logoot-style)")
    sub = ap.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init")
    p_init.add_argument("--site", required=True, help="Unique site id (e.g., A, phone, laptop)")

    p_ins = sub.add_parser("insert")
    p_ins.add_argument("index", type=int)
    p_ins.add_argument("text")

    p_del = sub.add_parser("delete")
    p_del.add_argument("index", type=int)
    p_del.add_argument("--count", type=int, default=1)

    sub.add_parser("show")
    sub.add_parser("export")

    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--port", type=int, default=9600)

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--host", required=True)
    p_sync.add_argument("--port", type=int, default=9600)

    args = ap.parse_args()
    if args.cmd == "init":
        init(args.site)
    elif args.cmd == "insert":
        st = load_state()
        ops = op_insert(st, args.index, args.text)
        print(f"✅ Inserted {len(ops)} char(s) at {args.index}")
    elif args.cmd == "delete":
        st = load_state()
        ops = op_delete(st, args.index, args.count)
        print(f"✅ Deleted {len(ops)} char(s) starting at {args.index}")
    elif args.cmd == "show":
        st = load_state(); show(st)
    elif args.cmd == "export":
        st = load_state(); export_txt(st)
    elif args.cmd == "serve":
        serve(args.port)
    elif args.cmd == "sync":
        sync(args.host, args.port)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()