# Day 12 — Distributed MiniGit 🌐
# Author: Stuart Abhishek
#
# Adds a minimal TCP protocol on top of Day 11 MiniGit:
# - serve: start a thread-per-connection TCP server
# - push  : send local commits/objects that the remote lacks
# - pull  : fetch remote commits/objects that we lack
#
# Protocol: newline-delimited JSON "frames" with {"type": "...", ...}
# Content is addressed by SHA-256; receiver verifies integrity.
#
# Usage examples:
#   python day12_distributed_minigit.py serve --port 9999
#   python day12_distributed_minigit.py push  --host 127.0.0.1 --port 9999
#   python day12_distributed_minigit.py pull  --host 127.0.0.1 --port 9999
#
# Requires Day-11 repo layout (.minigit/objects, .minigit/commits, HEAD).

import argparse, json, os, socket, threading, hashlib, sys
from pathlib import Path

REPO_DIR = ".minigit"
OBJ_DIR  = Path(REPO_DIR) / "objects"
CMT_DIR  = Path(REPO_DIR) / "commits"
HEAD_F   = Path(REPO_DIR) / "HEAD"

# -------------------- Storage primitives (Day-11 compatible) --------------------

def ensure_repo():
    if not Path(REPO_DIR).exists():
        raise SystemExit("❌ Not a MiniGit repo here. Run Day-11 init & commits first.")

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def list_objects():
    if not OBJ_DIR.exists(): return []
    return [p.name for p in OBJ_DIR.iterdir() if p.is_file()]

def list_commits():
    if not CMT_DIR.exists(): return []
    return [p.name for p in CMT_DIR.iterdir() if p.is_file()]

def read_commit(oid: str) -> bytes:
    return Path(CMT_DIR / oid).read_bytes()

def read_object(oid: str) -> bytes:
    return Path(OBJ_DIR / oid).read_bytes()

def write_blob(dirpath: Path, oid: str, data: bytes):
    # verify content-address
    calc = sha256(data)
    if calc != oid:
        raise ValueError("Hash mismatch while writing blob")
    path = dirpath / oid
    if not path.exists():
        path.write_bytes(data)

def get_head() -> str:
    return HEAD_F.read_text().strip() if HEAD_F.exists() else ""

def set_head(oid: str):
    HEAD_F.write_text(oid)

# -------------------- Tiny framed JSON protocol --------------------

def send_frame(sock: socket.socket, payload: dict):
    data = (json.dumps(payload) + "\n").encode("utf-8")
    sock.sendall(data)

def recv_frame(sock: socket.socket) -> dict | None:
    # Read until newline
    buf = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            return None
        if chunk == b"\n":
            break
        buf += chunk
    return json.loads(buf.decode("utf-8"))

# -------------------- Server side handlers --------------------

def handle_client(conn: socket.socket, addr):
    try:
        # Greet and announce inventory
        inv = {
            "type": "HELLO",
            "server": "DistributedMiniGit/1.0",
            "head": get_head(),
            "commits": list_commits(),
            "objects": list_objects(),
        }
        send_frame(conn, inv)

        while True:
            req = recv_frame(conn)
            if req is None:  # client closed
                return
            t = req.get("type")

            if t == "LIST":
                send_frame(conn, {
                    "type": "INV",
                    "head": get_head(),
                    "commits": list_commits(),
                    "objects": list_objects(),
                })

            elif t == "GET_COMMIT":
                oid = req["oid"]
                try:
                    data = read_commit(oid)
                    send_frame(conn, {"type":"COMMIT_DATA","oid":oid,"size":len(data)})
                    conn.sendall(data + b"\n")  # raw blob + newline delimiter
                except FileNotFoundError:
                    send_frame(conn, {"type":"ERROR","msg":"commit not found"})

            elif t == "GET_OBJECT":
                oid = req["oid"]
                try:
                    data = read_object(oid)
                    send_frame(conn, {"type":"OBJECT_DATA","oid":oid,"size":len(data)})
                    conn.sendall(data + b"\n")
                except FileNotFoundError:
                    send_frame(conn, {"type":"ERROR","msg":"object not found"})

            elif t == "PUT_COMMIT":
                oid = req["oid"]; size = int(req["size"])
                blob = recv_raw_blob(conn, size)
                # integrity check & store
                write_blob(CMT_DIR, oid, blob)
                send_frame(conn, {"type":"STORED","what":"commit","oid":oid})

            elif t == "PUT_OBJECT":
                oid = req["oid"]; size = int(req["size"])
                blob = recv_raw_blob(conn, size)
                write_blob(OBJ_DIR, oid, blob)
                send_frame(conn, {"type":"STORED","what":"object","oid":oid})

            elif t == "SET_HEAD":
                oid = req.get("oid","")
                if oid and (CMT_DIR / oid).exists():
                    set_head(oid)
                    send_frame(conn, {"type":"OK"})
                else:
                    send_frame(conn, {"type":"ERROR","msg":"unknown commit for HEAD"})

            elif t == "BYE":
                send_frame(conn, {"type":"OK"})
                return

            else:
                send_frame(conn, {"type":"ERROR","msg":"unknown request"})
    except Exception as e:
        try: send_frame(conn, {"type":"ERROR","msg":str(e)})
        except: pass
    finally:
        conn.close()

def recv_raw_blob(conn: socket.socket, size: int) -> bytes:
    buf = bytearray()
    remaining = size
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("unexpected EOF")
        buf += chunk
        remaining -= len(chunk)
    # trailing newline from sender
    trailing = conn.recv(1)
    if trailing != b"\n":
        # tolerate: push back not available; ignore
        pass
    return bytes(buf)

def serve(port: int):
    ensure_repo()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(5)
    print(f"🌐 MiniGit server listening on 0.0.0.0:{port}")
    try:
        while True:
            conn, addr = sock.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nServer shutting down.")

# -------------------- Client ops: push/pull --------------------

def dial(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    # read HELLO
    hello = recv_frame(s)
    if not hello or hello.get("type") != "HELLO":
        raise SystemExit("Bad handshake")
    return s

def push(host: str, port: int):
    """Send our missing commits/objects to remote and advance remote HEAD to our HEAD."""
    ensure_repo()
    s = dial(host, port)
    try:
        # Ask remote inventory
        send_frame(s, {"type":"LIST"})
        inv = recv_frame(s)
        remote_commits = set(inv.get("commits", []))
        remote_objs    = set(inv.get("objects", []))

        # Diff our inventory vs remote
        our_commits = set(list_commits())
        our_objs    = set(list_objects())

        missing_commits = sorted(our_commits - remote_commits)
        missing_objects = sorted(our_objs    - remote_objs)

        print(f"➡️  Pushing commits: {len(missing_commits)}, objects: {len(missing_objects)}")

        # Send commits
        for oid in missing_commits:
            blob = read_commit(oid)
            send_frame(s, {"type":"PUT_COMMIT","oid":oid,"size":len(blob)})
            s.sendall(blob + b"\n")
            _ = recv_frame(s)

        # Send objects
        for oid in missing_objects:
            blob = read_object(oid)
            send_frame(s, {"type":"PUT_OBJECT","oid":oid,"size":len(blob)})
            s.sendall(blob + b"\n")
            _ = recv_frame(s)

        # Update remote HEAD to our HEAD (fast-forward style)
        head = get_head()
        if head:
            send_frame(s, {"type":"SET_HEAD","oid":head})
            resp = recv_frame(s)
            if resp.get("type") != "OK":
                print("⚠️ Remote HEAD not updated:", resp)
        send_frame(s, {"type":"BYE"})
        _ = recv_frame(s)
        print("✅ Push complete.")
    finally:
        s.close()

def pull(host: str, port: int):
    """Fetch remote commits/objects we don't have; fast-forward our HEAD."""
    ensure_repo()
    s = dial(host, port)
    try:
        # Remote inventory
        send_frame(s, {"type":"LIST"})
        inv = recv_frame(s)
        remote_head    = inv.get("head","")
        remote_commits = set(inv.get("commits", []))
        remote_objs    = set(inv.get("objects", []))

        our_commits = set(list_commits())
        our_objs    = set(list_objects())

        need_commits = sorted(remote_commits - our_commits)
        need_objects = sorted(remote_objs    - our_objs)

        print(f"⬇️  Pulling commits: {len(need_commits)}, objects: {len(need_objects)}")

        for oid in need_commits:
            send_frame(s, {"type":"GET_COMMIT","oid":oid})
            hdr = recv_frame(s)
            if hdr.get("type") != "COMMIT_DATA": raise SystemExit("protocol error")
            blob = recv_raw_blob(s, int(hdr["size"]))
            # integrity check
            if sha256(blob) != oid: raise SystemExit("commit hash mismatch")
            write_blob(CMT_DIR, oid, blob)

        for oid in need_objects:
            send_frame(s, {"type":"GET_OBJECT","oid":oid})
            hdr = recv_frame(s)
            if hdr.get("type") != "OBJECT_DATA": raise SystemExit("protocol error")
            blob = recv_raw_blob(s, int(hdr["size"]))
            if sha256(blob) != oid: raise SystemExit("object hash mismatch")
            write_blob(OBJ_DIR, oid, blob)

        # Fast-forward our HEAD if the remote advertised a known commit
        if remote_head and (CMT_DIR / remote_head).exists():
            set_head(remote_head)
            print(f"🔁 Fast-forwarded HEAD to {remote_head[:8]}")

        send_frame(s, {"type":"BYE"})
        _ = recv_frame(s)
        print("✅ Pull complete.")
    finally:
        s.close()

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Distributed MiniGit (Day 12)")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("serve")
    a.add_argument("--port", type=int, default=9999)

    b = sub.add_parser("push")
    b.add_argument("--host", required=True)
    b.add_argument("--port", type=int, default=9999)

    c = sub.add_parser("pull")
    c.add_argument("--host", required=True)
    c.add_argument("--port", type=int, default=9999)

    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.port)
    elif args.cmd == "push":
        push(args.host, args.port)
    elif args.cmd == "pull":
        pull(args.host, args.port)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()