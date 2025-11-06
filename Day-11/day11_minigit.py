# Day 11 — MiniGit: Content-Addressable Version Store 🗃️
# Author: Stuart Abhishek
#
# Features:
# - init repository (.minigit/)
# - commit snapshot of working dir
# - log commits
# - checkout (restore) old commit
# - content-addressable storage via SHA-256

import os, sys, hashlib, json, shutil, time
from pathlib import Path

REPO_DIR = ".minigit"
OBJECTS = Path(REPO_DIR) / "objects"
COMMITS = Path(REPO_DIR) / "commits"

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ensure_repo():
    if not Path(REPO_DIR).exists():
        print("⚠️  Not a MiniGit repo (run 'init' first).")
        sys.exit(1)

# ------------------ INIT ------------------

def cmd_init():
    if Path(REPO_DIR).exists():
        print("Repo already initialized.")
        return
    (Path(REPO_DIR)).mkdir()
    OBJECTS.mkdir()
    COMMITS.mkdir()
    with open(Path(REPO_DIR)/"HEAD","w") as f: f.write("")
    print("✅ Initialized empty MiniGit repository in ./"+REPO_DIR)

# ------------------ HASH FILES ------------------

def hash_file(path: Path) -> str:
    with open(path,"rb") as f:
        data=f.read()
    oid=sha256(data)
    obj_path=OBJECTS/oid
    if not obj_path.exists():
        with open(obj_path,"wb") as f: f.write(data)
    return oid

# ------------------ SNAPSHOT + COMMIT ------------------

def snapshot_tree(root=".") -> dict:
    """Return dict: relpath -> object id"""
    mapping={}
    for p in Path(root).rglob("*"):
        if p.is_file() and not str(p).startswith(REPO_DIR):
            mapping[str(p.relative_to(root))]=hash_file(p)
    return mapping

def cmd_commit(message:str):
    ensure_repo()
    tree=snapshot_tree(".")
    commit={
        "tree":tree,
        "message":message,
        "timestamp":time.ctime(),
        "parent":get_head(),
    }
    data=json.dumps(commit,indent=2).encode()
    oid=sha256(data)
    with open(COMMITS/oid,"wb") as f: f.write(data)
    with open(Path(REPO_DIR)/"HEAD","w") as f: f.write(oid)
    print(f"✅ Commit created {oid[:8]} - {message}")

def get_head()->str:
    head_path=Path(REPO_DIR)/"HEAD"
    if not head_path.exists(): return ""
    with open(head_path) as f: return f.read().strip()

# ------------------ LOG ------------------

def cmd_log():
    ensure_repo()
    oid=get_head()
    if not oid:
        print("No commits yet.")
        return
    while oid:
        with open(COMMITS/oid) as f: c=json.load(f)
        print(f"\n🕒 {c['timestamp']}\n🧩 {oid}\n💬 {c['message']}")
        oid=c.get("parent","")
    print("\nEnd of history.")

# ------------------ CHECKOUT ------------------

def cmd_checkout(oid_prefix:str):
    ensure_repo()
    # find full oid
    candidates=[p.name for p in COMMITS.iterdir() if p.name.startswith(oid_prefix)]
    if not candidates:
        print("Commit not found.")
        return
    oid=candidates[0]
    with open(COMMITS/oid) as f: c=json.load(f)
    for rel,oidf in c["tree"].items():
        obj=OBJECTS/oidf
        os.makedirs(Path(rel).parent,exist_ok=True)
        shutil.copy(obj,rel)
    with open(Path(REPO_DIR)/"HEAD","w") as f: f.write(oid)
    print(f"✅ Checked out commit {oid[:8]}")

# ------------------ STATUS ------------------

def cmd_status():
    ensure_repo()
    tree_now=snapshot_tree(".")
    head=get_head()
    if not head:
        print("No commits yet.")
        return
    with open(COMMITS/head) as f: commit=json.load(f)
    tree_prev=commit["tree"]
    added=[f for f in tree_now if f not in tree_prev]
    removed=[f for f in tree_prev if f not in tree_now]
    changed=[f for f in tree_now if f in tree_prev and tree_now[f]!=tree_prev[f]]
    print("📊 Status vs last commit:")
    if not (added or removed or changed):
        print("  No changes.")
    if added: print("  Added:", *added)
    if removed: print("  Removed:", *removed)
    if changed: print("  Modified:", *changed)

# ------------------ CLI DISPATCH ------------------

def main():
    if len(sys.argv)<2:
        print("Usage: python day11_minigit.py [init|commit|log|checkout|status]")
        return
    cmd=sys.argv[1]
    if cmd=="init": cmd_init()
    elif cmd=="commit":
        msg=" ".join(sys.argv[2:]) or "(no message)"
        cmd_commit(msg)
    elif cmd=="log": cmd_log()
    elif cmd=="checkout":
        if len(sys.argv)<3: print("Need commit id."); return
        cmd_checkout(sys.argv[2])
    elif cmd=="status": cmd_status()
    else: print("Unknown command.")

if __name__=="__main__":
    main()