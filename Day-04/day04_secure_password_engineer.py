# Day 4 — Secure Password Engineer 🔐
# Author: Stuart Abhishek
# Description: Cryptographically secure password generator + strength analyzer
# Concepts: secrets, entropy estimation, pattern detection, clean CLI design

import secrets
import string
import math
import re
import time
from datetime import datetime
from pathlib import Path

# --------- Security heuristics & utilities ---------

COMMON_SUBSTRINGS = {
    "password", "qwerty", "iloveyou", "admin", "welcome",
    "letmein", "dragon", "monkey", "football", "baseball",
    "abc123", "123456", "123456789", "111111", "000000",
    "shadow", "sunshine", "superman", "pokemon"
}

AMBIGUOUS = set("O0oIl1|`'\";:.,[]{}()")

def effective_charset_size(pw: str) -> int:
    """Estimate charset size based on categories present in the password."""
    has_lower = any(c.islower() for c in pw)
    has_upper = any(c.isupper() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    symbols = set([c for c in pw if not c.isalnum()])
    size = 0
    if has_lower: size += 26
    if has_upper: size += 26
    if has_digit: size += 10
    # Approximate symbols set size using common safe set
    common_symbol_space = set("!@#$%^&*()-_=+[]{}\\|;:',.<>/?~`")
    if symbols:
        # Count only those likely from the typical symbol alphabet
        size += max(1, len(common_symbol_space))
    return size or 1

def entropy_bits(pw: str) -> float:
    """Return estimated entropy in bits: len * log2(charset)."""
    N = effective_charset_size(pw)
    return round(len(pw) * math.log2(N), 2)

def has_repeated_runs(pw: str) -> bool:
    """Detect runs like aaa, 1111, ==== of length >= 3."""
    return re.search(r"(.)\1\1", pw) is not None

def longest_sequential_run(pw: str) -> int:
    """Detect length of longest ↑/↓ alphanumeric sequence (e.g., 1234, cdef, 9876)."""
    if not pw:
        return 0
    def seq_len(i, step):
        L = 1
        while i + 1 < len(pw) and ord(pw[i+1]) - ord(pw[i]) == step:
            L += 1
            i += 1
        return L
    best = 1
    for i in range(len(pw) - 1):
        best = max(best, seq_len(i, 1), seq_len(i, -1))
    return best

def contains_common_substring(pw: str) -> bool:
    low = pw.lower()
    return any(s in low for s in COMMON_SUBSTRINGS)

def ambiguous_fraction(pw: str) -> float:
    if not pw: return 0.0
    return sum(1 for c in pw if c in AMBIGUOUS) / len(pw)

def grade_from_score(score: int) -> str:
    if score >= 85: return "Very Strong"
    if score >= 70: return "Strong"
    if score >= 55: return "Moderate"
    if score >= 40: return "Weak"
    return "Very Weak"

def analyze_password(pw: str) -> dict:
    """Return a rich analysis with score, issues, and suggestions."""
    issues, suggestions = [], []

    ent = entropy_bits(pw)
    seq_run = longest_sequential_run(pw)
    repeats = has_repeated_runs(pw)
    common = contains_common_substring(pw)
    ambi = ambiguous_fraction(pw)

    # Base score from entropy (cap at 90 to leave room for penalties/bonuses)
    base = min(90, int(ent))

    # Diversity bonus
    diversity = sum([
        any(c.islower() for c in pw),
        any(c.isupper() for c in pw),
        any(c.isdigit() for c in pw),
        any(not c.isalnum() for c in pw),
    ])
    base += (diversity - 2) * 5  # reward 3–4 categories

    # Penalties
    if len(pw) < 12:
        base -= 20
        issues.append("Too short (< 12 characters).")
        suggestions.append("Use at least 12–16 characters.")
    if common:
        base -= 25
        issues.append("Contains common or easily guessed pattern.")
        suggestions.append("Avoid words like 'password', 'qwerty', '12345', etc.")
    if repeats:
        base -= 10
        issues.append("Has repeated characters/runs (e.g., aaa, 1111).")
        suggestions.append("Break up repeated characters.")
    if seq_run >= 3:
        base -= 10
        issues.append(f"Has sequential run length {seq_run} (e.g., 1234, abcd).")
        suggestions.append("Avoid ascending/descending sequences.")
    if ambi > 0.25:
        issues.append("Uses many ambiguous characters (O/0, l/1, I).")
        suggestions.append("Reduce ambiguous look-alike characters for usability.")

    score = max(0, min(100, base))
    grade = grade_from_score(score)

    # Positive feedback
    if len(pw) >= 16:
        suggestions.append("Great length — keep 16+ for stronger security.")
    if diversity >= 3:
        suggestions.append("Good character diversity — aim for 3–4 categories.")

    return {
        "password_length": len(pw),
        "entropy_bits": ent,
        "score": score,
        "grade": grade,
        "longest_sequential_run": seq_run,
        "has_repeats": repeats,
        "contains_common_pattern": common,
        "ambiguous_ratio": round(ambi, 2),
        "issues": issues,
        "suggestions": sorted(set(suggestions)),
    }

# --------- Generator ---------

def build_alphabet(use_lower=True, use_upper=True, use_digits=True, use_symbols=True, avoid_ambiguous=True) -> str:
    alph = ""
    if use_lower: alph += string.ascii_lowercase
    if use_upper: alph += string.ascii_uppercase
    if use_digits: alph += string.digits
    if use_symbols: alph += "!@#$%^&*()-_=+[]{}\\|;:',.<>/?~`"
    if avoid_ambiguous:
        alph = "".join(ch for ch in alph if ch not in AMBIGUOUS)
    return alph

def generate_password(length=16, use_lower=True, use_upper=True, use_digits=True, use_symbols=True, avoid_ambiguous=True) -> str:
    if length < 4:
        raise ValueError("Length must be at least 4.")
    pools = []
    if use_lower: pools.append(string.ascii_lowercase)
    if use_upper: pools.append(string.ascii_uppercase)
    if use_digits: pools.append(string.digits)
    if use_symbols: pools.append("!@#$%^&*()-_=+[]{}\\|;:',.<>/?~`")

    # Remove ambiguous if requested
    pools = ["".join(ch for ch in p if ch not in AMBIGUOUS) if avoid_ambiguous else p for p in pools]
    alphabet = "".join(pools)
    if not alphabet:
        raise ValueError("Alphabet is empty. Enable at least one category.")

    # Ensure at least one from each selected pool
    rng = secrets.SystemRandom()
    password_chars = []
    for p in pools:
        if p:  # pool active
            password_chars.append(rng.choice(p))

    # Fill remaining
    while len(password_chars) < length:
        password_chars.append(rng.choice(alphabet))

    # Shuffle to avoid predictable positions
    rng.shuffle(password_chars)
    return "".join(password_chars)

# --------- CLI ---------

LOG_PATH = Path("Day-04/generated_passwords.log")

def log_password(pw: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {pw}\n")

def print_analysis(report: dict):
    print("\n" + "-" * 60)
    print(f"Score: {report['score']}/100  |  Grade: {report['grade']}")
    print(f"Length: {report['password_length']}  |  Entropy: {report['entropy_bits']} bits")
    print(f"Longest sequential run: {report['longest_sequential_run']}")
    print(f"Repeated runs: {report['has_repeats']}  |  Common pattern: {report['contains_common_pattern']}")
    print(f"Ambiguous chars ratio: {report['ambiguous_ratio']}")
    if report["issues"]:
        print("\nIssues:")
        for i in report["issues"]:
            print(f"  • {i}")
    print("\nSuggestions:")
    for s in report["suggestions"]:
        print(f"  • {s}")
    print("-" * 60 + "\n")

def menu():
    print("🔐 Secure Password Engineer — Day 4")
    print("1) Generate a strong password")
    print("2) Analyze an existing password")
    print("3) Generate & analyze (recommended)")
    print("q) Quit")
    return input("Choose an option: ").strip().lower()

def choose_bool(prompt: str, default=True):
    val = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not val: return default
    return val in ("y", "yes")

def cli():
    while True:
        choice = menu()
        if choice == "q":
            print("Goodbye! Stay safe. 🔐")
            break

        if choice in ("1", "3"):
            try:
                length = int(input("Desired length (recommend 16–24): ").strip() or "16")
            except ValueError:
                print("Invalid length. Using 16.")
                length = 16
            use_lower = choose_bool("Include lowercase?", True)
            use_upper = choose_bool("Include uppercase?", True)
            use_digits = choose_bool("Include digits?", True)
            use_symbols = choose_bool("Include symbols?", True)
            avoid_amb = choose_bool("Avoid ambiguous characters (O/0, l/1)?", True)

            try:
                pw = generate_password(length, use_lower, use_upper, use_digits, use_symbols, avoid_amb)
                print(f"\nGenerated password:\n{pw}\n")
                if choose_bool("Save to log file?", False):
                    log_password(pw)
                    print(f"Saved to {LOG_PATH}\n")
            except ValueError as e:
                print(f"Error: {e}")
                continue

            if choice == "1":
                # Offer quick analysis as a bonus
                if choose_bool("Analyze this password now?", True):
                    report = analyze_password(pw)
                    print_analysis(report)
                continue

            # choice == "3": analyze immediately
            report = analyze_password(pw)
            print_analysis(report)

        elif choice == "2":
            pw = input("Enter the password to analyze: ")
            report = analyze_password(pw)
            print_analysis(report)
        else:
            print("Invalid option. Try again.\n")

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        print("\nInterrupted. Stay secure! 🔐")