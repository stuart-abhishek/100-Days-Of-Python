 ---

🐍 100 Days of Python — by Stuart Abhishek

> “Small consistent steps create giant success.” — Stuart Abhishek



Welcome to my 100 Days of Python challenge — a personal mission to master Python, develop real-world problem-solving skills, and build a portfolio that proves passion, discipline, and creativity.

Each day I design, code, and upload a new project — from beginner fundamentals to AI-inspired applications — preparing myself for my dream:
🎯 IIT Madras → MIT / Stanford Computer Science.


---

🧩 Progress Tracker

Day	Project	Description	Status

1 	Hello World	My first Python program — starting my journey	✅
2	 AI Quote Generator	Interactive personalized motivational quote app	✅
3 	Smart Math Quiz	Adaptive arithmetic quiz with scoring system	✅
4 Secure Password Engineer (generator + analyzer) ✅



---

📘 Project Details


---

🧠 Day 1 — Hello World 👋

🔹 Project Title

Hello World Program — my first ever Python code.

🔹 Description

The simplest beginning: printing a message to prove everything works!
This moment marks the first step of my lifelong journey into programming.

🔹 Code

# Day 1 – Hello World Program
# Author: Stuart Abhishek

print("Hello, World! This is Day 1 of my 100 Days of Python challenge.")

🔹 Example Output

Hello, World! This is Day 1 of my 100 Days of Python challenge.

🔹 What I Learned

Running my first Python script

Understanding print()

Importance of syntax & indentation

Confidence boost — the journey begins 🚀



---

🧠 Day 2 — AI-Style Quote Generator 🧠

🔹 Project Title

AI Quote Generator — a personalized console program that delivers motivational quotes.

🔹 Description

Combines creativity and logic: asks for your name & goal, then builds a personalized, time-aware quote using Python’s random and datetime modules.
It’s my first project that feels alive — like a tiny AI friend.

🔹 Code

# Day 2 – AI-Style Quote Generator
# Author: Stuart Abhishek

import random, datetime

print("🤖 Welcome to the AI Quote Generator!")
print("Let's create a personalized quote to inspire you today.\n")

name = input("What is your name? ")
goal = input("What’s one goal you’re working on right now? ")

quotes = [
  f"{name}, remember — every expert was once a beginner. Keep pushing toward {goal}!",
  f"Success doesn’t come from what you do occasionally, {name}, it comes from what you do consistently for {goal}.",
  f"{name}, when you feel like quitting, think about why you started {goal}.",
  f"The future belongs to those like {name} who never stop learning while chasing {goal}.",
  f"{name}, small steps every day towards {goal} will lead to massive results."
]

quote = random.choice(quotes)
hour = datetime.datetime.now().hour
greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

print("\n" + "=" * 60)
print(f"{greeting}, {name}! 🌟")
print("Here’s your motivational message:")
print(f"💬  {quote}")
print("=" * 60)
print("~ Program created by Stuart Abhishek (Day 2 of 100 Days of Python) ~")

🔹 Example Output

🤖 Welcome to the AI Quote Generator!
Let's create a personalized quote to inspire you today.

What is your name? Stuart
What’s one goal you’re working on right now? Python mastery

============================================================
Good evening, Stuart! 🌟
Here’s your motivational message:
💬  Stuart, when you feel like quitting, think about why you started Python mastery.
============================================================
~ Program created by Stuart Abhishek (Day 2 of 100 Days of Python) ~

🔹 Concepts Used

Variables & Input

Lists + random.choice()

datetime module

Conditional Statements

Formatted Strings (f-strings)

Readable Code Structure


🔹 What I Learned

Making interactive programs

Mixing logic with emotion through code

Documentation and clean layout

My first program that feels intelligent 🤖



---

🧠 Day 3 — Smart Math Quiz with Scoring System 🎯

🔹 Project Title

Smart Math Quiz — an adaptive Python quiz that tests your math skills and rewards progress.

🔹 Description

An interactive math quiz that auto-generates arithmetic questions, adjusts difficulty based on score, and shows a final performance report.
Demonstrates functions, loops, conditionals, randomization, and real-time scoring.

🔹 Code

# Day 3 – Smart Math Quiz with Scoring System
# Author: Stuart Abhishek

import random, time

def generate_question(level):
  if level == 1:
    a, b = random.randint(1, 10), random.randint(1, 10)
    op = random.choice(['+', '-'])
  elif level == 2:
    a, b = random.randint(10, 50), random.randint(1, 20)
    op = random.choice(['+', '-', '*'])
  else:
    a, b = random.randint(20, 100), random.randint(1, 25)
    op = random.choice(['+', '-', '*', '//'])
  question = f"{a} {op} {b}"
  return question, eval(question)

def math_quiz():
  print("🧮 Welcome to the Smart Math Quiz!")
  print("Answer as many questions as you can. Type 'quit' to stop.\n")
  level = 1; score = 0; count = 0; start = time.time()
  while True:
    count += 1
    q, ans = generate_question(level)
    user = input(f"Q{count}: {q} = ")
    if user.lower() == "quit": break
    try:
      if int(user) == ans:
        score += 10
        print("✅ Correct!")
        if score % 50 == 0:
          level = min(level + 1, 3)
          print("🚀 Level Up! Difficulty increased.")
      else:
        score -= 5
        print(f"❌ Wrong! Correct answer was {ans}.")
    except ValueError:
      print("⚠️ Enter a number or 'quit'.")
    print(f"Current Score: {score}\n")

  t = round(time.time() - start, 2)
  print("=" * 55)
  print("🏁 Quiz Summary")
  print(f"Questions: {count - 1} | Final Score: {score} | Time: {t}s")
  if score >= 100: print("🌟 Brilliant work!")
  elif score >= 50: print("💪 Great job!")
  else: print("📘 Keep practicing!")
  print("=" * 55)
  print("~ Program created by Stuart Abhishek (Day 3 of 100 Days of Python) ~")

if __name__ == "__main__":
  math_quiz()

🔹 Example Output

🧮 Welcome to the Smart Math Quiz!
Answer as many questions as you can. Type 'quit' to stop.

Q1: 3 + 4 = 7
✅ Correct!
Current Score: 10

Q2: 12 – 8 = 4
✅ Correct!
Current Score: 20

Q3: 6 * 5 = 31
❌ Wrong! Correct answer was 30.
Current Score: 15

🚀 Level Up! Difficulty increased.

🏁 Quiz Summary
Questions: 10  | Final Score: 85  | Time: 48.7 s
💪 Great job! Keep sharpening your mind.

🔹 Concepts Used

Functions and Modular Design

Loops and Conditionals

Random Number Generation

Scoring & Difficulty Progression

Time Measurement (time module)


🔹 What I Learned

Designing adaptive logic

Writing clean, structured functions

Handling user input gracefully

Thinking algorithmically like an engineer


🔹 Future Improvements

Add leaderboard / save scores to file

Introduce GUI using tkinter

Add division and power levels for advanced math


---

## 🧠 Day 4 — Secure Password Engineer 🔐

### 🔹 Project Title
**Secure Password Engineer** — Cryptographically secure password generator + strength analyzer.

### 🔹 Project Description
Generates strong passwords with customizable character sets using Python’s `secrets` (CSPRNG),
and analyzes any password for entropy (bits), common patterns, repeated/sequential runs, and ambiguous characters.
Produces a 0–100 score, a clear grade, and practical suggestions. Optional local logging.

### 🔹 Concepts Used
- Cryptographic randomness with `secrets`
- Entropy estimation & effective charset analysis
- Pattern detection (common words, repeats, sequences)
- Clean CLI design, modular functions, robust I/O
- Filesystem logging with `pathlib`

### 🔹 Example (Generate & Analyze)

🔐 Secure Password Engineer — Day 4

1. Generate a strong password


2. Analyze an existing password


3. Generate & analyze (recommended) q) Quit Choose an option: 3 Desired length (recommend 16–24): 18 Include lowercase? [Y/n]: Include uppercase? [Y/n]: Include digits? [Y/n]: Include symbols? [Y/n]: Avoid ambiguous characters (O/0, l/1)? [Y/n]:



Generated password: 7uG}xVbR%pZt_fH3q*

Score: 93/100  |  Grade: Very Strong Length: 18  |  Entropy: 113.61 bits Longest sequential run: 1 Repeated runs: False  |  Common pattern: False Ambiguous chars ratio: 0.0

Suggestions: • Good character diversity — aim for 3–4 categories. • Great length — keep 16+ for stronger security.

### 🔹 What I Learned
- Why `secrets` > `random` for security
- How to estimate entropy and explain strength in bits
- How to detect human patterns attackers exploit
- How to give users actionable security guidance

### 🔹 Future Improvements
- Add passphrase mode (diceware-style)
- Add exclusion list (user-provided words)
- Export analysis report as JSON/CSV


---

🌟 Journey Continues...

Each day is one step closer to my dream — to think, code, and create like a world-class computer scientist.
Stay tuned for:

Day 5: 🧮 Mini Calculator App

Day 6+: 🤖 AI and Data Science Projects



---

✍️ Author

Stuart Abhishek
15-year-old developer on a mission to reach IIT Madras → MIT / Stanford CSE.

> “Code with purpose. Learn with passion.”




---


  


