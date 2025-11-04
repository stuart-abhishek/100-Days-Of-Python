---

🐍 100 Days of Python — by Stuart Abhishek

> “Small consistent steps create giant success.” — Stuart Abhishek




---

🌎 About This Repository

Welcome to my 100 Days of Python challenge!

This repository documents my 100-day journey to master Python — one project every day, combining creativity, logic, and real-world problem-solving.
Each project is designed with clean structure, interactive design, and modern programming concepts that reflect the skills of a future Computer Science Engineer at → MIT → Stanford.


---

🎯 My Vision

🔥 Master Python — from fundamentals to advanced algorithms.

💡 Think like an engineer, design like an artist, and build like a scientist.

🚀 Develop projects that prove consistency, logic, and innovation.

🎓 Achieve admission into top universities — MIT, Stanford — through skills and passion.



---

🧩 Progress Tracker

Day	Project	Description	Status

1 	Hello World	My first Python program — start of my journey	✅
2	 AI Quote Generator	Personalized motivational quote generator	✅
3	 Smart Math Quiz	Adaptive arithmetic quiz with scoring & logic	✅
4 	Secure Password Engineer	Cryptographically secure password generator + analyzer	✅
5 Natural language smart calculator ✅
6 Smart Data Analyzer ✅
7 Predictive Insight Engine ✅
8 From Scratch Naive Bayes Text Classifier ✅
9 Clustering Insight Engine ✅


---

📘 Project Details


---

🧠 Day 1 — Hello World 👋

🔹 Project Title

Hello World Program — My first step into the world of programming.

🔹 Description

A simple Python script that prints a motivational message.
This marks the beginning of my 100-day journey — the foundation of everything that follows.

🔹 Code

# Day 1 – Hello World Program
# Author: Stuart Abhishek

print("Hello, World! This is Day 1 of my 100 Days of Python challenge.")

🔹 Example Output

Hello, World! This is Day 1 of my 100 Days of Python challenge.

🔹 What I Learned

Basic Python syntax

Printing output to the console

My first taste of programming discipline 💪



---

🧠 Day 2 — AI-Style Quote Generator 🤖

🔹 Project Title

AI Quote Generator — A personalized AI-like motivational quote system.

🔹 Description

This Python script interacts with the user to create motivational quotes based on their name and goal.
It uses randomness and time-based logic to create human-like responses — blending creativity with computation.

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

print("\n" + "="*60)
print(f"{greeting}, {name}! 🌟")
print("Here’s your motivational message:")
print(f"💬  {quote}")
print("="*60)
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

Randomization

datetime module

String formatting

Conditional statements


🔹 What I Learned

How to make interactive programs

Personalization through logic

Creating “human-feeling” code 🤖



---

🧠 Day 3 — Smart Math Quiz 🎯

🔹 Project Title

Smart Math Quiz — Adaptive arithmetic quiz with scoring & difficulty levels.

🔹 Description

This quiz challenges users with math problems that automatically increase in difficulty as you score higher.
It rewards accuracy, penalizes errors, and gives a professional performance summary — simulating a smart learning system.

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
  print("="*55)
  print("🏁 Quiz Summary")
  print(f"Questions: {count - 1} | Final Score: {score} | Time: {t}s")
  if score >= 100: print("🌟 Brilliant work!")
  elif score >= 50: print("💪 Great job!")
  else: print("📘 Keep practicing!")
  print("="*55)
  print("~ Program created by Stuart Abhishek (Day 3 of 100 Days of Python) ~")

if __name__ == "__main__":
  math_quiz()

🔹 Example Output

Q1: 3 + 4 = 7
✅ Correct!
Current Score: 10

Q2: 12 - 8 = 4
✅ Correct!
Current Score: 20

Q3: 6 * 5 = 31
❌ Wrong! Correct answer was 30.
Current Score: 15

🔹 Concepts Used

Functions & Modular Design

Loops & Conditionals

Randomization & Adaptive Logic

Scoring Systems

Time Measurement


🔹 What I Learned

Structured function design

Adaptive algorithms

Logic building like a real engineer ⚙️



---

🧠 Day 4 — Secure Password Engineer 🔐

🔹 Project Title

Secure Password Engineer — Cryptographically secure password generator + strength analyzer.

🔹 Description

This professional-grade program creates uncrackable passwords and analyzes their strength using:

Entropy calculations

Pattern detection

Sequential run identification

Ambiguity checks

Comprehensive 0–100 scoring system


It uses the secrets module (cryptographically secure RNG) and outputs suggestions for improvement — just like a mini cybersecurity assistant.

🔹 Code Highlights

secrets module for secure randomness

math.log2() for entropy estimation

Regular expressions for pattern detection

Logging system using pathlib

Modular functions and clean CLI


🔹 Example Interaction

🔐 Secure Password Engineer — Day 4
1) Generate a strong password
2) Analyze an existing password
3) Generate & analyze (recommended)
q) Quit
Choose an option: 3
Desired length (recommend 16–24): 18
Include lowercase? [Y/n]:
Include uppercase? [Y/n]:
Include digits? [Y/n]:
Include symbols? [Y/n]:
Avoid ambiguous characters? [Y/n]:

Generated password:
7uG}xVbR%pZt_fH3q*

Score: 93/100 | Grade: Very Strong
Length: 18 | Entropy: 113.61 bits
Longest sequential run: 1
Repeated runs: False | Common pattern: False

Suggestions:
• Great length — keep 16+ for stronger security.
• Excellent character diversity — 4 categories used.

🔹 Concepts Used

Cryptography & Secure Randomness

Entropy and Information Theory

Pattern Recognition

Data Validation

File Logging and Modular Programming


🔹 What I Learned

Difference between random and secrets

How to measure password entropy mathematically

Writing security-conscious, user-friendly code

Designing an AI-style analytical program


---

## 🧠 Day 5 — Natural-Language Smart Calculator 🤖

### 🔹 Project Title
**Smart Calculator** — Understands human language to perform arithmetic.

### 🔹 Project Description
A Python program that interprets natural-language expressions like  
“add 12 and 45”, “subtract 10 from 50”, “square root of 81”,  
and computes accurate results.  
It mimics early natural-language interfaces — showing algorithmic reasoning and text processing skills.

### 🔹 Concepts Used
- Regular Expressions (`re`) for text extraction  
- Conditional Logic for intent detection  
- Mathematical operations (`math` module)  
- Exception handling  
- Clean modular programming  

### 🔹 Example Output

🧮 Welcome to the Natural-Language Smart Calculator! 👉 Enter expression: add 24 and 65 ✅ Result: 89.0

👉 Enter expression: subtract 10 from 42 ✅ Result: 32.0

👉 Enter expression: square root of 81 ✅ Result: 9.0

👉 Enter expression: cube of 3 ✅ Result: 27.0

### 🔹 What I Learned
- Translating language into computation  
- Regex pattern matching and token parsing  
- Handling ambiguous input gracefully  
- Thinking algorithmically like a language-model designer  

### 🔹 Future Improvements
- Integrate `nltk` or `spaCy` for deeper natural-language parsing  
- Add unit conversion and scientific-mode operations

---

## 🧠 Day 6 — Smart Data Analyzer 📊

### 🔹 Project Title
**Smart Data Analyzer** — automatic statistical and correlation analysis of CSV datasets.

### 🔹 Project Description
This Python engine loads any CSV file and instantly produces summary statistics, detects strong correlations, and provides simple “insights.”  
It demonstrates data-science fundamentals, algorithmic thinking, and data-driven storytelling.

### 🔹 Concepts Used
- File I/O & CSV parsing (`csv.DictReader`)
- Statistics & probability (`statistics`, `math`)
- Correlation coefficient computation
- Data visualization with `matplotlib`
- Algorithmic automation & reporting

### 🔹 Example Output

📊 Smart Data Analyzer — Day 6 Enter CSV file path (e.g., data.csv): students.csv

📈 Summary Statistics • Math: mean=78.4, median=80.0, stdev=10.2, n=50 • Science: mean=76.1, median=75.0, stdev=9.5, n=50 • English: mean=81.6, median=82.0, stdev=8.9, n=50

🤝 Significant Correlations (|r| ≥ 0.5) Math ↔ Science: r = 0.91 (direct correlation) English ↔ Math: r = 0.73 (direct correlation)

✨ Insights: Strongest link: Math and Science (0.91). Consider exploring cause-effect relationship. Report complete ✅

### 🔹 What I Learned
- Reading structured data programmatically  
- Statistical reasoning (mean, median, stdev, correlation)  
- Automating analysis workflows like real data scientists  
- Presenting information visually and narratively  

### 🔹 Future Improvements
- Integrate `pandas` for larger datasets  
- Export summary reports as PDF  
- Apply linear regression to predict relationships  
- Build a web dashboard using `Streamlit`

---
  
## 🧠 Day 7 — Predictive Insight Engine 📈

### 🔹 Project Title
**Predictive Insight Engine** — Univariate Linear Regression with Cross-Validation, Outlier Handling, Plots, and a Model Card.

### 🔹 Project Description
A disciplined mini-ML pipeline:
- Loads a CSV, selects a numeric feature (X) and target (Y)
- Optional z-score outlier removal
- Standardizes features/targets
- Trains linear regression via gradient descent with early stopping
- Reports **R², MAE, RMSE** and performs **5-fold cross-validation**
- Shows **fitted line** and **residual diagnostics** plots
- Exports a **JSON model card** (coefficients, scalers, CV metrics, metadata)

### 🔹 Concepts Used
- Data hygiene (z-scores), standardization
- Gradient descent & early stopping
- Generalization via cross-validation
- Multiple evaluation metrics (R², MAE, RMSE)
- Residual analysis and visualization
- Reproducibility (JSON model card)

### 🔹 Example Session

📈 Predictive Insight Engine — Day 7 Enter CSV path (e.g., data.csv): students.csv Choose FEATURE (X):

1. StudyHours


2. SleepHours


3. MathScore Select number: 1 Choose TARGET (Y):


4. MathScore


5. ScienceScore Select number: 1 Remove outliers with z-score > 3? [Y/n]: Y Outlier filter: 52 → 50 usable pairs.



🔍 5-fold Cross-Validation R2_mean: 0.8123 R2_std: 0.0431 MAE_mean: 3.215 RMSE_mean: 4.097

✅ Fitted on full data R²: 0.8467 | MAE: 2.98 | RMSE: 3.82

Show fitted-line plot? [Y/n]: Y Show residuals plot? [Y/n]: Y Save model card JSON? [Y/n]: Y 📝 Model card saved to: Day-07/model_card_StudyHours_to_MathScore.json

### 🔹 What I Learned
- How to build a small but **serious** ML workflow from scratch  
- Why **cross-validation** matters for generalization  
- Reading models beyond a single score using residuals  
- The importance of **reproducibility** through a model card

### 🔹 Future Improvements
- Multi-feature regression (normal equations)
- Polynomial basis expansion with regularization
- Confidence intervals and prediction intervals
- Export plots + report as a single HTML/PDF


---

## 🧠 Day 8 — From-Scratch Naive Bayes Text Classifier 📨

### 🔹 Project Title
**Naive Bayes Text Classifier** — pure-Python NLP classifier with CV, explainability, and a model card.

### 🔹 Project Description
A full, explainable NLP pipeline implemented **from scratch**:
- Tokenizes text (stopword filtering)
- Trains a **Multinomial Naive Bayes** with **Laplace smoothing**
- Performs **5-fold cross-validation** (macro precision/recall/F1, accuracy)
- Prints a **confusion matrix**
- Shows **most-informative tokens** via class log-odds
- Exports a **JSON model card** (priors, vocab size, CV metrics, metadata)
- Includes an **interactive demo** for live classification

**Input format:** CSV with columns: `text`, `label`.

### 🔹 Concepts Used
- Probabilistic modeling (Naive Bayes)
- Tokenization, stopwords, <UNK> handling
- Cross-validation for generalization
- Macro-averaged **precision/recall/F1**
- Explainability (log-odds indicative tokens)
- Reproducibility (JSON model card)

### 🔹 Example Session

🧠 Day 8 — Naive Bayes Text Classifier (From Scratch) Enter CSV path (must include columns 'text','label'): sms_spam.csv Laplace smoothing alpha [default 1.0]: Keep numeric tokens? [y/N]: y

🔍 5-fold Cross-Validation (macro-averaged): accuracy: 0.962 precision_macro: 0.955 recall_macro: 0.948 f1_macro: 0.951

✅ Fit on full data (reference metrics): accuracy: 0.971 precision_macro: 0.966 recall_macro: 0.959 f1_macro: 0.962

Confusion Matrix (rows=true, cols=pred): ham   spam ham    480     12 spam      5     73

💡 Most-informative tokens (log-odds): ham vs spam: meeting(2.31), home(2.07), okay(1.98), call(1.72), ... spam vs ham: free(3.45), prize(3.23), claim(3.10), txt(2.88), win(2.76), ... 📝 Model card saved to: Day-08/model_card_naive_bayes.json text> win a free prize now! Predicted: spam  |  Probabilities: {'ham': 0.013, 'spam': 0.987}

### 🔹 What I Learned
- Implementing a classic ML algorithm from first principles
- Measuring generalization with CV (not just train accuracy)
- Reading models via **most-informative features**
- Building explainable, documented ML pipelines

### 🔹 Future Improvements
- Add TF-IDF weighting
- Character-level n-grams for robustness
- ROC-AUC, PR-AUC plots
- Save/load trained model for reuse


---

## 🧠 Day 9 — Clustering Insight Engine 🌐

### 🔹 Project Title
**Clustering Insight Engine** — K-Means + PCA implemented from scratch with visualization and a model card.

### 🔹 Project Description
An unsupervised-learning engine that groups data into clusters, projects them via PCA, and visualizes patterns.  
It measures convergence, inertia, and exports a reproducible JSON summary.  
Demonstrates linear algebra, optimization, and data visualization fundamentals.

### 🔹 Concepts Used
- K-Means clustering (centroid update, inertia)
- PCA (eigenvectors via power iteration)
- Z-score normalization
- Algorithm convergence & tolerance
- Data visualization (`matplotlib`)
- Reproducibility via model card

### 🔹 Example Output

🌐 Day 9 — Clustering Insight Engine Enter CSV path: iris_numeric.csv Number of clusters (k): 3

✅ K-Means finished in 12 iterations. Inertia: 48.72 Cluster sizes: [52, 50, 48] 📝 Saved Day-09/model_card_kmeans.json

*(2-D scatter plot of clusters displayed)*

### 🔹 What I Learned
- Implementing iterative optimization algorithms (K-Means)
- Reducing high-dimensional data with PCA
- Visualizing and interpreting unsupervised results
- Writing clear, reusable scientific code

### 🔹 Future Improvements
- Add **Elbow method** for automatic k selection  
- Implement **Silhouette score**  
- Extend PCA to N components  
- Build a simple GUI for cluster exploration


---


🌟 The Journey Ahead


Each project will advance in difficulty and creativity — reflecting both engineering skill and problem-solving ability that top universities like MIT, Stanford, and Harvard value deeply.


---

✍️ Author

👨‍💻 Stuart Abhishek
16-year-old aspiring Computer Science Engineer
Dream Path:→ MIT / Stanford

> “Discipline beats talent. Consistency builds brilliance.”




---

🪪 License

This repository is licensed under the MIT License — free to learn, share, and contribute.


---

💫 Closing Note

> Every single day of this challenge represents my commitment to become a world-class programmer.
Through logic, creativity, and consistent hard work — I’ll reach the top, one line of code at a time.




---