# Objective

## The Problem

NCAA FBS football lacks a transparent, reproducible method for ranking teams.

### Current System: The College Football Playoff Committee

A 13-member committee meets weekly to produce rankings that determine which teams enter the College Football Playoff. Their process is opaque—they watch games, discuss in private, and vote. No formula is published. No weights are disclosed.

This creates several documented issues:

**1. Brand Bias**

Teams with historical prestige receive favorable treatment. A one-loss Alabama is ranked differently than a one-loss Tulane, even with comparable résumés. The committee has explicitly stated they consider "championships won" and program history—factors disconnected from current-season performance.

**2. Poll Inertia**

Preseason rankings influence final rankings. Teams that start ranked high tend to stay high because voters anchor to prior positions. A team that starts #3 and loses once often remains above an undefeated team that started unranked.

**3. Inconsistent Standards**

The committee applies different logic to different teams:
- "They don't have a quality win" (used against Group of Five teams)
- "They have a quality loss" (used to justify Power Five teams)

These criteria are selectively invoked with no systematic application.

**4. Non-Reproducibility**

Two reasonable people watching the same games can reach different conclusions. There is no way to audit the committee's decisions or identify errors in their reasoning.

### Previous Automated Systems

The BCS (1998–2013) used computer rankings, but:
- Multiple algorithms with different methodologies were averaged
- Margin of victory was banned after 2001, removing a key signal
- Human polls still comprised 2/3 of the formula

The computer components (Sagarin, Massey, Colley, etc.) were an improvement but were deliberately weakened and eventually abandoned.

---

## The Goal

Build a ranking system with the following properties:

| Property | Definition |
|----------|------------|
| **Deterministic** | Same inputs always produce same outputs |
| **Transparent** | Every ranking can be fully explained by the formula |
| **Auditable** | Anyone can verify calculations independently |
| **Bias-resistant** | No consideration of team name, conference, or history |
| **Comprehensive** | Accounts for wins, losses, margin, venue, and opponent strength |

The system should answer: "If we ignore everything except what happened on the field this season, how should teams be ranked?"

---

## The Approach

### Core Principle: Recursive Strength of Schedule

A team's rating depends on who they beat and lost to. But opponent quality depends on *their* opponents. This creates a circular dependency that must be solved iteratively.

The algorithm:
1. Assigns every team an initial rating (0.500)
2. Calculates each team's new rating based on game outcomes and current opponent ratings
3. Repeats until ratings stabilize (converge)

This is mathematically similar to Google's PageRank: a win against a highly-rated team is worth more than a win against a low-rated team, and ratings propagate through the network of games.

### Objective Inputs Only

| Input | Source | Ambiguity |
|-------|--------|-----------|
| Final score | Box score | None |
| Location | Schedule data | Minimal (some neutral sites mislabeled) |
| Date/week | Schedule data | None |

Not used:
- Team name or conference affiliation
- Historical performance
- Recruiting rankings
- Preseason predictions
- "Eye test" or style points

### Capped Margin of Victory

Margin matters—a 28-point win demonstrates more control than a 1-point win. But unlimited margin credit incentivizes running up scores.

Solution: Logarithmic bonus capped at 28 points. A 35-point win earns the same as a 28-point win. Diminishing returns mean the first touchdown of margin is worth more than the third.

### Loss Penalty Structure

The key insight distinguishing this algorithm: **losses subtract opponent rating from your score**.

- Beat a 0.80-rated team: +0.80 to your rating component
- Lose to a 0.80-rated team: −0.80 to your rating component

This means:
- Losing to good teams hurts less than losing to bad teams
- But losing always hurts (no "quality loss" bonus)
- The damage scales with opponent strength

---

## Success Criteria

The algorithm succeeds if:

1. **Rankings pass sanity checks**: Undefeated teams with reasonable schedules appear near the top; winless teams appear at the bottom

2. **Explainability**: Any ranking can be decomposed into "you beat X (rated Y), lost to A (rated B), with these margins and venues"

3. **Correlation with outcomes**: Teams ranked higher should win head-to-head matchups more often than chance (testable against bowl games)

4. **Stability**: Small changes in inputs (one game) produce proportional changes in output (no chaotic swings)

5. **Convergence**: Algorithm terminates reliably within reasonable iterations

---

## What This Is Not

This project does not claim to:

- Predict future game outcomes (it's retrospective)
- Determine who the "best" team is in some platonic sense
- Replace all human judgment in playoff selection
- Account for injuries, weather, or other contextual factors

It provides one objective input to the ranking discussion—a formula-derived ordering based solely on game results and schedule strength.

---

## Why Now

The College Football Playoff expanded to 12 teams in 2024. More slots means more edge cases, more arguments about who deserves inclusion, and more opportunities for inconsistent committee logic to produce controversial outcomes.

A transparent algorithm provides:
- A benchmark to compare committee decisions against
- Evidence when committee rankings diverge from on-field results
- A starting point for discussions about what factors *should* matter

The goal is not to replace human judgment entirely, but to make the implicit criteria explicit and testable.
