# Objective

## What I Want

I want to know who the best college football teams actually are, based on what happened on the field - not what 13 people in a room decided after watching highlight reels.

I want to understand *why* a team is ranked where they are, down to the individual game. Not "they pass the eye test" or "they're battle-tested." Show me the math. Show me which wins helped, which losses hurt, and by exactly how much.

I want to predict games. Not with gut feelings, but with a system I can backtest, validate, and trust. When I say "Georgia has a 62% chance to beat Texas," I want that number to mean something - derived from actual performance data, not vibes.

I want to experiment. What if margin of victory mattered more? What if road wins counted double? What if we penalized bad losses more harshly? I want to turn those knobs, see the rankings shift in real-time, and discover which philosophy produces the most accurate predictions.

I want to catch the upsets before they happen. When my algorithm disagrees with Vegas, I want to know why. Is it because the favorite played a weak schedule? Because the underdog has been undervalued all season? Give me the edge.

I want transparency in a sport drowning in opacity.

---

## The Frustration

Every Tuesday during the season, I watch the CFP rankings drop and feel my blood pressure rise.

**"Alabama moves up after their bye week."** How? They didn't play. Meanwhile, Boise State wins by 30 against a ranked opponent and stays put. The committee mumbles something about "body of work" and "conference strength" - criteria they invented on the spot and will contradict next week.

**"We look at championships won."** Excuse me? Championships won this season, or championships won by players who graduated a decade ago? You're telling me the name on the jersey matters? That's not ranking - that's brand management.

**"Quality loss."** The phrase that launched a thousand arguments. Apparently losing to Georgia is actually *good* for your resume, but beating Tulane doesn't count because... reasons. The committee applies these standards selectively, and there's no way to prove it because there's no formula to audit.

I'm tired of arguing about rankings with no shared framework. I'm tired of "agree to disagree" when we could actually compute the answer. I'm tired of a system designed to be unaccountable.

---

## What This Project Is

This is a college football ranking engine that:

1. **Takes only on-field results as input** - scores, locations, dates. No preseason polls. No recruiting stars. No conference labels. No "brand equity."

2. **Solves the circular problem of strength of schedule** - your rating depends on who you beat, but their rating depends on who *they* beat. The algorithm iterates until everything stabilizes, like Google's PageRank but for football.

3. **Explains every ranking completely** - click on any team and see exactly why they're ranked #7: these wins contributed this much, this loss cost that much, here's their schedule strength relative to the field.

4. **Lets me tune the philosophy** - 44 configurable parameters across 12 categories. Want to reward road wins more? Punish bad losses harder? Value margin of victory less? Adjust the sliders, watch the rankings update, see if your philosophy predicts games better.

5. **Validates against reality** - backtest any configuration against historical seasons. Check correlation with the actual CFP selections. Measure prediction accuracy against game outcomes. See if your tweaks make things better or worse.

6. **Predicts future games** - combine multiple signals (my algorithm, Vegas lines, SP+, Elo, pregame win probability) into a consensus prediction. Find games where the sources disagree. Spot value.

---

## How I Want to Use It

### Scenario 1: It's Tuesday, Rankings Just Dropped

The committee put Ohio State at #2 and Penn State at #6. My algorithm has them at #3 and #4. I want to see:
- What's causing the gap?
- Is it a philosophical difference (committee values wins, I value margin)?
- Or is the committee just wrong?

I load up the comparison view, see that Penn State's strength of victory is actually higher than Ohio State's, and now I have ammunition for the group chat.

### Scenario 2: It's Thursday, I'm Setting My Picks

I've got a 10-game slate to predict. I don't want to spend 2 hours researching each matchup. I want to:
- Dump the matchups into the system
- Get win probabilities for each game
- See where my algorithm disagrees with Vegas by more than 10%
- Focus my research time on those high-edge games

I run batch predictions, find three games where I think Vegas is wrong, and place my bets with confidence (or at least with math).

### Scenario 3: It's Saturday, Something Wild Just Happened

Vanderbilt just beat Alabama. Everyone's losing their minds. I want to know:
- Did my algorithm see this coming?
- What were the factors? (Weak favorite SOS? Small rating gap? Road favorite?)
- How does this change the rankings?

I check the upset analysis, see that my algorithm had Vandy at 38% (much higher than the 18% Vegas implied), and feel vindicated. Then I run the new rankings and see how the landscape shifted.

### Scenario 4: It's December, Playoffs Are Set

The 12-team bracket is locked. I want to simulate:
- What's each team's probability of winning it all?
- If Georgia beats Texas in round 1, what happens to their championship odds?
- Which first-round upset would cause the biggest chaos?

I build the bracket, assign my win probabilities, run Monte Carlo simulations, and have data-driven takes for the office pool.

### Scenario 5: It's the Offseason, I'm Experimenting

I have a theory: recency weighting should matter more. Teams that are hot entering the playoffs perform better than teams that peaked in October. I want to test this.

I create a config with aggressive recency weighting (half-life of 4 weeks instead of 8). I backtest against 2019-2024 seasons. I check if prediction accuracy improves. If it does, I save this as my new default profile for next season.

---

## The Core Beliefs

### 1. On-Field Results Are the Only Fair Input

The moment you let "conference strength" or "brand value" into the formula, you've created a system that advantages incumbents. The only way to give Boise State a fair shot against Alabama is to ignore everything except what happened in actual games this season.

### 2. Losses Should Hurt, But Context Matters

Losing is bad. There's no such thing as a "good loss." But losing to the #1 team by 3 points on the road is *less bad* than losing to an unranked team at home by 20. The algorithm should capture this nuance without ever rewarding a loss.

### 3. Margin Matters, But With Limits

Beating someone 42-7 is more impressive than beating them 21-20. But we shouldn't incentivize running up the score. Cap the benefit at ~4 touchdowns and use diminishing returns so the first touchdown of margin is worth more than the fourth.

### 4. Transparency Beats Accuracy

A formula that's 85% accurate and fully explainable is more valuable than a black-box neural net that's 90% accurate but can't tell you why. I need to trust the system, and trust requires understanding.

### 5. Configuration Over Dogma

There's no single "correct" way to rank teams. Reasonable people disagree about how much margin should matter, how much to penalize bad losses, whether conference strength should factor in. The system should let me express my philosophy through parameters and see the consequences.

---

## Success Looks Like

**For rankings:**
- I can explain any team's position in 30 seconds using the decomposition view
- Rankings feel "right" to the eye test while being completely derived from data
- Controversial committee decisions are surfaced and quantified ("Committee ranks X 4 spots higher than algorithm")

**For predictions:**
- Consensus predictions beat Vegas at least 52% of the time (the threshold for long-term profit)
- High-confidence predictions (70%+) hit at 70%+ rate
- Upset detection catches at least 30% of Vegas upsets before they happen

**For experimentation:**
- I can create a new config, backtest it against 5 seasons, and see accuracy metrics in under a minute
- I can compare two configs side-by-side and understand exactly where they diverge
- I've found at least one configuration tweak that improved prediction accuracy vs the defaults

**For the community:**
- I can share a config as a JSON file or URL and others can reproduce my rankings exactly
- I can export rankings as images or tables for social media arguments
- I can prove my claims with data instead of assertions

---

## What This Is Not

This is not a claim that the algorithm is "correct" and humans are "wrong." It's a tool for making the ranking debate empirical instead of tribal.

This is not a betting system (though it helps with betting). Past performance doesn't guarantee future results. The edge is small. Gamble responsibly.

This is not a replacement for watching games. The algorithm tells you who *has been* best, not who *is* best in some cosmic sense. It can't see that the quarterback is injured or that the coach just got fired.

This is not a finished product. The philosophy is encoded in 44 parameters, each representing a choice about what matters. I'll keep tuning, keep backtesting, keep learning. The goal is a system that gets better over time as I understand the sport better.

---

## Why This Matters to Me

I've been watching college football for 20 years. I've seen teams get robbed by poll inertia, brand bias, and arbitrary committee logic. I've watched undefeated Group of Five teams get told they "don't belong" while two-loss blue bloods waltz into the playoff.

I can't fix the system. But I can build a mirror to hold up to it.

When the committee makes a decision that contradicts on-field results, I want to quantify exactly how wrong they are. When someone says "Alabama just looks like the better team," I want to ask "better by what measure?" and have an answer ready.

This is my attempt to bring receipts to the argument.

---

## The Promise

Give me the game results. I'll give you rankings you can trust, predictions you can test, and explanations you can understand.

No smoke. No mirrors. No committee.

Just math.
