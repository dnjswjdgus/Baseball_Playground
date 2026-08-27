# Bunt Decision Analysis by Batting Order

Comparing the value of swinging away vs. sacrifice bunting with a runner on first and no outs, based on the offensive quality of the current and next batter — measured both by expected runs and by the probability of scoring at least one run.

## Background

The sacrifice bunt is one of the plays most consistently criticized as inefficient in sabermetrics. Bill James's "Ten Commandments of Sabermetrics" lists *"Thou shalt not bunt"* as its very first principle — trading an out for a base generally reduces expected runs.

But I wondered whether this conclusion holds in every situation. If the current batter is weak and the next batter is a top-of-the-league hitter like Aaron Judge, it might actually be more advantageous to advance the runner and hand the scoring opportunity to the next batter, rather than have the current batter try to drive in the run himself.

So I analyzed how the value of swinging away versus sacrifice bunting, in a no-out runner-on-first situation, changes depending on the offensive quality of the current and next batter. I compared strategy value not just by expected runs, but also by the probability of scoring at least one run before the inning ends — to see whether the optimal choice depends on the batting team's actual goal.

## Data & Methods

I used 2025 MLB Statcast play-by-play data collected via `pybaseball`. First, I built a Run Expectancy Matrix by calculating the average runs scored through the end of the inning for each of the 24 base-out states (combinations of outs and runners). I also separately calculated the probability of scoring at least one run before the inning ends from each state.

I represented batter quality using wOBA and a base-out run expectancy framework, following the strategy evaluation approach used in Tom Tango, Mitchel Lichtman, and Andrew Dolphin's *The Book: Playing the Percentages in Baseball*. Using 2025 FanGraphs wOBA weights, I calculated each batter's wOBA, then, for batters with at least 300 plate appearances, weighted-averaged the outcomes of players with wOBA close to a given target value. This produced estimated probabilities for BB/HBP, single, double, triple, home run, strikeout, in-play out, and double play.

Swinging away was calculated as the expected runs and score probability when the current batter hits with a runner on first and no outs. To simplify the bunt scenario, I assumed a successful sacrifice bunt converts a no-out runner-on-first state into a one-out runner-on-second state, then calculated the expected runs and score probability for the next batter hitting from that state.

Finally, I varied both the current and next batter's wOBA from .200 to .498 across all combinations, and visualized the difference between swinging and bunting as a heatmap.

## Results

By **expected runs**, swinging away outperformed bunting across every batter combination analyzed. Even in the most extreme case — a weak current batter and a strong next batter — the deliberate cost of an out was never offset.

By **probability of scoring at least one run**, however, the picture changed. As the current batter's wOBA decreased and the next batter's wOBA increased, a region emerged where bunting produced a higher scoring probability than swinging. For example, with a current batter at .299 wOBA and a next batter at .480 wOBA, swinging away had about 0.129 more expected runs than bunting, but bunting had about 1.56 percentage points higher probability of scoring at least one run.

In other words, if the goal is to maximize runs scored, swinging away was consistently better — but in situations where scoring even a single run is essential, bunting could offer a higher scoring probability depending on the batting order composition. This shows that the same strategic choice can lead to different conclusions depending on whether expected runs or score probability is used as the decision criterion.

## Limitations & Future Work

This analysis assumed a successful sacrifice bunt cleanly converts a no-out runner-on-first state into a one-out runner-on-second state. In real games, bunts can fail, lead runners can be thrown out, bunts can go for hits, or fielding errors can occur — so the actual expected value of a bunt attempt could differ from this analysis.

I also only modeled the offensive quality of the current and next batter individually; the rest of the batting order is implicitly folded into the league-average run expectancy. Real decision-making should also weigh the composition of the lineup beyond the next batter, baserunner speed, defensive positioning, and pitcher type.

This analysis is meant less as a model for predicting any single game situation, and more as a way to quantify, with sufficient sample size, under what conditions a given strategy becomes relatively more favorable. Future work could incorporate bunt success probability, baserunner speed, and the rest of the batting order into the model, and extend it to Win Expectancy that accounts for inning and score differential — producing more concrete, game-usable criteria for strategic decisions.

## Run It

```bash
pip install -r requirements.txt
python main.py
```

`notebooks/bunt_analysis.ipynb` contains the original analysis notebook with exploratory data analysis and step-by-step validation.

## Folder Structure

```
bunt-strategy-analysis/
├── main.py                 # Pipeline entry point
├── src/
│   ├── preprocessing.py    # Statcast collection, PA table, RE / score-probability tables
│   ├── modeling.py         # wOBA-based event probability estimation, swing vs. bunt comparison
│   └── visualize.py        # Decision heatmap
└── notebooks/
    └── bunt_analysis.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, [pybaseball](https://github.com/jldbc/pybaseball) (Statcast collection)
