# Location+: A Pitcher Command Metric

A custom metric that quantifies a pitcher's command as the value of a pitch location combined with how consistently that location is repeated, expressed on a 20-80 scouting scale.

## Background

I developed a custom metric, Location+, to quantitatively evaluate a pitcher's command. Command has traditionally been evaluated using outcome-based metrics like BB%, but walk rate alone doesn't fully capture how well a pitcher locates the ball. Metrics like Command+ attempt to estimate a pitcher's intended location, but public data alone has clear limits in accurately inferring a pitcher's actual intent.

Instead of trying to directly estimate intent, I defined command as *"the ability to consistently throw to locations that are difficult for hitters to attack."* I aimed to measure both throwing to good locations and repeating the same location, then combine both into a single metric.

## Data & Methods

I used 2025 MLB Statcast data collected via `pybaseball`. First, to estimate how valuable each pitch location was to the pitcher, I built an XGBoost regression model using `delta_pitcher_run_exp` as the target variable, with pitch location (`plate_x`, `plate_z`), pitch type, count, and batter/pitcher handedness as inputs. I applied 5-fold out-of-fold (OOF) prediction so that each pitch was predicted by a model that hadn't seen it during training, producing a conditional location value for each pitch.

Next, I measured pitch consistency separately. I applied Gaussian KDE to pitch coordinates grouped by pitcher, pitch type, and batter handedness to estimate a 2D distribution of pitch locations, then computed a log-density score for how tightly clustered each pitch was within its own distribution. This was then standardized as a Z-score by pitch type, indicating how consistent a given pitch type's location was relative to the league average.

Finally, I converted the consistency score into a cumulative probability and turned it into a weight of the form `2 x CDF(z)`, which was combined with the location value computed earlier. Pitches repeatedly thrown to good locations were rewarded, while pitches repeatedly thrown to poor locations were penalized more heavily.

I validated the metric's reliability by splitting pitches from the same pitcher/pitch-type into odd/even sequence groups, comparing the means of the two groups, and applying the Spearman-Brown formula to the Pearson correlation. For the final calculation, I required at least 60 pitches per pitch type, and only included pitchers where at least half of their pitch types met this threshold.

## Results

I compared each pitch type's location value to the league average, then computed a pitcher-level value as a pitch-count-weighted average. I converted each pitcher's value into a rank-based cumulative probability, mapped it to a normal distribution quantile, and rescaled it into a 20-80 scouting scale centered at 50 — **Location+**.

Based on the current code, 595 pitchers met the criteria for final evaluation. Tyler Mahle and Steven Matz received the highest grade with a Location+ of 80, followed by Parker Messick, Austin Cox, and Trevor Williams at 75.

I also tested the final distribution for normality. The D'Agostino K² test returned a p-value of 0.5914, failing to reject the null hypothesis of normality at the 0.05 significance level — confirming that the overall distribution was not overly distorted even after converting to the 20-80 scale.

## Limitations & Future Work

Location+ does not directly measure a pitcher's intended target location. So rather than measuring "Command" in the traditional sense, it's closer to a metric evaluating the value and repeatability of actual pitch locations. A pitcher who deliberately spreads the same pitch type across multiple locations as a strategy could be scored as having lower consistency under this framework.

The current location-value model also doesn't include velocity, movement, release point, or batter-specific characteristics. The actual difficulty of attacking a pitch at a given location can vary depending on stuff or batter type, so future work should add stuff-related variables to separate "a good location produced by good stuff" from "a pitch that is good purely because of its location."

Ultimately, if a team's own catcher glove/target-location or sign data were available to confirm what a pitcher actually aimed for, Location+ could be extended with intent-vs-execution error to become a more direct command evaluation metric.

## Run It

```bash
pip install -r requirements.txt
python main.py
```

`notebooks/pitch_location_plus.ipynb` contains the original analysis notebook with exploratory data analysis and step-by-step validation.

## Folder Structure

```
pitch-location-plus/
├── main.py                 # Pipeline entry point
├── src/
│   ├── preprocessing.py    # Statcast collection, count / target variable engineering
│   ├── modeling.py         # OOF XGBoost, KDE consistency score, stabilization test, Location+ scoring
│   └── visualize.py        # Zone value maps, KDE hot zones, stabilization curve, distribution plots
└── notebooks/
    └── pitch_location_plus.ipynb
```

## Tech Stack

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, Seaborn/Matplotlib, [pybaseball](https://github.com/jldbc/pybaseball)
