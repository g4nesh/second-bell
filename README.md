# Second Bell

Second Bell is an AI-powered rescue clock for high school cafeterias. It predicts "ghost components": unopened milk, whole fruit, and sealed sides that students take during lunch but are likely to return uneaten. The system recommends timed, safety-gated actions before those items become waste.

## Challenge Fit

- Track: High School
- Challenge 2: Make Climate Action Local and Real
- Direction A: Food Waste Rescue Radar
- Local user: Ms. Rivera, a cafeteria manager at Cedar Grove High School, working with a student eco-club monitor

Second Bell is not a general food-waste dashboard. It focuses on one local pathway:

```text
lunch-line behavior -> unwanted sealed item -> unopened return -> cold-chain timer -> share table or after-school snack demand
```

## AI Capabilities

- Synthetic data generation for 540 post-break synthetic school days.
- Pattern detection for menu, weather, attendance, event, lunch-period, and line-position effects.
- Feature encoder for 22 aggregate item-period cafeteria signals.
- RandomForestClassifier for high ghost-component risk.
- Quantile GradientBoostingRegressor heads for Q10, Q50, and Q90 unopened-return intervals.
- RandomForestRegressor for after-school snack demand.
- IsolationForest anomaly detector for unusual operating days.
- Rule-constrained safety compiler and recommendation engine with food-safety guardrails.
- Capped impact receipt that counts only approved primary actions.

## Demo Flow

1. Morning setup: menu, attendance, event, weather, cooler capacity, monitor availability, and after-school demand.
2. Ghost risk forecast: which unopened items are likely to return and when.
3. Why this is happening: model drivers and anomaly review flags.
4. Action plan: timed cards for share-table cooler, two-wave staging, line placement, or after-school routing.
5. Human approval: Ms. Rivera approves only actions that satisfy the safety gate.
6. Impact receipt: recovered items, kg diverted, CO2e avoided, food value protected, labor minutes, and stockout risk.

## Run Locally

```bash
pip install -r requirements.txt
python src/simulate_data.py
python src/train_models.py
python scripts/smoke_check.py
streamlit run app.py
```

The app validates model artifacts at startup. If a checked-in model cannot be loaded or its manifest does not match the current feature schema, Second Bell regenerates synthetic data and retrains the models automatically.

## Data Disclosure

All data is synthetic. The generator creates item-by-lunch-period rows with realistic patterns for day of week, weather, school events, expected and actual attendance, lunch periods, menu items, line placement, share-table monitor availability, cooler capacity, after-school activity count, unopened returns, share-table pickups, after-school pickups, discarded items, and estimated food/CO2e/dollar impacts.

No student-level or private school data is used. The app does not use student names, IDs, demographics, free/reduced-lunch status, or individual eating behavior.

## Responsible AI

Second Bell does not decide whether food is safe or policy-approved to redistribute. It predicts risk and suggests possible actions, but the cafeteria manager remains responsible for food-safety, policy, and operational approval.

Key guardrails:

- Cold-chain items require sealed packaging, a monitored cooler, and a 41 F-or-colder safety gate.
- Every counted action requires an approval toggle.
- Prepared or opened food is blocked from the share-table pathway in this MVP.
- Impact totals are capped so alternatives do not inflate recovery claims.

## Source Grounding

- USDA share-table guidance: https://www.usda.gov/sites/default/files/guidance-documents/fns.sp41cacfp13sfsp15-2016-shareTables.pdf
- EPA Wasted Food Scale: https://www.epa.gov/sustainable-management-food/wasted-food-scale
- ReFED food-waste solutions: https://refed.org/food-waste/the-solutions/
