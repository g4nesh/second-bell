# Second Bell

Second Bell is an AI-powered rescue clock for high school cafeterias. It predicts "ghost components": unopened milk, whole fruit, and sealed sides that students take during lunch but are likely to return uneaten. The system recommends timed, safety-gated actions before those items become waste.

## Challenge Fit

- Track: High School
- Challenge 2: Make Climate Action Local and Real
- Direction A: Food Waste Rescue Radar
- Local user: Ms. Rivera, a cafeteria manager at Cedar Grove High School, working with a student eco-club monitor
- Devpost: https://devpost.com/software/second-bell

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
- Two-week LinearRegression forecast for the anonymous count of students staying after school.
- RandomForestRegressor for item-level after-school snack demand once that aggregate count is predicted.
- IsolationForest anomaly detector for unusual operating days.
- Sensor-fusion digital twin that reconciles camera, scale, load-cell, cooler, POS, kitchen, after-school, and disposal-bin signals.
- YOLO-ready computer-vision scaffold for detecting sealed returns, serving-bin drawdown, and opened tray waste.
- Rule-constrained safety compiler and recommendation engine with food-safety guardrails.
- Capped impact receipt that counts only approved primary actions.

## Demo Flow

1. Morning setup: menu, attendance, event, weather, cooler capacity, and monitor availability.
2. Ghost risk forecast: which unopened items are likely to return and when.
3. Sensor fusion digital twin: physical sensors and model predictions are reconciled into visible uncertainty ranges.
4. Computer vision reduction model: simulated or live YOLO-compatible detections show what is reducing and what is not.
5. Why this is happening: model drivers and anomaly review flags.
6. Action plan: timed cards for share-table cooler, two-wave staging, line placement, or after-school routing.
7. Human approval: Ms. Rivera approves only actions that satisfy the safety gate.
8. Impact receipt: recovered items, kg diverted, CO2e avoided, food value protected, labor minutes, and stockout risk.

## After-School Forecast

The sidebar no longer asks the user to guess "after-school students needing snacks." The app builds a daily aggregate history from the synthetic cafeteria data, takes the last 10 school days, and fits a `LinearRegression` model using day, weather, event, trend, and expected-attendance signals. The predicted count and range are then passed into the cafeteria scenario and recommendation engine.

## Sensor Fusion Digital Twin

The app models a practical cafeteria sensor layer:

- Serving-line camera for visible sealed-item and serving-bin detections.
- Scale beneath the tray-return platform.
- Load cells beneath high-waste serving bins.
- Temperature logger in the share-table cooler.
- Door-open sensor on the cooler.
- Production and staging counts from kitchen staff.
- Point-of-sale meal counts.
- Anonymous after-school attendance or demand counts.
- Scale beneath the final compost or disposal bin.

Second Bell reconciles inconsistent observations rather than trusting one source. For each item-period, the dashboard shows the camera count, POS/staging count, load-cell estimate, model prior, fused digital-twin estimate, and uncertainty range.

## Hardware Prototype Path

For a hackathon-friendly prototype, the repo documents a Mac Force Touch trackpad bridge:

- TrackWeight: https://github.com/KrishKrosh/TrackWeight
- OpenMultitouchSupport: https://github.com/KrishKrosh/OpenMultitouchSupport

Those projects are useful for demonstrating pressure-derived load-cell readings through Swift/macOS. They are not treated as certified production food-service scales; deployment should use calibrated load cells and temperature loggers.

## Vision Deployment Path

The included `ImageReductionVisionModel` can compare before/current serving-bin images to detect whether an item station is reducing or staying untouched. The `CafeteriaVisionModel` wrapper can also load `models/cafeteria_yolo.pt` if trained weights are added and `ultralytics` is installed. The app runs deterministic simulated detections for the default dashboard and exposes an upload-based baseline model for real before/after image checks. Export commands are shown in the dashboard for ONNX, Apple CoreML, and TensorRT targets.

Optional live CV install:

```bash
pip install -r requirements-vision.txt
```

## Run Locally

```bash
pip install -r requirements.txt
python src/simulate_data.py
python src/train_models.py
python scripts/smoke_check.py
streamlit run app.py
```

The app validates model artifacts at startup. If a checked-in model cannot be loaded or its manifest does not match the current feature schema, Second Bell regenerates synthetic data and retrains the models automatically.

## Deploy to Vercel

The Vercel deployment publishes the static demo site, not the Streamlit dashboard runtime:

```bash
npm run build
```

Vercel uses `vercel.json` to run that build and serve the generated `dist/` directory. The interactive Streamlit dashboard should be deployed to a Python app host that supports long-running Streamlit processes.

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
