# Devpost Copy - Second Bell

## Qualifier Approval Code

Enter the 8-character code from the AI Qualifier email.

## Track and Challenge Selection

- Track: High School
- Challenge: Challenge Brief 2 - Make Climate Action Local and Real
- Direction: Direction A - Food Waste Rescue Radar

## Elevator Pitch

Second Bell predicts ghost components - unopened milk, whole fruit, and sealed sides that students took but never actually wanted - and helps a cafeteria manager rescue them before the safe recovery window closes.

## Project Description

Most food-waste tools treat a cafeteria like one big trash stream. Second Bell goes narrower: it focuses on unopened school-lunch items that count as served but come back untouched. These "ghost components" can include milk cartons, apples, bananas, sealed fruit cups, yogurt cups, carrots, and packaged sides.

The local setting is Cedar Grove High School, a fictional but realistic high school with two lunch periods, one cafeteria manager, a student eco-club, a monitored share-table cooler, and after-school programs such as tutoring, robotics, and track practice.

The user is Ms. Rivera, the cafeteria manager. She has only a few minutes between lunch periods to decide whether to stage cold items differently, deploy a monitored share-table cooler, move default fruit placement, or route eligible sealed surplus to approved after-school snack demand.

Second Bell helps her answer:

- Which unopened items are likely to return during each lunch period?
- Why is this happening today: menu, weather, event, attendance, line placement, or after-school demand?
- Which action should happen before the rescue window closes?
- Which actions are blocked because food-safety conditions are missing?
- What impact should the school log after human approval?

The novelty is the combination of cafeteria line behavior, meal-component selection pressure, cold-chain share-table timing, after-school snack matching, and AI action recommendations. It is local, operational, and measurable rather than awareness-only.

## AI Architecture Explanation

### Inputs

- Menu and entree
- Day of week
- Lunch period
- Expected attendance
- Weather tag
- School event tag, such as field trip or exam day
- Line setup and component placement
- Planned and staged item counts
- Share-table monitor availability
- Cooler capacity
- After-school activity demand
- Item metadata: sealed/whole, cold-chain required, cost, weight, and CO2e estimate

### AI Capabilities Used

- Pattern detection
- Predictive modeling
- Quantile forecasting
- Recommendation scoring
- Anomaly detection

### Processing

Second Bell trains on a synthetic cafeteria history of 540 post-break school days. Each row represents one food item in one lunch period. The model stack includes:

1. RandomForestClassifier for high ghost-component risk.
2. Quantile GradientBoostingRegressor models for low, median, and high unopened-return forecasts.
3. RandomForestRegressor for after-school snack demand.
4. IsolationForest for unusual operating days.
5. A rule-constrained rescue-window state machine that blocks unsafe or unmonitored actions.

The recommender scores possible actions using expected recovery, CO2e avoided, food value protected, labor, stockout risk, safety risk, and confidence. The final impact receipt counts only one approved primary marginal action per item/lunch, so alternatives do not inflate the numbers.

### Outputs

- Ghost risk forecast by item and lunch period
- Predicted unopened-return range
- Model drivers in plain language
- Anomaly review flags
- Timed action cards with deploy-by times and rescue windows
- Safety gates and manager approval toggles
- Impact receipt: items recovered, kg diverted, CO2e avoided, food value protected, labor minutes, and stockout risk

## Human-in-the-Loop Design

The AI does not decide whether food is safe or policy-approved to redistribute.

That decision must stay with cafeteria staff because a model cannot verify packaging integrity, temperature logs, local health rules, school policy, or whether a cooler was actually monitored. Second Bell can recommend an action and explain its assumptions, but Ms. Rivera must approve the action before it counts in the impact receipt.

## Responsible AI Guardrail

Risk: unsafe redistribution or over-reliance on a forecast.

Mitigation: Second Bell uses strict eligibility gates. Cold-chain items require sealed packaging, a monitored cooler, and a 41 F-or-colder safety gate. Opened or prepared food is blocked from the share-table workflow in this MVP. Every counted action uses confidence ranges and human approval. The app also surfaces anomaly flags so unusual days get manager review.

## Data Disclosure

The project uses synthetic data only. The simulator creates realistic item-by-lunch-period cafeteria rows for a typical high school, including:

- menu item and food component type
- lunch period
- planned, staged, selected, returned, rescued, and discarded counts
- weather and school-event tags
- expected and actual attendance
- line position
- share-table monitor and cooler capacity
- after-school activity count
- estimated kg diverted, CO2e avoided, and dollar value protected

No private school data is required. The system does not use student names, IDs, demographics, free/reduced-lunch status, or individual eating behavior.

## Tools Used

Free/open-source:

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly
- joblib
- Synthetic CSV data generated locally

AI assistance:

- OpenAI/Codex was used to help plan, code, debug, and polish the MVP and submission materials.

Paid tools/APIs:

- None required for the working prototype.

## Built With

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly, joblib, synthetic data generation, Random Forests, Gradient Boosting, quantile regression, conformal-style padding, Isolation Forest anomaly detection, rule-based recommendation engine, rescue-window state machine, model manifest validation, and local CSV/model artifacts.

## Example Demo Scenario

Cedar Grove High has a rainy Tuesday pasta lunch. Juniors are away on a field trip, and robotics, tutoring, and track practice all meet after school. The cafeteria usually stages most milk before second lunch, but pasta days often create a hidden pattern: students take milk, do not drink it, and unopened cartons pile up near tray return.

Second Bell predicts which items are likely to come back unopened, explains the drivers, surfaces any unusual-day flags, and recommends timed interventions. Ms. Rivera can approve or reject each primary action. The impact receipt updates only for approved safe actions.

## What Makes It Different

Second Bell is not just visualizing waste. It predicts a narrow, recoverable waste mechanism and recommends an operational response inside a short time window.

Most school food-waste projects focus on dashboards, donation matching, or general cafeteria forecasting. Second Bell focuses on the hidden pathway from "student took a required or convenient component" to "sealed item becomes trash unless the cafeteria acts now."

## Impact

Second Bell reports impact as ranges instead of false precision:

- unopened items recovered
- kilograms of food diverted
- estimated CO2e avoided
- food value protected
- labor minutes required
- highest stockout risk
- approved and blocked actions

The goal is to help a cafeteria move from awareness to repeatable action: stage differently, deploy a monitored share table, route approved sealed items to after-school demand, or block unsafe reuse.

## Source Grounding

- USDA share-table guidance supports whole or unopened items and unopened milk with cold storage when local rules allow it: https://www.usda.gov/sites/default/files/guidance-documents/fns.sp41cacfp13sfsp15-2016-shareTables.pdf
- EPA's Wasted Food Scale prioritizes prevention, donation/redistribution, and keeping food used for its intended purpose: https://www.epa.gov/sustainable-management-food/wasted-food-scale
- ReFED frames food-waste action around prevention, rescue, recycling, measurement, and AI/predictive analytics: https://refed.org/food-waste/the-solutions/

## What We Learned

The strongest climate-action tools are not always the broadest ones. A small local bottleneck can be more useful than a giant dashboard. By focusing on ghost components and the short rescue window after lunch, Second Bell turns a vague sustainability goal into a concrete cafeteria decision: act now, stage differently, route safely, or block the action.

## Challenges

The hardest part was making the project technical without making it unrealistic for a school. We avoided private student data, sensors, and production-grade food-safety automation. Instead, we built a synthetic cafeteria simulator, lightweight ML models, transparent recommendation scoring, and a human approval workflow.

## What's Next

Future versions could add real cafeteria audit logs, anonymous menu feedback, district-approved share-table rules, QR-based eco-club logging, and weekly intervention experiments comparing line placement, two-wave staging, and after-school routing.

## Submission Checklist

- Qualifier approval code entered.
- High School Track selected.
- Challenge 2 and Direction A selected.
- Project description pasted.
- AI architecture field explains inputs, processing, models, and outputs.
- Human-in-loop field states that food-safety approval stays with cafeteria staff.
- Responsible AI field names unsafe redistribution as the risk and safety gates as the mitigation.
- Tools used field separates open-source tools from OpenAI/Codex assistance.
- Data disclosure field says the data is synthetic and uses no student-level records.
- Working demo link or walkthrough added.
- 3-5 minute video covers: problem/user, how the AI works, app walkthrough, and responsible AI choice.
