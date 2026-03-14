# Project journal

## 14 January 2026 - Supervisor kick-off meeting

Met [supervisor] at 14:00 in the Diamond. Notes from the discussion:

- They want a *live dashboard demo on a recent day* at the viva, not a static
  screenshot. Implies the pipeline has to be runnable end-to-end on a laptop
  with data through to within a few days of the demo. Adding this as a
  functional requirement (live demo on contemporary data).
- Strong preference for *results-first sequencing* in the report - they
  pushed back on the conventional dev-then-results ordering and want the
  CV table on page one of the analysis chapter. Will restructure the chapter
  plan accordingly.
- Agreed n=1 is acceptable but the report has to be explicit about the
  noise ceiling. Their phrasing: "do not pretend you have a sample of
  athletes."
- Sprint cadence: five two-week sprints, retros at the end of each, no
  mid-sprint scope creep without it being logged here.
- Next check-in pencilled in for 14 March around the modelling phase.

Actions for me:
- Add FR for live dashboard demo (FR-9 or FR-10).
- Add FR for results-first report ordering as a documentation requirement.
- Move on to data plan with the public-dataset fallback the proposal already mentioned.

## 28 February 2026 - Back-to-back workouts and alignment edge case

Found a corner in the listening alignment that I had not thought about: when
two workouts happen within four hours of each other (a *brick* session, e.g.
gym then run) the post-workout listening windows overlap. With a fixed two-hour
window after each workout, tracks played between workout-B's start and
workout-A's window-end get double-counted.

In my own data this has happened twice that I can see - both gym-then-run
weekend sessions. Not enough to matter statistically but the decision needs
logging or it will get queried at the viva.

Resolution: a track is assigned to the *earlier* workout's window, and the
later workout's window starts only after the earlier window closes. This is
implemented in `align_workouts_to_listening` by truncating the earlier window
whenever the next workout's start time lands inside it.

Protected by `tests/test_features_window.py::test_back_to_back_workouts_assign_to_earlier`.

## 14 March 2026 - Supervisor check-in

Forty-minute video call covering modelling progress. Headline points:

- Showed early pilot CV results on synthetic data. Ridge, GBM and the
  attention-LSTM are all within a tenth of an MAE point of each other and
  all roughly track the seasonal day-of-week baseline. Supervisor was
  unsurprised - their phrasing was "this is the n=1 ceiling, not a
  model problem."
- Discussed MSE vs MAE as the training loss for the LSTM. Pilot runs with
  MSE produced sharply contracted predictions (regression to the in-fold
  mean). Agreed to switch to `nn.L1Loss` - the predicted distribution
  widens, correlation improves slightly, MAE rises by ~0.1 point. Worth it
  for the wider spread.
- Supervisor flagged that the daily-email FR-12 may need de-scoping if the
  models do not show a positive cross-validated \(R^2\). Will reassess once
  the real data arrives and the full CV table is in.
- Reminded me to be ruthless about negative results in the report:
  frame the day-of-week tie as a *finding* about the data, not a model
  failure.

Next check-in: mid-late April once real data is ingested.
