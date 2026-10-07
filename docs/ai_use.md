There are 15 modules within the vetuner with their tests, these were largly drafted by Clude from the specifications I set, I then reviewed, executed and corrected the modules to reflect what I needed from the program. The project design, the validation strategy, every parameter choice, and the interpretation of results are mine. Where AI output was wrong, the error was found by running it.


## 2026-09-04 — Project scoping

Used for: pressure-testing my project idea against the brief, and identifying a validation strategy.

Outcome: I had the pipeline concept (log in, VE table out) but no validation plan. Claude proposed the manufactured solution approach known true VE surface, deliberately wrong starting table, noise on the simulated measurements only. I adopted this.

Verified: checked against the brief, which lists manufactured solutions as an accepted validation method. Reasoned through the logic myself to confirm the true surface stays hidden from the pipeline.

My decision: scope boundary, deferring exhaust transport delay to a later phase,  synthetic data.

## 2026-09-04 Environment setup

Used for: setting up and connecting the git environment to vs code, Claude gave Bash commands (`mkdir -p`, `touch`)

Used becuase: first time using vs code and git for a full scale project, was still new to the setting up side.

Limitations: Claude advised downgrading Python 3.14 to 3.12 when 3.14 was already
working and as such caused issues, reverted back to 3.14. When trying to setup, Claude linked to a python.org page carrying no installer



## 2026-09-04 Drive cycle and VE surface 

Used for: Understanding how to implament a model a drive cycle of an engine, generation of sample points, there were 10500 sample points that needed to be mapped out onto a grid in order to simulate what an actual system would do. Claude helped me easily generate those points.

Not used for: all ideas about how the drive cycle and volumetric efficiency surface should be built and used, Claude was used to help me with the implamentation as well as the harder dialogs of code.

Limitations: There is a nautral region in an engine where there was a spike in the error rate to which a test asserting the base map varies smoothly failed as the Ai had set the threshold values low.

My Decision: Increase the threshold, when doing the reruns of different engines with different maps, no further errors were raised, as such the new error rate held.


## 2026-09-13 Engine model and sensors 

Used for: Creating and coding the engine model as well as creating the injector pulsewidth table,

How this was verafied: The values and parameters were all set by me with the values coming from my own project and real life expiriance of tuning my own bike. All functions were inspected and all values verified to an accurate level.

Limitations: A test made by Ai assumed a correct VE table would deliver target AFR to machine precision. It does not, the actual error is 0.03%, from bilinear
interpolation of a curved surface. This is a real property of the method,
not a bug: the "discretisation floor".

My decision: The test made by Ai was testing for something valid however it was not measuring the right value which I understood, I only needed to increase the tolerance on the test value to the error of the bilinear interpolation.

## 2026-09-24 Gating 

Used for: Code genereation as well as help imporving the filter idea, I new in genereal what I wanted to filter out like values that are physically impossible to map like off throttle response or impossible values, however Claude improved on my ideas by adding the transients response which was one I did not think about as well as helping me flesh out my ideas of the filter.

How this was verified: The concepts intuitvly make sense, if the operating point moves past where the sensor is still at than the point belongs in the next table along. Furthermore all inputs data points were verified to make sure they were realistic against known values.

Limitations: The first implimetation of the code had regected 93% of the sample points dude to the MAP sensor which has an error of 0.6 KPA at an interval of 0.02s per reading, meaning a 21 kPa/s spurious rate which exceeds the limit failing the test, however the engine was never actually changing that fast it was just the error causing it to jump.

My decision: Instead of sampling each point individually, I took a sample of the surrounding 10 points and then found the rate of change of that, this smoothed over a large percentage of the sensor noise and after it had an acceptace rate of 88% which seemed more manageable.

## 2026-09-24 Surface fitting 

Used for: Fitting workshop and prac methods to use case

How this was verified: The techniques were all taught in 3001, Claude helped me transpose them into my code however I still verified everything against the textbook

Limitations: A test comparing bilinear against nearest-node failed on max error because
of a corner boundary effect

My decision: split into an interior-RMSE test plus `test_bilinear_is_one_sided_at_table_edges`.


## 2026-09-24 Convergence 

Used for: Suggested the idea of a dampening factor to dampen out noise which was able to be checked against the theoretical decay of (1-k)^n, code generation and help.

How this was verified: The dampening ratio was taught in MMA2005, it made compelate sense to use it to dampen the noise form the system, Ai helped me impalent this.

Limitations: The iteration converges, but not to the true table. Over 12 passes with bilinear fitting, held-out AFR error falls from 10.26% to 0.67% while table RMSE only moves from 7.18 to 5.60 and then sits flat. Because the ECU reads the table by interpolation, the scheme settles on whichever table makes the interpolated value correct, which differs from the truth by the interpolation error. The table error cannot reach zero no matter how many passes are run. The stopping criterion therefore uses held-out error rather than the size of the table update, since the update keeps shrinking after accuracy has stopped improving.


My decision: Use a damping factor of 0.6 and stop on held out error.

## 2026-09-24 Pulse width, performance and transport delay

Used for: Code generation for the injector pulse width output, the profiling and FLOP estimates, and the exhaust transport delay model. I specified the injector and engine parameters from my own bike.

How this was verified: Pulse widths were checked against hand calculations at reference conditions. The timing results were run several times to confirm they were stable, and the arithmetic cost estimates were compared against the measured times.

Limitations: the transport delay rounds to whole samples which is 20 ms at 50 Hz, that is a failry bad assumptioncompared to the delay at high rpm when  the engine is spinning really really fast

My decision: For a large majority of the use case, the numbers that the  Ai gave are still valid, and when the values are not as valid, the error  is likley to not be significant enough to cause massive issues.

## 2026-09-25 Sensitivity analysis

Used for: Code to sweep each sensor imperfection in turn and rank the effect on accuracy.

How this was verified: Ran the sweep myself and read both columns. A 5 kPa manifold pressure bias costs +1.15 points of AFR accuracy, far more than any random noise. Slowing the AFR sensor to 0.50 s does less to AFR error (+0.54) but much more to the table (RMSE 4.42 against a 1.15 baseline).

## 2026-09-25 Filtering

Used for: Code for three filters on the AFR channel — moving average, Savitzky-Golay and a hand-written Kalman filter — and a phase lag measurement to compare them.

How this was verified: Compared all four options on the same gated data. Table RMSE was 6.37 for every one of them, and AFR error went very slightly worse (3.83% unfiltered against 3.84-3.85% filtered). The Kalman filter increased the residual noise rather than reducing it.

Limitations: Filtering turned out to be unnecessary here. Per-cell averaging already reduces noise by root-n and gating removes the samples where a noisy reading would be misattributed, so there is little left for a filter to do.

My decision: Keep the filter, it added an extra layer of protection to the system, this was a case of the Ai trying to overdo itself

## 2026-10-03 Robustness and repeatability study

Used for: Code for a study asking how far the pipeline can be trusted when its inputs change through repeat runs, different noise seeds, different engines, logging rate, table resolution, and logs with poor coverage.

How this was verified: Ran each section myself and read the results against what I expected. Repeat runs are bit-for-bit identical. Across ten noise seeds the spread is about 8% coefficient of variation, small next to the improvement the calibration makes. Twenty randomised engines, with new true surfaces and new starting maps each time, all twenty improved, worst final AFR error 0.98%. Logging rate is flat from 10 to 200 Hz. Table resolution has an optimum near 16 x 12, which is why the default grid is that size chosen on this evidence rather than by convention.

Unexpected result which Ai helped me find: One case makes the table worse  than doing nothing. A cruise only log reaches only 5% of cells and the Gaussian process extrapolates across the rest from that one small island. A 34-second log covering idle, cruise and a wide-open pull beats 100 seconds of steady cruising by a wide margin, so variety in the log matters far more than length.

My decision: Keep that result, the finding by Ai helped show limitations in my method.

## 2026-10-03 Sensor health screening

Used for: My idea was a detection algorithm that finds sensor faults in the data, fixes the data, then reports what was wrong. Claude argued against the fixing part: repairing requires assuming the fault's form and leaves no trace in the output, so a table built from repaired data looks exactly as trustworthy as one built from good data. The design became detect, report and refuse instead. Claude then wrote the module and its tests.

How this was verified: Ran the checks against healthy logs across several noise seeds and both drive cycles, and against logs with faults I injected deliberately. 342 tests pass with full coverage of the new code.

Limitations: The first version of the stuck sensor check flagged the throttle and battery voltage channels on a perfectly healthy log, because it compared each channel against engine speed. That reference was wrong,  engine speed follows the throttle rather than the other way round, and battery voltage is held near constant by the alternator. Two further rounds of testing were needed before it stopped firing on good data. A sluggish AFR sensor still cannot be detected at all: the cross-correlation test moved only 0.06 s for a fourfold change in sensor time constant, against a baseline of 0.46 s that depends on the drive cycle, and an exhaust transport delay shifts it by the same amount. That was documented as a limitation rather than shipped as a weak detector.

My decision: Even with the issues with the sensors, the results were  still valid enough to justify having the sensor health module there. 

## 2026-10-03 Sensor fault demonstration

Used for: Code to inject eleven named faults into a log, each paired with the verdict the screening should reach, and a notebook comparing a healthy and a faulty dataset from the same drive cycle.

How this was verified: Ran the full calibration on every faulty log and compared what the screening said against what the fault actually cost.

Limitations: The result contradicted what the section set out to show. Severity does not predict damage — only two of the nine detected faults harm the table at all. Gating already discards implausible readings, and air density cancels in AFR_measured / AFR_target = VE_true / VE_table, so a temperature sensor fault cannot reach the VE correction. I checked this by confirming the correction factors are identical to the last decimal with a dead intake temperature sensor. What the screening adds is therefore naming the broken sensor, not protecting the table.

My decision: It was still useful useful information worth having, however  the main part I wanted to have in the table was the naming of the broken  sensor so I am still happy with the results

## 2026-09-22 to 10-03 Results table

Used for: Reviewing the README before submission. Claude flagged the iterated row of the results table as wrong, saying 5.60 should be 1.18. Its reasoning was that 5.60 matched the bilinear figure in the table directly below, and that iteration cannot make the result worse than a single pass.

How this was verified: I asked where the number had come from rather than changing it. Reproducing notebook 05 showed 5.60 is correct — that notebook iterates with method="bilinear", not the Gaussian process, and gives exactly 5.60 at every gain. Claude had assumed the wrong method and was wrong twice before being checked.

Limitations: There was a real problem, but not the one flagged. The table compares a single-pass Gaussian process result against an iterated bilinear result without naming either method, so it reads as though iteration made things worse. The project's best result, iterated Gaussian process at 1.18, is not in the table at all.

My decision: The results made sense and the mistake was corrected within the readme 