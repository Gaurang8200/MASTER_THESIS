---
name: engineering
description: Apply senior software, AI, ML, and robotics engineering standards to this repository. Use for every implementation, refactor, bug fix, review, test change, model integration, perception change, language pipeline change, safety change, or robot interface change.
---

# Engineering

Build production quality Python systems for multimodal human robot interaction. Keep the implementation modular, typed, testable, observable, and safe.

## Start with the contract

1. Read the issue, acceptance criteria, current tests, and relevant call path.
2. Trace data from sensor input through perception, language, fusion, safety, and robot output.
3. Define inputs, outputs, invariants, failure modes, latency limits, and ownership before editing.
4. Confirm units, coordinate frames, timestamps, confidence meaning, and optional values at every boundary.
5. Implement the smallest complete vertical slice that satisfies the contract.

## Organize the code

1. Keep `Code/app` responsible for composition, shared contracts, configuration, orchestration, safety, and robot boundaries.
2. Keep `Code/voice`, `Code/gesture`, and `Code/nlp` responsible for their modality adapters and domain logic.
3. Keep tests under `Code/tests` and mirror the behavior being tested.
4. Introduce domain, service, and adapter packages only when growth creates a clear dependency or ownership problem.
5. Keep domain contracts independent from hardware, model runtimes, web frameworks, and storage clients.
6. Make orchestration depend on interfaces and make adapters implement those interfaces.
7. Split a file when it owns more than one reason to change. Review files as they approach 250 lines.
8. Keep one abstraction level inside each method.

## Choose classes and functions deliberately

1. Use a class when behavior owns resources, dependencies, state transitions, caching, or invariants.
2. Use a pure function for stateless transformation, validation, scoring, and policy decisions.
3. Keep constructors cheap and free from blocking model or hardware startup when possible.
4. Expose explicit start, close, and health methods for cameras, microphones, models, network clients, and robots.
5. Inject dependencies through constructors or function parameters.
6. Use `Protocol` or an abstract interface when multiple adapters or test doubles need the same contract.
7. Avoid global mutable state, hidden singletons, utility classes, and classes that only wrap one unrelated function.
8. Keep public names precise and keep private helpers close to their owner.

## Model data explicitly

1. Add complete type hints to public functions and methods.
2. Use Pydantic models at validation and transport boundaries.
3. Use dataclasses for lightweight internal state when validation is not required.
4. Use enums or constrained literals for closed command sets and safety states.
5. Represent units, frames, timestamps, model version, source, and confidence semantics explicitly.
6. Reject invalid or incomplete commands at the boundary that first knows they are invalid.
7. Avoid untyped dictionaries for stable domain contracts.

## Protect the robot

1. Fail closed when command validity, signal freshness, system health, or confidence is uncertain.
2. Give stop commands and emergency signals priority over all planned actions.
3. Validate every robot command immediately before transport.
4. Separate command policy from robot communication so each can be tested independently.
5. Use bounded timeouts, retries, queues, and motion limits.
6. Prevent stale inference results from controlling current motion.
7. Make repeated commands safe or attach an idempotency mechanism.
8. Record the reason for every rejected, modified, or stopped command.

## Engineer AI and ML components

1. Separate model loading, preprocessing, inference, postprocessing, and confidence calibration.
2. Load models through configuration and record model identity and version.
3. Make device selection, precision, thresholds, and runtime options explicit.
4. Keep model loading out of module import side effects.
5. Define deterministic behavior for tests through seeds, fixtures, and stable sample data.
6. Measure latency and memory at expensive boundaries.
7. Treat model scores according to their real semantics and do not present arbitrary scores as calibrated probability.
8. Provide a safe fallback for missing models, unavailable devices, malformed inputs, and low confidence output.

## Handle concurrency and resources

1. Keep blocking capture, inference, and robot communication away from the async event loop.
2. Use workers, executors, or queues when blocking libraries cannot be avoided.
3. Bound queue size and define overload behavior.
4. Release cameras, audio streams, model sessions, files, and sockets on success, failure, and cancellation.
5. Propagate cancellation and preserve the original error context.
6. Use monotonic time for durations and deadlines.

## Build observability

1. Emit structured events for lifecycle changes, inference latency, command decisions, safety rejection, transport result, and degraded operation.
2. Include stable event names and useful correlation identifiers.
3. Avoid logging raw audio, images, credentials, personal data, or secrets unless the issue explicitly requires a protected diagnostic path.
4. Use metrics for rates and latency, logs for events, and traces for cross component flow.
5. Make health checks report dependency state rather than only process availability.

## Test the behavior

1. Add unit tests for pure policy, parsing, fusion, and safety logic.
2. Add contract tests for model and hardware adapters.
3. Add integration tests for the multimodal pipeline with controlled fixtures.
4. Use simulated or fake hardware by default. Never require a real robot for the normal test suite.
5. Add regression tests for bugs and safety failures.
6. Cover cancellation, timeouts, stale data, low confidence, missing resources, malformed input, and transport failure.
7. Keep tests deterministic and assert behavior rather than internal implementation details.
8. Run targeted tests first and the relevant full suite after the change.

## Keep implementation disciplined

1. Preserve stable public behavior unless the issue approves a change.
2. Avoid unrelated refactors inside a focused issue.
3. Remove dead code and completed placeholders in the touched path.
4. Do not add a dependency when the standard library or an existing dependency is sufficient.
5. Keep comments focused on why, safety, invariants, and nonobvious constraints.
6. Follow the punctuation rules in `AGENTS.md` for every comment and description.
7. Review the final diff for architecture, correctness, safety, tests, performance, and accidental files.
8. Finish only when acceptance criteria and validation pass.
