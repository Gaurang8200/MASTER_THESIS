# Project Instructions

## Delivery priority

1. Spend most effort on implementation, validation, and integration.
2. Prefer working code over long explanations.
3. Complete the requested vertical slice whenever the repository and access allow it.
4. Keep status updates concise and reserve detail for decisions, risks, and results.
5. Do not leave placeholders when a real implementation is within scope.

## Required engineering skill

1. Read `.agents/skills/engineering/SKILL.md` before every implementation, refactor, bug fix, code review, test change, AI change, ML change, or robotics change.
2. Apply that skill as the repository coding standard.
3. Resolve any conflict in favor of safety, correctness, user instructions, and this file.

## Engineering judgment

1. Work with the judgment expected from a senior AI engineer, ML engineer, software engineer, and robotics engineer.
2. Consider architecture, reliability, safety, latency, resource ownership, observability, testing, deployment, and maintenance before implementation.
3. Identify the root cause before fixing a bug.
4. Prefer the smallest complete design that supports the next realistic extension.
5. Avoid speculative abstractions, unnecessary dependencies, and broad rewrites.

## GitHub delivery workflow

1. Create or reuse one GitHub issue before changing code.
2. Assign the issue to `Gaurang8200`.
3. Add the issue to the `Master Thesis` project with status `Backlog`.
4. Record the plan and acceptance criteria in the issue.
5. Ask the user before creating a branch.
6. Use one branch for one coherent issue and include the branch name in the issue.
7. Group closely related small changes under one issue and branch when that keeps the history clear.
8. Create a dedicated issue and approved branch for a bug when it is not already covered by active work.
9. Set the project status to `In progress` when implementation begins.
10. Make changes locally before committing or pushing.
11. Stage only files that belong to the issue.
12. Use a concise human commit message without artificial wording.
13. Push the approved branch and create a pull request that links the issue with `Closes #<issue>`.
14. Include the change, reason, impact, and validation in the pull request.
15. Set the project status to `In review` after opening the pull request.
16. Merge only after relevant checks pass and the diff matches the issue scope.
17. Set the project status to `Done` after the merge and confirm that the issue is closed.

## Bug workflow

1. Capture expected behavior, actual behavior, reproduction evidence, impact, and root cause in the issue.
2. Add a regression test that fails for the reported defect when practical.
3. Fix the root cause rather than hiding the symptom.
4. Validate adjacent safety and integration paths.

## Writing style

1. Do not use dash punctuation or semicolons in prose, descriptions, documentation, commit messages, issue text, pull request text, or code comments.
2. Permit a hyphen only when required by language syntax, command flags, immutable external names, URLs, dependency names, paths, generated metadata, or YAML delimiters.
3. Prefer underscore characters in new branch names and identifiers when the language permits them.
4. Keep comments rare and natural.
5. Use comments to explain intent, constraints, safety decisions, or nonobvious reasoning.
6. Do not narrate code that already explains itself.

## Change discipline

1. Inspect the current branch, status, tests, and relevant modules before editing.
2. Preserve unrelated user changes.
3. Keep public contracts backward compatible unless the issue approves a breaking change.
4. Add migrations or compatibility handling when stored data or interfaces change.
5. Never expose credentials, personal data, model secrets, or robot access details.
6. Prefer configuration over embedded environment values.

## Completion standard

1. Satisfy every acceptance criterion.
2. Run targeted tests first and the relevant full suite second.
3. Validate types, formatting, imports, and configuration when tools exist.
4. Review the final diff for correctness, safety, scope, and accidental files.
5. Report completed implementation, validation results, issue, branch, commit, pull request, and merge state concisely.
