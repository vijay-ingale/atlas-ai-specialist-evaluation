# Atlas engineering deployment guide

These procedures describe a fictional assessment service, Atlas. They are not instructions for a real production environment.

## Pre-deployment requirements
Every production release requires a passing CI pipeline and approval from the on-call engineer. Deploy to staging first and verify the health endpoint before requesting production approval.

## Production rollout
Use the approved CI release workflow to deploy a versioned container image. Direct production deployment from a developer laptop is prohibited. Observe error rate and request latency for 15 minutes after rollout.

## Rollback
If the HTTP 5xx error rate exceeds 2% for five consecutive minutes after a release, roll back through the CI workflow to the previous approved image. Notify the on-call engineer and attach the release identifier to the incident record.
