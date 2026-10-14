# Atlas troubleshooting runbook

## API 503 errors
Check the service health endpoint, then inspect dependency availability and the recent deployment history. If a dependency is unavailable, contact its owning team. Do not repeatedly restart the service without investigating the dependency failure.

## Database connection exhaustion
Inspect active database connections and pool utilization. The documented connection pool limit is 20 connections per service instance. Do not increase this limit without database-team approval.

## Logs and sensitive data
Use the centralized logging dashboard and filter by request correlation ID. Never paste access tokens, passwords, customer personal data, or private keys into incident comments or chatbot questions.
