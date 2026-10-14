# Atlas local development

## Starting locally
Install Docker and Docker Compose. Copy the example configuration to a local environment file and obtain development credentials through the approved secrets manager. Run `docker compose up --build` from the service repository.

## Health check
The Atlas service exposes `/health`. An HTTP 200 response indicates the application is alive; it does not guarantee that every downstream dependency is healthy.

## Support ownership
The platform team owns deployment tooling. The database team owns connection-limit changes. The on-call engineer coordinates production incidents. This documentation does not specify individual employee contact details.
