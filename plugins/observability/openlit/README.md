# OpenLIT Plugin

This folder runs a local OpenLIT stack for OpenTulpa observability.

## Services

- OpenLIT UI/API: `http://127.0.0.1:3000`
- OTLP HTTP ingest: `http://127.0.0.1:4318`
- ClickHouse backend (local): `127.0.0.1:8123`

## Start manually

```bash
cd plugins/observability/openlit
cp .env.example .env
docker compose up -d
```

## Stop manually

```bash
cd plugins/observability/openlit
docker compose down
```

## App integration

Set in the main project `.env`:

```dotenv
OPENLIT_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=opentulpa
OTEL_DEPLOYMENT_ENVIRONMENT=development
OPENLIT_APPLICATION_NAME=opentulpa
```

When enabled, the app initializes OpenLIT in-process at startup.

Install the Python SDK in your project environment:

```bash
uv pip install openlit
```
