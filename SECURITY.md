# Security Policy

## Supported Versions

vAquila is currently in public beta.

The project team provides security fixes on a best-effort basis for:

- `main` (latest development branch)
- Latest beta tag series (for example: `v0.1.0-beta.*`)

Older branches and experimental forks are not guaranteed to receive patches.

## Reporting a Vulnerability

Please do not open public issues for undisclosed vulnerabilities.

Report security issues privately using one of these channels:

- GitHub Security Advisories (preferred): `Security` tab -> `Report a vulnerability`
- Direct maintainer contact (if advisory flow is unavailable)

Include as much detail as possible:

- Affected component/file
- Reproduction steps or proof of concept
- Impact assessment
- Suggested remediation (if available)

## Response Targets

Best-effort response targets:

- Initial acknowledgment: within 72 hours
- Triage decision: within 7 days
- Patch or mitigation timeline: communicated after triage

## Disclosure Process

- We validate and triage the report
- We prepare a fix and tests
- We coordinate disclosure timing with the reporter when possible
- We publish a changelog note and/or advisory after a fix is available

## Scope Notes

- Runtime security includes CLI, Web UI, container orchestration, and build/deploy workflows.
- Third-party dependencies (Python/Node/Docker images) are handled by upstream advisories and periodic update cycles.
