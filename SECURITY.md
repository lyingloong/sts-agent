# Reporting a security issue

Use [GitHub private vulnerability reporting](https://github.com/lyingloong/sts-agent/security/advisories/new) for issues that could expose credentials or private data, or allow unintended control of the game or its host.

Include the affected commit, platform, mode, observed impact and a minimal reproduction using synthetic data. Remove API tokens, private hostnames, usernames and local paths from excerpts. Configuration files, complete session logs and model request traces can contain account or gameplay data.

If a working credential has been exposed, revoke it with the issuing provider. A report can describe where the credential appeared without including its value.

For installation questions, recording problems and reproducible non-security bugs, use [GitHub Issues](https://github.com/lyingloong/sts-agent/issues). For a problem specific to the companion Java mod, use the [ActionRecorder repository](https://github.com/lyingloong/sts-action-recorder).
