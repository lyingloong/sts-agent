"""Inspect named runtime profiles without starting the game or a model."""

from __future__ import annotations

import argparse
import sys

from sts_agent.config import ConfigError, config_path, load_named_agent_config
from sts_agent.core.modes import MODE_FRAMEWORKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to the local config, relative to the project or absolute")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="list supported profile names")
    selection.add_argument("--mode", choices=sorted(MODE_FRAMEWORKS), help="check a profile without switching to it")
    selection.add_argument("--all", action="store_true", help="check the active config and every other profile")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(sorted(MODE_FRAMEWORKS)))
        return 0

    path = config_path(config_file=args.config)
    print(f"Config: {path}")
    profiles = [args.mode]
    if args.all:
        profiles.extend(sorted(MODE_FRAMEWORKS))
    checked = set()
    failed = False
    for requested in profiles:
        if requested in checked:
            continue
        try:
            name, config = load_named_agent_config(config_file=path, mode=requested)
        except (ConfigError, OSError, UnicodeError) as exc:
            print(f"{requested or 'active profile'}: {exc}", file=sys.stderr)
            failed = True
            continue
        checked.add(name)
        framework = config["framework"]
        print(f"{name}: control={framework['control_mode']}  backend={framework['policy_backend']}  "
              f"learning={framework['learning_method']}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
