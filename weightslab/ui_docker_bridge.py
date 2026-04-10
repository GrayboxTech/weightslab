import argparse
import os
import shutil
import subprocess
import sys
from importlib.resources import files


def _get_compose_file():
    """Return the path to the bundled docker-compose.yml."""
    return files("weightslab.ui") / "docker-compose.yml"


def _get_envoy_config():
    """Return the path to the bundled envoy.yaml."""
    return files("weightslab.ui") / "envoy.yaml"


def _check_docker():
    """Verify that docker is installed and the daemon is running."""
    if shutil.which("docker") is None:
        print(
            "Error: Docker is required but not found on your PATH.\n"
            "Install it from: https://docs.docker.com/get-docker/",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(
            "Error: Docker is installed but the daemon is not running.\n"
            "Start it with:  sudo systemctl start docker\n"
            "                or open Docker Desktop.",
            file=sys.stderr,
        )
        sys.exit(1)


def _compose_cmd(compose_file, envoy_config, action):
    """Build and run a docker compose command."""
    env = os.environ.copy()
    env["WS_ENVOY_CONFIG"] = str(envoy_config)

    cmd = ["docker", "compose", "-f", str(compose_file)] + action
    subprocess.run(cmd, env=env, check=True)


def ui_launch(args):
    """Pull images and start UI containers."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["up", "-d", "--pull", "always"],
    )
    port = os.environ.get("VITE_PORT", "5173")
    print(f"\nWeights Studio UI is running at: http://localhost:{port}")


def ui_stop(args):
    """Stop UI containers (keeps images)."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["stop"],
    )
    print("Weights Studio UI stopped.")


def ui_drop(args):
    """Stop and remove containers, networks, and images."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["down", "--rmi", "all"],
    )
    print("Weights Studio UI containers and images removed.")


def main():
    parser = argparse.ArgumentParser(
        prog="weightslab",
        description="WeightsLab CLI",
    )
    sub = parser.add_subparsers(dest="command")

    ui_parser = sub.add_parser("ui", help="Manage the Weights Studio UI")
    ui_sub = ui_parser.add_subparsers(dest="action")

    ui_sub.add_parser("launch", help="Pull images and start the UI")
    ui_sub.add_parser("stop", help="Stop the UI containers (keeps images)")
    ui_sub.add_parser("drop", help="Stop containers and remove images")

    sub.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    actions = {
        "launch": ui_launch,
        "stop": ui_stop,
        "drop": ui_drop,
    }

    if args.command == "help" or args.command is None:
        parser.print_help()
    elif args.command == "ui" and args.action in actions:
        actions[args.action](args)
    elif args.command == "ui":
        ui_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
