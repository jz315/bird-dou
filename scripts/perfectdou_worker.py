"""Python-3.7/Linux JSONL worker for the official PerfectDou binary release."""

import argparse
import json
import os
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    source = os.path.abspath(arguments.source)
    sys.path.insert(0, source)
    from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent

    agents = {role: PerfectDouAgent(role) for role in ("landlord", "landlord_down", "landlord_up")}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("protocol_version") != 1:
                raise ValueError("unsupported protocol version")
            role = request["role"]
            raw = request["public_infoset"]
            infoset = SimpleNamespace(**raw)
            action = agents[role].act(infoset)
            response = {"action": action}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
