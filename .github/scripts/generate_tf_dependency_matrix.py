import json
from argparse import ArgumentParser


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--requirements",
        default="requirements.txt",
        help="Path to requirements.txt file.",
    )
    return parser.parse_args()


def main():
    """Print Tensorflow version matrix to test."""
    args = _parse_arguments()

    content = _read_file(args.requirements)
    tf_requirement = _get_tf_requirement(content)

    for r in (">=", ",<"):
        tf_requirement = tf_requirement.replace(r, ",")
    _, lower, upper = tf_requirement.split(",")

    lower_minor = int(lower.split(".")[-1])
    upper_minor = int(upper.split(".")[-1])

    # Skip versions below 2.6 - they seem to be unsupported
    lower_minor = 6 if lower_minor < 6 else lower_minor
    matrix = [f"2.{minor}" for minor in range(lower_minor, upper_minor)]

    print(json.dumps(matrix))


def _read_file(path):
    with open(path, "r") as f:
        content = f.readlines()
    return content


def _get_tf_requirement(content):
    requirement = [x for x in content if x.startswith("tensorflow>=")][0]
    if requirement.endswith("\n"):
        requirement = requirement.rstrip("\n")
    return requirement


if __name__ == "__main__":
    main()
