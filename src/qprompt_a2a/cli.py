import argparse
import sys

from qprompt_langgraph.ir.loader import IRLoadError, IRVersionError, load_ir

from qprompt_a2a.codegen.generator import write_project


def main() -> None:
    parser = argparse.ArgumentParser(prog="qprompt-a2a")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser(
        "render", help="render a qprompt IR JSON file to an A2A-services + Docker Compose project"
    )
    render.add_argument("ir_file", help="path to a *.ir.json file produced by `qprompt-cli generate --format ir`")
    render.add_argument("workflow_name", help="name of the workflow (within the IR) to render")
    render.add_argument("-o", "--out", default="generated", help="output directory (default: ./generated)")

    args = parser.parse_args()

    if args.command == "render":
        try:
            ir = load_ir(args.ir_file)
        except (IRVersionError, IRLoadError) as e:
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(1) from e

        written = write_project(ir, args.workflow_name, args.out)
        for path in written:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
