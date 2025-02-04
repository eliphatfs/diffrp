import argparse
from .quick_render import build_argparse


def main():
    argp = argparse.ArgumentParser(prog='diffrp', description='diffrp CLI entrypoint.')
    subparsers = argp.add_subparsers(help='available diffrp scripts.', required=True)
    build_argparse(subparsers.add_parser("quick-render", help="Quickly render previews with the deferred rasterization PBR pipeline for 3D models."))
    args = argp.parse_args()
    args.entrypoint(args)


if __name__ == '__main__':
    main()
