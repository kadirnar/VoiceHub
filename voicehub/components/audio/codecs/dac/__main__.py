"""Command-line entry point for VoiceHub's native DAC utilities."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from voicehub.components.audio.codecs.dac.utils import download
from voicehub.components.audio.codecs.dac.utils.decode import decode
from voicehub.components.audio.codecs.dac.utils.encode import encode


def _common_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--model-tag", default="latest")
    parser.add_argument("--model-bitrate", default="8kbps")
    parser.add_argument("--model-type", default="44khz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voicehub.components.audio.codecs.dac",
        description="Encode and decode audio with VoiceHub's native DAC.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode_parser = commands.add_parser("encode")
    encode_parser.add_argument("input")
    encode_parser.add_argument("--output", default="")
    encode_parser.add_argument("--n-quantizers", type=int)
    encode_parser.add_argument("--win-duration", type=float, default=5.0)
    _common_model_options(encode_parser)

    decode_parser = commands.add_parser("decode")
    decode_parser.add_argument("input")
    decode_parser.add_argument("--output", default="")
    _common_model_options(decode_parser)

    download_parser = commands.add_parser("download")
    download_parser.add_argument("--model-type", default="44khz")
    download_parser.add_argument("--model-bitrate", default="8kbps")
    download_parser.add_argument("--model-tag", default="latest")
    return parser


def main(argv: Sequence[str] | None = None):
    arguments = vars(build_parser().parse_args(argv))
    command = arguments.pop("command")
    if command == "encode":
        return encode(**arguments)
    if command == "decode":
        return decode(**arguments)
    return download(
        model_type=arguments["model_type"],
        model_bitrate=arguments["model_bitrate"],
        tag=arguments["model_tag"],
    )


if __name__ == "__main__":
    main()
