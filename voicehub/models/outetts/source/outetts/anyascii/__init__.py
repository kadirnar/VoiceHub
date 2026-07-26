# ISC License

# Copyright (c) 2020-2023, Hunter WB <hunterwb.com>

# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Unicode to ASCII transliteration"""

from sys import intern
from zlib import MAX_WBITS, decompress
from pathlib import Path

def read_binary(resource):
    data_dir = Path(__file__).parent / "_data"
    return (data_dir / resource).read_bytes()

_blocks = {}


def anyascii(string):
    # type: (str) -> str
    """Transliterate a string into ASCII."""
    try:
        if string.isascii():
            return string
    except AttributeError:
        pass
    result = []
    for char in string:
        codepoint = ord(char)
        if codepoint <= 0x7F:
            result.append(char)
            continue
        blocknum = codepoint >> 8
        lo = codepoint & 0xFF
        try:
            block = _blocks[blocknum]
        except KeyError:
            try:
                b = read_binary("%03x" % blocknum)
                s = decompress(b, -MAX_WBITS).decode("ascii")
                block = tuple(map(intern, s.split("\t")))
            except FileNotFoundError:
                block = ()
            _blocks[blocknum] = block
        if len(block) > lo:
            result.append(block[lo])
    return "".join(result)
