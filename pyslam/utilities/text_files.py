"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

# Text files written by a user's shell, e.g. the associations file of a TUM sequence made with
# `python scripts/associate.py rgb.txt depth.txt > associations.txt`: in Windows PowerShell 5.1, `>`
# writes UTF-16 with a byte order mark, which a plain open() cannot decode.

import codecs


def text_encoding(path):
    """The encoding of a text file: UTF-16 or UTF-8 when it starts with the byte order mark of one, UTF-8
    otherwise."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"  # the codec reads the byte order mark and drops it
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    return "utf-8"


def open_text(path):
    """open(path) for reading, in the encoding of the file (see text_encoding())."""
    return open(path, "r", encoding=text_encoding(path))
