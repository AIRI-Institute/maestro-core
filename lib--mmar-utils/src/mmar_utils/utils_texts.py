def pretty_line(text: str, cut_count: int = 100) -> str:
    if len(text) > 100:
        text_cut = text[:cut_count]
        size = len(text)
        text_pretty = f"{text_cut}..(total {size} characters)"
    else:
        text_pretty = text
    text_pretty = text_pretty.replace("\n", "\\n")
    return text_pretty


def remove_prefix_if_present(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    else:
        return text


def remove_suffix_if_present(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    else:
        return text


def rindex_safe(text, sub, end) -> int | None:
    try:
        return text.rindex(sub, 0, end)
    except ValueError:
        return None


def chunk_respect_semantic(text: str, max_chunk_size: int) -> list[str]:
    """
    This function tries to chunk respecting
    - sections ( starts with '#' )
    - paragraphs
    - lines
    - words
    This function is slow: assumed that `len(text) / max_chunk_size` is small, e.g. < 10
    """
    text = text.strip()

    if len(text) < max_chunk_size:
        return [text]

    split_separators = [
        "\n#",
        "\n\n",
        "\n",
        " ",
    ]

    for sep in split_separators:
        pos = rindex_safe(text, sep, max_chunk_size)
        if pos is None:
            continue
        parts = [text[:pos], text[pos:]]
        res = [pt for pts in parts for pt in chunk_respect_semantic(pts, max_chunk_size)]
        res = [pt for pt in res if pt]
        return res

    res = [
        text[:max_chunk_size].strip(),
        *chunk_respect_semantic(text[max_chunk_size:], max_chunk_size),
    ]
    res = [pt for pt in res if pt]
    return res


def extract_text_inside(text: str, start_marker: str, end_marker: str) -> str | None:
    start_pos = text.index(start_marker)
    end_pos = text.index(end_marker, start_pos)
    res = text[start_pos + len(start_marker) : end_pos].strip()
    return res
