import re
import unicodedata


def fix_unicode_symbols(text: str) -> str:
    """
    Universal fix for various Unicode escape sequences and malformed Unicode
    """

    def replace_unicode_escape(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except ValueError:
            return match.group(0)  # Return original if conversion fails

    # Fix various Unicode escape formats
    patterns = [
        (r"/uni([0-9A-Fa-f]{4})", replace_unicode_escape),  # /uniXXXX
        (r"\\u([0-9A-Fa-f]{4})", replace_unicode_escape),  # \uXXXX
        (r"&#x([0-9A-Fa-f]+);", replace_unicode_escape),  # &#xXXXX;
        (r"&#([0-9]+);", lambda m: chr(int(m.group(1)))),  # &#DDDD;
        (r"%u([0-9A-Fa-f]{4})", replace_unicode_escape),  # %uXXXX
        (r"U\+([0-9A-Fa-f]{4,6})", replace_unicode_escape),  # U+XXXXXX
    ]

    for pattern, replacer in patterns:
        text = re.sub(pattern, replacer, text)

    # Normalize Unicode (decomposed to composed forms)
    text = unicodedata.normalize("NFC", text)

    # Fix common encoding issues
    try:
        # Try to fix double-encoded UTF-8
        if text.encode("latin1").decode("utf-8") != text:
            text = text.encode("latin1").decode("utf-8")
    except (UnicodeError, UnicodeDecodeError):
        pass

    return text


def clean_and_fix_text(text: str) -> str:
    """
    Complete text cleaning: fix Unicode + normalize whitespace + remove artifacts
    """
    # Fix Unicode issues
    text = fix_unicode_symbols(text)

    # Remove or replace common text artifacts
    artifacts = [
        (r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", ""),  # Control characters
        (r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\<\>\|\~\`]", ""),  # Keep only printable
        (r"\ufeff", ""),  # BOM
        (r"â€™", "'"),  # Smart quote artifacts
        (r"â€œ", '"'),  # Left double quote
        (r"â€\x9d", '"'),  # Right double quote
        (r'â€"', "—"),  # Em dash
    ]

    for pattern, replacement in artifacts:
        text = re.sub(pattern, replacement, text)

    # Normalize whitespace BUT preserve newlines
    text = re.sub(r"[ \t]+", " ", text)  # Replace multiple spaces/tabs with single space
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Preserve paragraph breaks
    text = re.sub(r"[ \t]+\n", "\n", text)  # Remove trailing spaces before newlines
    text = re.sub(r"\n[ \t]+", "\n", text)  # Remove leading spaces after newlines

    return text.strip()


def remove_chars(text: str, chars: str):
    chars_trans: dict[int, int | None] = str.maketrans("", "", chars)
    text = text.translate(chars_trans)
    return text


def postprocess_text(text: str) -> str:
    text = remove_chars(text, "|[]").strip()
    text = re.sub(" +", " ", text)  # reduce many spaces to one
    text = re.sub(r"\s*\n\s*", "\n", text)  # reduce many newlines to one
    return text
