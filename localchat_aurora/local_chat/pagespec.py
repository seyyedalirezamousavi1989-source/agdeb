"""Page selection for big PDFs: "only 10 pages", "pages 12-30", "صفحهٔ ۵", "آخرین ۴ صفحه".

A selection is a small JSON-able dict ``{"ranges": [[first, last], ...], "last": n}`` (explicit ranges plus «the last n pages»).
It comes from the tasks form (``parse_list``) or from a chat sentence (``parse_request``) and is resolved against the real
page count with ``resolve``. Page numbers are PDF page indexes (1 = first page of the file), not the numbers printed in the book.
"""
import re
import unicodedata

MAX_SELECTED_PAGES = 2000

_WORDS = {"یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "یازده": 11,
          "دوازده": 12, "سیزده": 13, "چهارده": 14, "پانزده": 15, "شانزده": 16, "هفده": 17, "هجده": 18, "نوزده": 19,
          "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
          "صد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400, "پانصد": 500}
_WORD_ALT = "|".join(sorted(_WORDS, key=len, reverse=True))
_WORD_RUN = re.compile(rf"(?<!\w)(?:{_WORD_ALT})(?:\s+و\s+(?:{_WORD_ALT}))*(?!\w)")
_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})

_PG = r"(?:صفحه|صفحات|pages?|pp?\.?)(?:\s+(?:های|ی))?"
_SEP = r"\s*(?:تا|الی|-|–|—|to|through|thru|until)\s*"
_N = r"(\d{1,5})"


class PageSpecError(ValueError):
    pass


def fold(text):
    """Letters/digits/ZWNJ normalised and Persian number words («ده», «بیست و پنج») turned into digits."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ٔ", "").replace("‌", " ").translate(_DIGITS).lower()
    return _WORD_RUN.sub(lambda m: str(sum(_WORDS.get(w, 0) for w in re.split(r"\s+و\s+|\s+", m.group(0)))), text)


def _clean(spec):
    ranges = []
    for pair in spec.get("ranges") or []:
        first, last = int(pair[0]), int(pair[1])
        if first > last:
            first, last = last, first
        if first < 1:
            raise PageSpecError("شمارهٔ صفحه باید از ۱ شروع شود.")
        ranges.append([first, last])
    tail = max(0, int(spec.get("last") or 0))
    if not ranges and not tail:
        return None
    return {"ranges": sorted(ranges), "last": tail}


def parse_request(text):
    """Selection mentioned in a sentence, or None when the sentence does not restrict pages."""
    t = fold(text)
    state = {"ranges": [], "last": 0}

    def cut(pattern, handler):
        nonlocal t

        def sub(match):
            handler(match)
            return " "
        t = re.sub(pattern, sub, t)

    def add_range(m):
        state["ranges"].append([int(m.group(1)), int(m.group(2))])

    def add_first(m):
        state["ranges"].append([1, int(m.group(1))])

    def add_last(m):
        state["last"] = max(state["last"], int(m.group(1)))

    # «from the first page to the last page» / «از صفحه اول تا آخر» names the WHOLE document: consume it, restrict nothing.
    cut(rf"(?:from\s+|از\s+)?(?:the\s+)?(?:first|اولین|نخستین)?\s*{_PG}?\s*(?:اول|نخست|first|1)?\s*{_SEP}"
        rf"(?:the\s+)?(?:last|end|آخر|آخرین|پایان)(?:\s*{_PG})?", lambda m: None) if re.search(
        rf"(?:first|اول|نخست)\s*(?:{_PG})?\s*{_SEP}(?:the\s+)?(?:last|end|آخر|پایان)", t) else None
    cut(rf"(?:از\s+)?{_PG}\s*{_N}{_SEP}(?:{_PG}\s*)?{_N}", add_range)
    cut(rf"از\s+{_N}{_SEP}{_N}\s*{_PG}", add_range)
    cut(rf"{_N}\s*{_PG}\s+(?:آخر|آخرین|پایانی|انتهایی)", add_last)
    cut(rf"(?:آخرین|last)\s+{_N}\s*{_PG}", add_last)
    cut(rf"{_N}\s*{_PG}\s+(?:اول|ابتدایی|ابتدای|نخست|اولیه)", add_first)
    cut(rf"(?:اولین|نخستین|first)\s+{_N}\s*{_PG}", add_first)
    cut(rf"(?:فقط|تنها|only|just)\s+{_N}\s*{_PG}(?!\s*ای)", add_first)

    def add_one_first(m):
        state["ranges"].append([1, 1])
    cut(rf"(?:اولین|نخستین|first)\s+{_PG}(?!\s*\d)", add_one_first)
    cut(rf"{_PG}\s+(?:اول|نخست|اولین|نخستین)(?!\w)", add_one_first)
    cut(rf"(?:آخرین|last|final)\s+{_PG}(?!\s*\d)", lambda m: state.__setitem__("last", max(state["last"], 1)))
    cut(rf"{_PG}\s+(?:آخر|آخرین|پایانی)(?!\w)", lambda m: state.__setitem__("last", max(state["last"], 1)))

    def add_singles(m):
        for number in re.findall(r"\d{1,5}", m.group(0)):
            state["ranges"].append([int(number), int(number)])
    cut(rf"{_PG}\s*{_N}(?:\s*(?:،|,|و|and)\s*{_N})*", add_singles)
    try:
        return _clean(state)
    except PageSpecError:
        return None


def parse_list(value):
    """Selection typed in the form: "1-10, 15, 20 تا 30" (also «آخر 5»). Empty → None; garbage → PageSpecError."""
    text = fold(value).strip()
    if not text:
        return None
    state = {"ranges": [], "last": 0}
    for token in re.split(r"[,،;\n]+|\s+و\s+", text):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(rf"{_N}{_SEP}{_N}", token)
        tail = re.fullmatch(r"(?:آخر|آخرین|last)\s*(\d{1,5})", token)
        single = re.fullmatch(_N, token)
        if match:
            state["ranges"].append([int(match.group(1)), int(match.group(2))])
        elif tail:
            state["last"] = max(state["last"], int(tail.group(1)))
        elif single:
            state["ranges"].append([int(token), int(token)])
        else:
            raise PageSpecError(f"«{token}» قابل‌فهم نیست؛ مثال درست: 1-10, 15, 30-40")
    return _clean(state)


def normalize(value):
    """None / '' → None; text → parse_list; dict → validated copy."""
    if value in (None, "", {}):
        return None
    if isinstance(value, dict):
        try:
            return _clean(value)
        except (TypeError, ValueError, IndexError) as exc:
            raise PageSpecError("انتخاب صفحه معتبر نیست.") from exc
    return parse_list(value)


def resolve(spec, total):
    """Sorted 1-based page numbers selected by ``spec`` in a document of ``total`` pages."""
    pages = set()
    for first, last in spec.get("ranges") or []:
        if first > total:
            raise PageSpecError(f"این PDF فقط {total} صفحه دارد؛ صفحهٔ {first} وجود ندارد.")
        pages.update(range(first, min(last, total) + 1))
    if spec.get("last"):
        pages.update(range(max(1, total - int(spec["last"]) + 1), total + 1))
    if not pages:
        raise PageSpecError("هیچ صفحه‌ای انتخاب نشد.")
    if len(pages) > MAX_SELECTED_PAGES:
        raise PageSpecError(f"حداکثر {MAX_SELECTED_PAGES} صفحه در هر کار قابل انتخاب است.")
    return sorted(pages)


def describe(spec):
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in spec.get("ranges") or []]
    if spec.get("last"):
        parts.append(f"آخرین {spec['last']}")
    return ", ".join(parts)
