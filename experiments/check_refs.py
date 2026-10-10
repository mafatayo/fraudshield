"""Cross-check endnotes in thesis/make_ch*.js against paper/references.bib (both directions) and report recency."""
import io
import os
import re

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
bib = io.open(os.path.join(base, "paper", "references.bib"), encoding="utf8").read()
entries = []
for m in re.finditer(r"@\w+\{(\w+),(.*?)\n\}", bib, re.S):
    body = m.group(2)
    t = re.search(r"title\s*=\s*\{(.*?)\},?\s*\n", body, re.S).group(1)
    y = int(re.search(r"year\s*=\s*\{(\d{4})\}", body).group(1))
    entries.append((m.group(1), t.replace("{", "").replace("}", "").replace("\\", ""), y))


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


NOTE = re.compile(r'note\("((?:[^"\\\n]|\\.)*)"\)')
CH1 = re.compile(r'^\s+"((?:[^"\\\n]|\\.)*)",?\s*$', re.M)
notes = []
for ch in range(1, 5):
    s = io.open(os.path.join(base, "thesis", f"make_ch{ch}.js"), encoding="utf8").read()
    found = NOTE.findall(s)
    if ch == 1:
        k = s.index("const endnoteTexts")
        found = CH1.findall(s[k:s.index("];", k)])
    notes += [(ch, f.replace('\\"', '"')) for f in found]
print("notes found:", len(notes))
alltext = norm(" ".join(t for _, t in notes))
cited = set()
for k, t, y in entries:
    if norm(t)[:40] in alltext:
        cited.add(k)
print("bib entries NOT cited in any endnote:")
for k, t, y in entries:
    if k not in cited:
        print("  ", k, y, t[:70])
rec = sum(1 for _, _, y in entries if y >= 2021)
print("entries", len(entries), "recent", rec, "pct %.1f" % (100 * rec / len(entries)))
tn = [norm(t)[:40] for _, t, _ in entries]
print("note chunks whose quoted title is not in the bib:")
for ch, txt in notes:
    for part in txt.split("; "):
        mm = re.search(r'"([^"]{15,}?)[,.]?"', part)
        if mm and norm(mm.group(1))[:40] not in tn:
            print("  ch%d:" % ch, part[:120])
