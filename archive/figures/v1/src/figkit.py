"""figkit - primitives for paper-style SVG block diagrams.

Visual language (v3), measured on ICRA / IROS / RSS / CoRL method figures
(see references/paper-figure-study.md):
- Print-size typography: the canvas maps to a real print width (1400 px ~ text width,
  700 px ~ one column, both ~2.7 px per point), labels print at 6 to 9 pt, and
  `Fig.fs(role)` returns those sizes in canvas pixels. Labels are names, not sentences;
  explanations belong in the caption, and verbatim prompts or code go in `example` cards.
- Real imagery: `Fig.image` embeds renders, photos and data crops.
- Muted flat palette: a few pastel role fills with a slightly darker stroke,
  warm-gray neutrals, near-black thin connectors. No gradients on containers.
- Typography carries hierarchy: bold only for panel titles and key words,
  regular or medium weight for module names, serif italic for data names,
  variables and quoted language, monospace for tokens, code and model output.
- Shape carries meaning: token pills for sequences, trapezoids for encoders,
  bracketed vectors for actions, cylinders for stored data, block arrows for
  stage transitions, circled numbers for steps, dashed outlines for groups.
- Tight layout: small margins and gutters; the title lives in the caption.

All geometry is explicit. `qa_svg_figure.py` renders the SVG in headless Chrome
and gates text overflow, text collisions, box overlap, lines through labels and
canvas coverage. See `references/visual-contract.md` for the full contract.

Text API: plain text with `$...$` math islands, for example
"FK → $\\hat{\\tau}_{\\mathrm{EEF}}$". Math supports _ ^ {} \\hat \\bar \\tilde
\\mathrm \\mathbb \\mathcal, Greek letters, relations (= \\in \\le \\to \\gets)
with TeX-like spacing, and primes written as a'_t.

Icons: `Fig.asset` embeds vendored open-source SVGs (Tabler outline icons, LobeHub
model logos) fetched with `svgicons.py`; hand-drawn shapes stay for method content.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import math
import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from typing import NamedTuple

SANS = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Hiragino Sans GB', sans-serif"
SERIF = "STIXGeneral, 'Times New Roman', Times, 'Songti SC', serif"
MONO = "'SF Mono', Menlo, 'Roboto Mono', Consolas, 'PingFang SC', monospace"
MATH = "STIXGeneral, 'STIX Two Math', 'Times New Roman', serif"
FAMILIES = {"sans": SANS, "serif": SERIF, "mono": MONO}

INK = "#1F1F1F"
MUTED = "#5C5C5C"
FAINT = "#8C8C8C"
HAIR = "#D8D4CA"
WIRE = "#3A3A3A"

# Printed sizes measured on robotics method figures (references/paper-figure-study.md).
TEXT_WIDTH_PT = 516.0  # two-column text width, the width of a figure* float
COLUMN_WIDTH_PT = 252.0  # one column
TYPE_PT = {"min": 6.0, "label": 6.7, "module": 7.8, "title": 9.0, "hero": 12.0}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
_RASTER_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"RIFF", "image/webp"))


class Role:
    """A semantic color role: accent stroke, tint fill, deep text, mid fill (tokens, bars)."""

    def __init__(self, accent: str, tint: str, deep: str, mid: str):
        self.accent, self.tint, self.deep, self.mid = accent, tint, deep, mid
        self.g0, self.g1 = mid, tint  # kept for scripts written against v1


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


PAL = {
    "blue": Role("#3B8EA5", "#DCEEF3", "#1F5E70", "#A9D0DC"),  # teal: motor policy, perception
    "purple": Role("#8870B8", "#EAE4F4", "#54408A", "#C8BAE5"),  # lavender: control, structure
    "red": Role("#C4583C", "#F8E1D9", "#8A3421", "#E9AE9C"),  # terracotta: override, error, loss
    "green": Role("#6F9E57", "#E3EEDA", "#3F6630", "#B9D6A6"),  # sage: accept, success, feedback
    "amber": Role("#CF9A22", "#FBEDC6", "#7A5506", "#F3CF72"),  # ochre: reasoning model, key module
    "orange": Role("#C9824A", "#F6E6D7", "#85522A", "#E5BE9C"),  # clay: proprioception, secondary signal
    "gray": Role("#9C968A", "#EEECE6", "#46423B", "#D6D2C4"),  # stone: inputs, neutral containers
}
for _alias, _key in (("teal", "blue"), ("lavender", "purple"), ("terracotta", "red"), ("sage", "green"),
                     ("ochre", "amber"), ("clay", "orange"), ("stone", "gray")):
    PAL[_alias] = PAL[_key]

# ---------------------------------------------------------------------------
# Text measurement (estimate only; qa_svg_figure.py measures the real render)
# ---------------------------------------------------------------------------


def _cw(ch: str) -> float:
    o = ord(ch)
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if ch == " ":
        return 0.28
    if ch in "il.,:;|!'·":
        return 0.26
    if ch in "fjtrI()[]{}/-":
        return 0.35
    if ch in "mwMW%":
        return 0.85
    if ch.isupper():
        return 0.68
    if ch.isdigit():
        return 0.56
    return 0.54


def text_w(s: str, size: float, bold: bool = False, ls: float = 0.0) -> float:
    plain = re.sub(r"\$[^$]*\$", lambda m: "x" * max(1, len(_math_plain(m.group(0)[1:-1]))), s)
    return sum(_cw(c) for c in plain) * size * (1.06 if bold else 1.0) + ls * len(plain)


# ---------------------------------------------------------------------------
# Mini TeX-like math: _ ^ {} \hat \bar \mathrm \mathbb \mathcal + symbols
# ---------------------------------------------------------------------------

_SYM = {
    "pi": "π", "tau": "τ", "theta": "θ", "eta": "η", "psi": "ψ", "phi": "φ", "ell": "ℓ",
    "alpha": "α", "beta": "β", "sigma": "σ", "epsilon": "ε", "Delta": "Δ", "Sigma": "Σ",
    "in": "∈", "times": "×", "le": "≤", "ge": "≥", "to": "→", "approx": "≈", "neq": "≠",
    "cdot": "·", "mid": "∣", "notin": "∉", "uparrow": "↑", "downarrow": "↓", "star": "⋆",
    "subseteq": "⊆", "subset": "⊂", "cup": "∪", "cap": "∩", "langle": "⟨", "rangle": "⟩", "forall": "∀", "exists": "∃",
    "prec": "≺", "preceq": "⪯", "mapsto": "↦", "Rightarrow": "⇒", "leftrightarrow": "↔", "rightarrow": "→", "leftarrow": "←",
    "wedge": "∧", "vee": "∨", "neg": "¬", "emptyset": "∅", "ldots": "…", "cdots": "⋯",
    "delta": "δ", "kappa": "κ", "rho": "ρ", "gamma": "γ", "omega": "ω", "Pi": "Π", "Gamma": "Γ", "Theta": "Θ", "Omega": "Ω",
    "Lambda": "Λ", "Phi": "Φ", "Psi": "Ψ",
    "top": "⊤", "sqrt": "√", "odot": "⊙", "oplus": "⊕", "otimes": "⊗", "lambda": "λ", "mu": "μ", "partial": "∂", "nabla": "∇", "prime": "′", "infty": "∞", "sim": "∼", "gets": "←",
    ";": " ", ",": " ", " ": " ", "quad": "  ",
}
_ITALIC_GREEK = {"pi", "tau", "theta", "eta", "psi", "phi", "ell", "alpha", "beta", "sigma", "epsilon", "lambda", "mu",
                 "delta", "kappa", "rho", "gamma", "omega"}
_BB = {"R": "ℝ", "N": "ℕ", "E": "𝔼"}
_CAL = {"L": "ℒ", "J": "𝒥", "D": "𝒟", "O": "𝒪", "A": "𝒜", "V": "𝒱", "R": "ℛ", "N": "𝒩", "M": "ℳ", "F": "ℱ", "S": "𝒮", "T": "𝒯",
        "G": "𝒢", "C": "𝒞", "H": "ℋ", "B": "ℬ", "X": "𝒳", "E": "ℰ", "P": "𝒫", "I": "ℐ", "K": "𝒦", "Q": "𝒬", "U": "𝒰",
        "W": "𝒲", "Y": "𝒴", "Z": "𝒵"}


def _math_plain(src: str) -> str:
    return re.sub(r"\\[a-zA-Z]+|[{}_^]", "", src)


def _single_token(src: str, i: int) -> str:
    """One unbraced script/accent argument: a command such as \\theta, or one character."""
    if i >= len(src):
        return ""
    m = re.match(r"\\[a-zA-Z]+|\\.", src[i:])
    return m.group(0) if m else src[i]


def _parse(src: str, i: int = 0, upright: bool = False, stop: str | None = None):
    atoms: list[dict] = []
    while i < len(src):
        ch = src[i]
        if stop and ch == stop:
            return atoms, i + 1
        if ch == "{":
            grp, i = _parse(src, i + 1, upright, "}")
            atoms.extend(grp)
            continue
        if ch in "_^":
            i += 1
            if i < len(src) and src[i] == "{":
                arg, i = _parse(src, i + 1, upright, "}")
            else:
                token = _single_token(src, i)
                arg, _ = _parse(token, 0, upright)
                i += len(token)
            if not atoms:
                atoms.append({"t": "", "it": False})
            atoms[-1]["sub" if ch == "_" else "sup"] = arg
            continue
        if ch == "\\":
            m = re.match(r"\\([a-zA-Z]+|.)", src[i:])
            cmd = m.group(1)
            i += len(m.group(0))
            if cmd in ("hat", "bar", "tilde", "mathrm", "mathbb", "mathcal", "text"):
                if i < len(src) and src[i] == "{":
                    arg, i = _parse(src, i + 1, cmd in ("mathrm", "text") or upright, "}")
                else:
                    token = _single_token(src, i)
                    arg, _ = _parse(token, 0, upright)
                    i += len(token)
                if cmd == "mathbb":
                    arg = [{"t": "".join(_BB.get(c, c) for c in a["t"]), "it": False} for a in arg]
                elif cmd == "mathcal":
                    arg = [{"t": "".join(_CAL.get(c, c) for c in a["t"]), "it": False} for a in arg]
                elif cmd in ("hat", "bar", "tilde") and arg:
                    mark = {"hat": "\u0302", "bar": "\u0304", "tilde": "\u0303"}[cmd]
                    arg[-1] = dict(arg[-1], t=arg[-1]["t"] + mark)
                atoms.extend(arg)
                continue
            atoms.append({"t": _SYM.get(cmd, cmd), "it": cmd in _ITALIC_GREEK and not upright})
            continue
        if ch == "'":
            atoms.append({"t": "′", "it": False})
            i += 1
            continue
        if ch == " ":
            i += 1
            continue
        t = "\u2212" if ch == "-" else ch
        atoms.append({"t": t, "it": ch.isalpha() and not upright})
        i += 1
    return atoms, i


_REL = set("=∈∉≤≥→≈≠∼←⊆⊂↦≺⪯⇒↔")
_BIN = set("+\u2212×")


def _spaced(t: str, script: bool) -> str:
    """TeX-like spacing: relations get a medium space, binaries/commas a thin one (base level only)."""
    if script:
        return t
    if t in _REL:
        return f"\u2005{t}\u2005"
    if t in _BIN:
        return f"\u2009{t}\u2009"
    if t == ",":
        return ",\u2009"
    if t == "∣":
        return f"\u2005{t}\u2005"
    return t


def _emit(atoms, size, base_off, cur, out, script=False):
    for a in atoms:
        if a["t"]:
            dy = base_off - cur[0]
            cur[0] = base_off
            style = "italic" if a["it"] else "normal"
            out.append(
                f'<tspan font-family="{MATH}" font-style="{style}" font-size="{size:.2f}"'
                + (f' dy="{dy:.2f}"' if abs(dy) > 1e-6 else "")
                + f">{escape(_spaced(a['t'], script))}</tspan>"
            )
        sup, sub = a.get("sup"), a.get("sub")
        s2 = size * 0.7
        if sup:
            _emit(sup, s2, base_off - size * 0.38, cur, out, True)
        if sub:
            if sup:
                back = -sum(_cw(c) for c in _math_plain("".join(x["t"] for x in sup))) * s2 * 0.95
                out.append(f'<tspan font-size="{s2:.2f}" dx="{back:.2f}"></tspan>')
            _emit(sub, s2, base_off + size * 0.28, cur, out, True)


def rich(s: str, size: float) -> str:
    """Plain text with $math$ islands -> tspans (baseline restored at the end)."""
    out: list[str] = []
    cur = [0.0]
    for part in re.split(r"(\$[^$]*\$)", s):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            atoms, _ = _parse(part[1:-1])
            _emit(atoms, size * 1.08, 0.0, cur, out)
        else:
            dy = -cur[0]
            cur[0] = 0.0
            out.append("<tspan" + (f' dy="{dy:.2f}"' if abs(dy) > 1e-6 else "") + f">{escape(part)}</tspan>")
    if abs(cur[0]) > 1e-6:
        out.append(f'<tspan dy="{-cur[0]:.2f}"></tspan>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Vendored open-source SVG assets
# ---------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"
# Root presentation attributes worth keeping. stroke-width is dropped so each use sets one display
# stroke; width, height, class and style belong to the icon's original page, not to this figure.
_ASSET_ROOT_KEEP = ("fill", "stroke", "stroke-linecap", "stroke-linejoin", "fill-rule", "clip-rule")
_ASSET_DROP_TAGS = {"title", "desc", "metadata"}
_BLACK_RE = re.compile(r"#0{3,8}|black|rgb\(\s*0\s*,\s*0\s*,\s*0\s*\)", re.IGNORECASE)
# Symbols share the figure's document, so scripts, stylesheets, embedded documents and external
# references could run code, restyle every label, or pull remote content at render time.
_ASSET_UNSAFE = re.compile(
    r"<\s*(?:script|style|foreignobject|iframe|image)\b|<!doctype|<!entity|\son[a-z]+\s*="
    r"|href\s*=\s*[\"'](?!#)|url\(\s*(?!#)",
    re.IGNORECASE,
)


class SvgAsset(NamedTuple):
    slug: str
    view_box: tuple[float, float, float, float]
    body: str


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def parse_svg_asset(raw: str, source: str = "asset", tint: bool = True) -> SvgAsset:
    """Validate a small standalone SVG and turn it into symbol-ready markup with namespaced ids.

    With `tint` (monochrome icons) literal black paint becomes currentColor so the icon takes the figure's
    colour; colour illustrations such as Fluent Emoji are parsed with tint=False and keep every fill."""
    bad = _ASSET_UNSAFE.search(raw)
    if bad:
        raise ValueError(f"{source}: unsafe SVG content near {bad.group(0)!r}")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"{source}: not well-formed SVG ({exc})") from exc
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError(f"{source}: root element is not an SVG <svg>")
    try:
        vb = tuple(float(v) for v in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()))
    except ValueError as exc:
        raise ValueError(f"{source}: invalid viewBox") from exc
    if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0:
        raise ValueError(f"{source}: missing or invalid viewBox")

    stem = re.sub(r"[^a-z0-9]+", "-", Path(source).stem.lower()).strip("-") or "asset"
    slug = f"{stem}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}"
    for el in root.iter():
        el.tag = _local(el.tag)
        for key, value in list(el.attrib.items()):
            del el.attrib[key]
            key = _local(key)
            if key == "id":
                value = f"{slug}-{value}"
            elif key == "href":
                value = f"#{slug}-{value[1:]}"
            else:
                value = re.sub(r"url\(\s*#([^)\s]+)\s*\)", lambda m: f"url(#{slug}-{m.group(1)})", value)
            el.set(key, value)
    painted = not tint
    for el in root.iter() if tint else ():
        for key in ("fill", "stroke"):
            value = (el.get(key) or "").strip()
            if not value or value.lower() == "none":
                continue
            # A solid-shape icon paints in literal black; tint it like a line icon does.
            el.set(key, "currentColor" if _BLACK_RE.fullmatch(value) else value)
            painted = True
    children = "".join(ET.tostring(child, encoding="unicode", short_empty_elements=True)
                       for child in root if child.tag not in _ASSET_DROP_TAGS)
    attrs = "".join(f' {k}="{escape(root.get(k), quote=True)}"' for k in _ASSET_ROOT_KEEP if root.get(k))
    if not painted:
        attrs += ' fill="currentColor"'  # Material Symbols and friends leave the paint implicit
    return SvgAsset(slug, vb, f"<g{attrs}>{children}</g>")


def load_svg_asset(path, tint: bool = True) -> SvgAsset:
    path = Path(path)
    return parse_svg_asset(path.read_text(encoding="utf-8"), str(path), tint)


def load_raster(path) -> tuple[str, bytes]:
    """Read a PNG, JPEG or WebP file for embedding; the type comes from its magic bytes, not its name."""
    data = Path(path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{path}: {len(data)} bytes exceeds the {MAX_IMAGE_BYTES} byte image limit; downscale it first")
    for magic, mime in _RASTER_MAGIC:
        if data.startswith(magic) and (mime != "image/webp" or data[8:12] == b"WEBP"):
            return mime, data
    raise ValueError(f"{path}: not a PNG, JPEG or WebP image")


# ---------------------------------------------------------------------------
# Icons (24x24 line glyphs; offline fallback when no vendored asset fits)
# ---------------------------------------------------------------------------

ICONS = {
    "spark": '<path d="M11 3c.6 4.3 2.7 6.4 7 7-4.3.6-6.4 2.7-7 7-.6-4.3-2.7-6.4-7-7 4.3-.6 6.4-2.7 7-7z"/><path d="M18.5 15c.3 1.7 1 2.4 2.7 2.7-1.7.3-2.4 1-2.7 2.7-.3-1.7-1-2.4-2.7-2.7 1.7-.3 2.4-1 2.7-2.7z"/>',
    "check": '<circle cx="12" cy="12" r="9.5"/><path d="M7.5 12.3l3 3 6-6.3"/>',
    "x": '<circle cx="12" cy="12" r="9.5"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/>',
    "wrench": '<path d="M14.7 4.2a4.6 4.6 0 0 0-5.6 5.9L3.7 15.5a1.9 1.9 0 0 0 2.7 2.7l5.4-5.4a4.6 4.6 0 0 0 5.9-5.6l-2.8 2.8-2.5-.4-.4-2.5z"/>',
    "joint": '<circle cx="12" cy="12" r="3.2"/><circle cx="12" cy="12" r="7.5"/><path d="M12 1.8v2.7M12 19.5v2.7M1.8 12h2.7M19.5 12h2.7M4.8 4.8l1.9 1.9M17.3 17.3l1.9 1.9M4.8 19.2l1.9-1.9M17.3 6.7l1.9-1.9"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "chat": '<path d="M4 4.5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-9l-5 4v-4H4a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1z"/><path d="M7.5 9.5h9M7.5 12.5h5.5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
    "plan": '<rect x="4" y="3.5" width="16" height="17" rx="2"/><path d="M7.5 8.5l1.5 1.5 2.5-2.5M7.5 14.5l1.5 1.5 2.5-2.5M14.5 9h2.5M14.5 15h2.5"/>',
    "recover": '<path d="M19.5 10.5A8 8 0 0 0 5.2 7.3"/><path d="M4.5 3.5v4.3h4.3"/><path d="M4.5 13.5a8 8 0 0 0 14.3 3.2"/><path d="M19.5 20.5v-4.3h-4.3"/>',
    "gripper": '<path d="M12 2.5v5"/><rect x="6.5" y="7.5" width="11" height="4" rx="1"/><path d="M8.5 11.5v4l2.3 5M15.5 11.5v4l-2.3 5"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "bot": '<rect x="4.5" y="8" width="15" height="11.5" rx="3"/><path d="M12 8V5"/><circle cx="12" cy="3.6" r="1.3"/><circle cx="9.2" cy="13.2" r="1.2"/><circle cx="14.8" cy="13.2" r="1.2"/><path d="M2.5 12.2v3M21.5 12.2v3"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "brain": '<path d="M11.3 5.2A3 3 0 0 0 6.2 6.4 3 3 0 0 0 4 11.5a3 3 0 0 0 1.6 4.8 3 3 0 0 0 3.2 3.5 2.6 2.6 0 0 0 2.5-1.4z"/><path d="M12.7 5.2a3 3 0 0 1 5.1 1.2 3 3 0 0 1 2.2 5.1 3 3 0 0 1-1.6 4.8 3 3 0 0 1-3.2 3.5 2.6 2.6 0 0 1-2.5-1.4z"/>',
    "contact": '<path d="M12 2.5v9"/><path d="M8.5 8l3.5 3.5L15.5 8"/><path d="M3 15h18"/><path d="M5.5 19l2.5-2.5M10 19l2.5-2.5M14.5 19l2.5-2.5M19 19l1.5-1.5"/>',
    "target": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.3"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/>',
    "dist": '<path d="M2 19.5h20"/><path d="M2 19c3 0 4.2-11 7-11s4 11 7 11"/><path d="M8 19c3 0 4.2-11 7-11s4 11 7 11" stroke-dasharray="2 2"/>',
    "eyeoff": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/><path d="M4 20L20 4"/>',
    "zigzag": '<path d="M2.5 17.5l4-6 3 4 4-9 3 6 5-5"/>',
    "traj": '<circle cx="4.5" cy="18.5" r="1.8"/><circle cx="19.5" cy="5.5" r="1.8"/><path d="M6.3 17.6c3.2-1.2 3.4-5.4 6.2-6.4 2.4-.9 3.9-2.4 5.3-4.3" stroke-dasharray="2.2 2.4"/>',
    "person": '<circle cx="12" cy="7.5" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>',
}

# Schematic bimanual tabletop scene: muted, photo-like tones rather than clip-art colors.
SCENE = """<symbol id="scene" viewBox="0 0 130 96">
<rect width="130" height="96" fill="#D3D6D8"/>
<rect width="130" height="20" fill="#C8CBCE"/>
<path d="M0 58H130V96H0z" fill="#E4DCCB"/><path d="M0 58H130" stroke="#C9BCA3" stroke-width="1"/>
<path d="M0 90H130" stroke="#D4C8B1" stroke-width="1"/>
<ellipse cx="105" cy="78" rx="19" ry="3" fill="#000" opacity="0.14"/>
<ellipse cx="68" cy="72" rx="8" ry="2" fill="#000" opacity="0.14"/>
<ellipse cx="76" cy="88" rx="8" ry="2" fill="#000" opacity="0.14"/>
<path d="M89 55H121L118 77H92z" fill="#34383E"/><path d="M89 55H121" stroke="#50555C" stroke-width="2"/>
<path d="M62 60l3-3h12l-3 3z" fill="#CF6B5F"/><rect x="62" y="60" width="12" height="11" rx="0.8" fill="#B4473B"/>
<path d="M70 76l3-3h12l-3 3z" fill="#6E93C8"/><rect x="70" y="76" width="12" height="12" rx="0.8" fill="#3E67A6"/>
<g fill="none" stroke-linecap="round" stroke-linejoin="round">
<path d="M14-4L20 26L42 36L48 49" stroke="#9AA0A6" stroke-width="8.5"/><path d="M14-4L20 26L42 36L48 49" stroke="#ECEEEF" stroke-width="5.5"/>
<path d="M116-4L110 26L90 34L82 46" stroke="#9AA0A6" stroke-width="8.5"/><path d="M116-4L110 26L90 34L82 46" stroke="#ECEEEF" stroke-width="5.5"/>
</g>
<circle cx="20" cy="26" r="3.4" fill="#5C6167"/><circle cx="42" cy="36" r="3.4" fill="#5C6167"/>
<circle cx="110" cy="26" r="3.4" fill="#5C6167"/><circle cx="90" cy="34" r="3.4" fill="#5C6167"/>
<path d="M45 50l1 7M51 50l-1 7M79 47l1 7M85 47l-1 7" stroke="#3E4247" stroke-width="2.4" stroke-linecap="round"/>
</symbol>"""


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


class Fig:
    def __init__(self, w: int, h: int, bg: str = "#FFFFFF", print_width_pt: float | None = None):
        """`print_width_pt` is the printed width of the whole canvas; the default treats canvases at least
        1000 px wide as text-width (figure*) floats and narrower ones as single-column floats."""
        self.w, self.h, self.bg = w, h, bg
        self.print_width_pt = print_width_pt or (TEXT_WIDTH_PT if w >= 1000 else COLUMN_WIDTH_PT)
        self.defs: dict[str, str] = {}
        self.body: list[str] = []
        self.rects: dict[str, tuple[float, float, float, float]] = {}  # box id -> (x, y, w, h), for ports and layout
        self._n = 0
        self._assets: dict[str, SvgAsset] = {}

    # -- plumbing ----------------------------------------------------------
    def fs(self, role: str = "label") -> float:
        """Font size in canvas px that prints at the measured paper size for `role`
        (min, label, module, title, hero)."""
        return round(TYPE_PT[role] * self.w / self.print_width_pt, 1)

    def _reg(self, cid: str, x, y, w, h) -> str:
        """Remember a container's rectangle so ports, connectors and layout helpers can use it."""
        self.rects[cid] = (float(x), float(y), float(w), float(h))
        return cid

    def rect(self, box: str) -> tuple[float, float, float, float]:
        """(x, y, w, h) of a container returned by panel, card, chip, example, image, scene and friends."""
        if box not in self.rects:
            raise KeyError(f"unknown box {box!r}; only containers register a rectangle")
        return self.rects[box]

    # -- layout grid ---------------------------------------------------------
    def cols(self, x0, x1, n, gap=12) -> list[tuple[float, float]]:
        """Split [x0, x1] into n equal columns with equal gaps, snapped to whole pixels.

        Returns [(x, w), ...]. Using it instead of hand-typed coordinates keeps columns and gaps exact,
        which is what makes a figure read as tidy.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        w = (x1 - x0 - gap * (n - 1)) / n
        if w <= 0:
            raise ValueError(f"{n} columns with {gap} px gaps do not fit in {x1 - x0} px")
        return [(round(x0 + i * (w + gap)), round(w)) for i in range(n)]

    def rows(self, y0, y1, n, gap=12) -> list[tuple[float, float]]:
        """Split [y0, y1] into n equal rows with equal gaps, snapped to whole pixels."""
        return self.cols(y0, y1, n, gap)

    def place(self, x0, x1, widths, gap=None) -> list[float]:
        """x positions for the given widths inside [x0, x1]: equal gaps, whole run centered."""
        widths = list(widths)
        n = len(widths)
        total = sum(widths)
        if gap is None:
            gap = (x1 - x0 - total) / (n - 1) if n > 1 else 0
        if total + gap * (n - 1) > (x1 - x0) + 0.5:
            raise ValueError(f"{widths} with {gap:g} px gaps need {total + gap * (n - 1):g} px, "
                             f"but only {x1 - x0:g} px are available")
        start = x0 + max(0.0, (x1 - x0 - total - gap * (n - 1)) / 2)
        out, x = [], start
        for w in widths:
            out.append(round(x))
            x += w + gap
        return out

    def zone(self, x, y, w, h) -> str:
        """Register an invisible rectangle so `connect`, `route`, `arc` and `port` can anchor to a row or
        band that has no card of its own (a numbered step, a label column, a reserved lane)."""
        return self._reg(self.uid("z"), x, y, w, h)

    def uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def add(self, *parts: str) -> None:
        self.body.extend(parts)

    def _marker(self, color: str, size: float = 7.0, open_: bool = False) -> str:
        mid = f"ah{color.strip('#')}{int(size)}{'o' if open_ else ''}"
        if mid not in self.defs:
            shape = (
                f'<path d="M1 1.2L9 5L1 8.8" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="miter"/>'
                if open_
                else f'<path d="M0 1L10 5L0 9z" fill="{color}"/>'
            )
            self.defs[mid] = (
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{size}" '
                f'markerHeight="{size}" markerUnits="userSpaceOnUse" orient="auto-start-reverse">{shape}</marker>'
            )
        return mid

    def svg(self) -> str:
        defs = "".join(self.defs.values())
        return (
            # Default family on the root (inherited) rather than a `text{}` CSS rule: a stylesheet rule
            # would override per-element font-family attributes and silently drop serif/mono labels.
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'font-family="{SANS}" data-print-width-pt="{self.print_width_pt:g}">'
            f"<style>text{{white-space:pre;font-kerning:normal}}text:not([fill]){{fill:{INK}}}"
            f".ic{{fill:none;stroke:currentColor;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}}</style>"
            f"<defs>{defs}</defs>"
            f'<rect width="{self.w}" height="{self.h}" fill="{self.bg}"/>' + "".join(self.body) + "</svg>"
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.svg())

    # -- text --------------------------------------------------------------
    def text(self, x, y, s, size=None, weight=400, color=INK, anchor="start", ls=0.0, box=None,
             italic=False, opacity=None, family="sans"):
        """Rich text. family: sans (labels), serif (data names, quoted language), mono (tokens, code).
        `size` defaults to the label size for the canvas print width."""
        size = size or self.fs("label")
        attrs = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"'
        if family != "sans":
            attrs += f' font-family="{FAMILIES[family]}"'
        if weight != 400:
            attrs += f' font-weight="{weight}"'
        if anchor != "start":
            attrs += f' text-anchor="{anchor}"'
        if ls:
            attrs += f' letter-spacing="{ls}"'
        if italic:
            attrs += ' font-style="italic"'
        if opacity is not None:
            attrs += f' opacity="{opacity}"'
        if box:
            attrs += f' data-in="{box}"'
        self.add(f"<text {attrs}>{rich(s, size)}</text>")

    # -- containers ----------------------------------------------------------
    def panel(self, x, y, w, h, role, title, sub=None, dashed=False, horizontal=False, r=8, title_size=None, fill=None,
              sub_below=False):
        """Flat stage panel with a left-aligned bold title and a regular gray subtitle on the same line
        (or on the next line with sub_below=True for narrow panels).

        Returns the content top y. `title=None` draws a headerless container.
        """
        p = PAL[role]
        bid = self.uid("panel")
        if dashed:
            body = f'fill="{fill or "#FFFFFF"}" stroke="{p.accent}" stroke-opacity="0.75" stroke-width="1.2" stroke-dasharray="5 4"'
        else:
            body = f'fill="{fill or _mix(p.tint, "#FFFFFF", 0.45)}" stroke="none"'
        self.add(f'<rect data-box="{bid}" data-kind="panel" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" {body}/>')
        self._reg(bid, x, y, w, h)
        if not title:
            return y + 10
        title_size = title_size or self.fs("title")
        sub_size = max(self.fs("min"), round(title_size * 0.8, 1))
        ty = y + round(title_size * 1.47)
        markup = f'<tspan font-weight="700">{rich(title, title_size)}</tspan>'
        if sub and not sub_below:
            markup += f'<tspan dx="{round(title_size * 0.6)}" font-size="{sub_size}" fill="{MUTED}">{rich(sub, sub_size)}</tspan>'
        self.add(f'<text x="{x + 12}" y="{ty}" font-size="{title_size}" fill="{INK}" data-in="{bid}">{markup}</text>')
        if sub and sub_below:
            sy = ty + round(sub_size * 1.35)
            self.text(x + 12, sy, sub, size=sub_size, color=MUTED, box=bid)
            return sy + round(sub_size * 0.83)
        return ty + round(title_size * 0.8)

    def card(self, x, y, w, h, role=None, fill=None, sw=1.0, r=5, stack=0, topbar=False, dashed=False,
             stroke=None, kind="card", key=False):
        """Pastel card: role tint fill with a slightly darker stroke. key=True draws the near-black outline
        used for the one module the reader should find first."""
        p = PAL[role] if role else None
        fl = fill or (p.tint if p else "#FFFFFF")
        st = stroke or (INK if key else (p.accent if p else HAIR))
        sw = 1.4 if key and sw == 1.0 else sw
        cid = self.uid("c")
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        for k in range(stack, 0, -1):
            o = 4 * k
            self.add(f'<rect data-qa="ignore" x="{x + o}" y="{y - o}" width="{w}" height="{h}" rx="{r}" '
                     f'fill="{_mix(fl, "#FFFFFF", 0.35)}" '
                     f'stroke="{st}" stroke-opacity="0.45" stroke-width="{sw}"/>')
        self.add(f'<rect data-box="{cid}" data-kind="{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                 f'fill="{fl}" stroke="{st}" stroke-width="{sw}"{dash}/>')
        self._reg(cid, x, y, w, h)
        if topbar and p:
            self.add(f'<path d="M{x + r} {y + 1.5}H{x + w - r}" stroke="{p.accent}" stroke-width="3" stroke-linecap="round"/>')
        return cid

    def example(self, x, y, w, h, role=None, fill=None, stroke=None, dashed=False, r=5):
        """Card for verbatim example content (an instruction, prompt, generated code, reasoning trace).
        Labels boxed in it may be full sentences; the QA word limits skip them."""
        return self.card(x, y, w, h, role, fill=fill, stroke=stroke, dashed=dashed, r=r, kind="example")

    def chip(self, x, y, w, h, s, role=None, size=None, fill=None, color=None, weight=400, r=3, stroke=None,
             dashed=False, family="sans", italic=False):
        size = size or self.fs("label")
        p = PAL[role] if role else None
        cid = self.card(x, y, w, h, fill=fill or (p.tint if p else "#FFFFFF"), stroke=stroke or (p.mid if p else HAIR),
                        sw=0.9, r=r, dashed=dashed, kind="chip")
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, weight=weight, color=color or (p.deep if p else INK),
                  anchor="middle", box=cid, family=family, italic=italic)
        return cid

    def badge(self, x, y, s, role, size=None, h=None, w=None):
        """Quiet tag for a key number: tint fill, bold deep text, no outline."""
        size = size or self.fs("min")
        h = h or round(size * 1.6)
        p = PAL[role]
        w = w or text_w(s, size, bold=True) + 12
        cid = self.uid("b")
        self.add(f'<rect data-box="{cid}" data-kind="chip" x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="3" fill="{p.tint}" '
                 f'stroke="{p.mid}" stroke-width="0.8"/>')
        self._reg(cid, x, y, w, h)
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, weight=700, color=p.deep, anchor="middle", box=cid)
        return w

    def pill(self, cx, cy, s, role="gray", size=None, h=None, fill="#FFFFFF", weight=400, italic=True, family="serif"):
        """Label that sits on a connector: white knock-out background, serif italic text."""
        size = size or self.fs("label")
        h = h or round(size * 1.6)
        p = PAL[role]
        w = text_w(s, size) + 12
        cid = self.uid("p")
        self.add(f'<rect data-box="{cid}" data-kind="chip" x="{cx - w / 2:.1f}" y="{cy - h / 2}" width="{w:.1f}" height="{h}" '
                 f'rx="3" fill="{fill}"/>')
        self._reg(cid, cx - w / 2, cy - h / 2, w, h)
        self.text(cx, cy + size * 0.34, s, size=size, color=p.deep, anchor="middle", box=cid, weight=weight,
                  italic=italic, family=family)
        return w

    def step(self, cx, cy, n, r=None, color=INK, size=None):
        """Circled step number (thin outline)."""
        size = size or self.fs("min")
        r = r or round(size * 0.78, 1)
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{color}" stroke-width="1"/>')
        self.text(cx, cy + size * 0.36, str(n), size=size, color=color, anchor="middle")

    # -- neural-network visuals ------------------------------------------------
    _SHADES = (0.35, 0.85, 0.5, 0.95, 0.25, 0.7, 0.45, 0.8, 0.3, 0.6, 0.9, 0.4)

    def _shade(self, role, v):
        p = PAL[role]
        return _mix(p.tint, p.accent, 0.15 + 0.8 * v)

    @staticmethod
    def cells(x, y, w, h, rows, cols):
        """Cell geometry of a rows x cols grid over (x, y, w, h): returns cell(r, c) -> (cx, cy, cw, ch), the
        cell's centre and size, so objects drawn inside a patch grid can sit on its lines and centres."""
        cw, ch = w / cols, h / rows

        def cell(r, c):
            return x + (c + 0.5) * cw, y + (r + 0.5) * ch, cw, ch
        return cell

    def patch_grid(self, x, y, w, h, rows, cols, masked=(), content=None, fill="#EEF0F0", line="#FFFFFF",
                   mask_fill="#C9C4B8", frame="#B9B4A8"):
        """An image cut into patches (ViT, MAE, JEPA). `content(x, y, w, h)` draws the picture between the
        background and the grid, `masked` lists (row, col) patches that are hidden from the encoder. Snap the
        picture to the patches with `Fig.cells`: put a horizon or table edge on a grid line and centre each
        object in a cell, because a line 2 px off a patch boundary reads as a mistake."""
        box = self.card(x, y, w, h, fill=fill, stroke=frame, r=2, kind="thumb")
        if content:
            content(x, y, w, h)
        cw, ch = w / cols, h / rows
        for r, c in masked:
            self.add(f'<rect data-qa="ignore" x="{x + c * cw:.1f}" y="{y + r * ch:.1f}" width="{cw:.1f}" '
                     f'height="{ch:.1f}" fill="{mask_fill}"/>')
        grid = "".join(f"M{x + c * cw:.1f} {y}V{y + h}" for c in range(1, cols))
        grid += "".join(f"M{x} {y + r * ch:.1f}H{x + w}" for r in range(1, rows))
        self.add(f'<path data-qa="ignore" d="{grid}" fill="none" stroke="{line}" stroke-width="1.2"/>')
        self.add(f'<rect data-qa="ignore" x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" '
                 f'stroke="{frame}" stroke-width="1"/>')
        return box

    def vector(self, x, y, n, role, cell=12, vertical=True, values=None, gap=1):
        """An embedding drawn as n cells shaded by value, the usual picture of z or h. Returns its box id."""
        values = values or [self._SHADES[i % len(self._SHADES)] for i in range(n)]
        for i, v in enumerate(values[:n]):
            cx, cy = (x, y + i * (cell + gap)) if vertical else (x + i * (cell + gap), y)
            self.add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell}" height="{cell}" rx="1.5" '
                     f'fill="{self._shade(role, v)}" stroke="{PAL[role].accent}" stroke-width="0.8"/>')
        span = n * cell + (n - 1) * gap
        w, h = (cell, span) if vertical else (span, cell)
        return self._box("chip", x, y, w, h)

    def token_grid(self, x, y, rows, cols, role, cell=12, gap=2, values=None, masked=(), highlight=(),
                   highlight_role="red"):
        """A grid of tokens or patch embeddings. `masked` cells are drawn empty and dashed (not seen),
        `highlight` cells take a second role, for example the targets a predictor has to fill in."""
        k = 0
        for r in range(rows):
            for c in range(cols):
                cx, cy = x + c * (cell + gap), y + r * (cell + gap)
                if (r, c) in masked:
                    self.add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell}" height="{cell}" rx="1.5" '
                             f'fill="#FFFFFF" stroke="#B9B4A8" stroke-width="0.8" stroke-dasharray="2 1.5"/>')
                    continue
                v = (values[k] if values else self._SHADES[k % len(self._SHADES)])
                use = highlight_role if (r, c) in highlight else role
                self.add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell}" height="{cell}" rx="1.5" '
                         f'fill="{self._shade(use, v)}" stroke="{PAL[use].accent}" stroke-width="0.8"/>')
                k += 1
        return self._box("chip", x, y, cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap)

    def heatmap(self, x, y, rows, cols, role, cell=10, values=None):
        """An attention or similarity map; the default values concentrate near the diagonal."""
        for r in range(rows):
            for c in range(cols):
                v = values[r][c] if values else max(0.08, 1 - abs(r - c * rows / max(cols, 1)) / 2.2)
                self.add(f'<rect x="{x + c * cell:.1f}" y="{y + r * cell:.1f}" width="{cell}" height="{cell}" '
                         f'fill="{PAL[role].accent}" fill-opacity="{0.1 + 0.85 * min(1, v):.2f}"/>')
        self.add(f'<rect data-qa="ignore" x="{x}" y="{y}" width="{cols * cell}" height="{rows * cell}" '
                 f'fill="none" stroke="{PAL[role].mid}" stroke-width="0.8"/>')
        return self._box("chip", x, y, cols * cell, rows * cell)

    def layer_stack(self, x, y, w, h, n, role, s=None, size=None, depth=5, repeat=None, fill=None,
                    dashed=False, key=False, family="sans", italic=False):
        """A deep network drawn as n stacked layers inside the footprint (x, y, w, h): the back layers step up
        and to the right by `depth`, the front layer carries the label, and `repeat` (for example
        "$\\times L$") marks how many blocks the stack stands for. Returns the footprint's box id, so a wire
        leaving to the right starts clear of the back sheets; the front face is left in `self.stack_front`."""
        p = PAL[role]
        off = depth * (n - 1)
        fw, fh = w - off, h - off
        for k in range(n - 1, 0, -1):
            o = depth * k
            self.add(f'<rect data-qa="ignore" x="{x + o}" y="{y + off - o}" width="{fw}" height="{fh}" rx="6" '
                     f'fill="{_mix(fill or p.tint, "#FFFFFF", 0.25 + 0.12 * k)}" stroke="{p.accent}" '
                     f'stroke-opacity="{0.35 + 0.1 * (n - k)}" stroke-width="1"/>')
        front = self.card(x, y + off, fw, fh, fill=fill or p.tint, stroke=INK if key else p.accent,
                          sw=1.4 if key else 1.1, r=6, dashed=dashed, kind="card")
        if s:
            size = size or self.fs("label")
            self.text(x + fw / 2, y + off + fh / 2 + size * 0.36, s, size=size, color=p.deep, anchor="middle",
                      box=front, family=family, italic=italic)
        if repeat:
            rs = self.fs("min")
            self.text(x + w, y - 4, repeat, size=rs, color=p.deep, anchor="end")
        self.stack_front = front  # the front face, for a label or marker placed by the caller
        # the box spans the whole width (a wire leaving right starts clear of the back sheets) but only the
        # front face's height, so stacks with different layer counts still line up on their front faces
        return self._box("card", x, y + off, w, h - off)

    def cuboid(self, x, y, w, h, d, role, fill=None, shade=0.0):
        """A block with depth: front face w x h at (x, y + d/2), receding up and to the right by d.
        The footprint is (x, y, w + 0.7 d, h + 0.5 d). Returns the front face's box id."""
        p = PAL[role]
        dx, dy = 0.7 * d, 0.5 * d
        fy = y + dy
        face = _mix(fill or p.mid, p.accent, shade)
        top = _mix(face, "#FFFFFF", 0.35)
        side = _mix(face, p.deep, 0.25)
        self.add(f'<path d="M{x} {fy:.1f}L{x + dx:.1f} {y:.1f}H{x + w + dx:.1f}L{x + w} {fy:.1f}Z" fill="{top}" '
                 f'stroke="{p.accent}" stroke-width="0.8" stroke-linejoin="round"/>')
        self.add(f'<path d="M{x + w} {fy:.1f}L{x + w + dx:.1f} {y:.1f}V{y + h:.1f}L{x + w} {fy + h:.1f}Z" '
                 f'fill="{side}" stroke="{p.accent}" stroke-width="0.8" stroke-linejoin="round"/>')
        self.add(f'<rect x="{x}" y="{fy:.1f}" width="{w}" height="{h}" fill="{face}" stroke="{p.accent}" '
                 f'stroke-width="0.8"/>')
        return self._box("card", x, fy, w, h)

    def feature_maps(self, x, cy, maps, role, gap=10, depth=0.45):
        """CNN feature maps as a row of blocks, one (thickness, side) pair each: the front face is thickness
        wide and side tall, the depth is `depth` x side, and every block is centred on the line cy. Returns
        the box ids from left to right."""
        ids = []
        for i, (thick, side) in enumerate(maps):
            d = depth * side
            ids.append(self.cuboid(x, cy - (side + 0.5 * d) / 2, thick, side, d, role, shade=0.12 * i))
            x += thick + 0.7 * d + gap
        return ids

    def mlp(self, x, y, w, h, layers, role, r=5, edge=HAIR, fill="#FFFFFF"):
        """A fully connected network as a node-link diagram: one column of nodes per layer, every node joined
        to every node of the next layer. Returns the box id of the whole drawing."""
        n = len(layers)
        xs = [x + r + (w - 2 * r) * i / max(1, n - 1) for i in range(n)]
        cols = []
        for cx, k in zip(xs, layers):
            step = (h - 2 * r) / max(1, k - 1) if k > 1 else 0
            top = y + r + ((h - 2 * r) - step * (k - 1)) / 2
            cols.append([(cx, top + i * step) for i in range(k)])
        lines = "".join(f"M{ax:.1f} {ay:.1f}L{bx:.1f} {by:.1f}" for a, b in itertools.pairwise(cols)
                        for ax, ay in a for bx, by in b)
        self.add(f'<path data-qa="ignore" d="{lines}" fill="none" stroke="{edge}" stroke-width="0.8"/>')
        p = PAL[role]
        for i, col in enumerate(cols):
            shade = p.tint if 0 < i < n - 1 else p.mid
            for cx, cy in col:
                self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{shade if fill is None else fill}" '
                         f'stroke="{p.accent}" stroke-width="1"/>')
        return self._box("chip", x, y, w, h)

    def _box(self, kind, x, y, w, h):
        """Register an invisible QA box around a drawing so overlap, coverage and anchoring see it."""
        cid = self.uid("nn")
        self.add(f'<rect data-box="{cid}" data-kind="{kind}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" fill="none" stroke="none"/>')
        return self._reg(cid, x, y, w, h)

    # -- shapes with meaning ----------------------------------------------------
    def tokens(self, x, y, n, role, w=16, h=8, gap=4, lit=None, dashed=False, to_role=None):
        """Row of token pills (a sequence). `lit` pills use the role color, the rest stay neutral;
        `to_role` blends the fill across the row.

        The row is registered as a box, so a row wider than its card is reported by the gate instead of
        quietly hanging over the card's edge. Returns the row width.
        """
        p = PAL[role]
        q = PAL[to_role] if to_role else None
        for i in range(n):
            on = lit is None or i < lit
            fillc = (_mix(p.mid, q.mid, i / max(1, n - 1)) if q else p.mid) if on else "#EEECE6"
            st = p.accent if on else "#C9C5BA"
            dash = ' stroke-dasharray="2 1.5"' if dashed else ""
            self.add(f'<rect x="{x + i * (w + gap):.1f}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{fillc}" '
                     f'stroke="{st}" stroke-width="0.9"{dash}/>')
        width = n * w + (n - 1) * gap
        cid = self.uid("tk")
        self.add(f'<rect data-box="{cid}" data-kind="chip" data-qa="ignore" x="{x}" y="{y}" width="{width:.1f}" '
                 f'height="{h}" rx="{h / 2}" fill="none" stroke="none"/>')
        self._reg(cid, x, y, width, h)
        return width

    def trapezoid(self, x, y, w, h, role, s=None, size=None, inset=0.16, wide_bottom=True, family="sans", italic=False,
                  direction=None, dashed=False, fill=None):
        """Encoder or decoder shape. `direction` names the narrow end: "up" or "down" for a vertical stack
        (the default follows `wide_bottom`), "right" for an encoder in a left-to-right flow, "left" for a
        decoder. `dashed` marks a copy that gradients do not train, such as an EMA target encoder."""
        size = size or self.fs("module")
        p = PAL[role]
        direction = direction or ("up" if wide_bottom else "down")
        if direction in ("up", "down"):
            d = w * inset
            pts = (f"{x + d},{y} {x + w - d},{y} {x + w},{y + h} {x},{y + h}" if direction == "up"
                   else f"{x},{y} {x + w},{y} {x + w - d},{y + h} {x + d},{y + h}")
        elif direction in ("right", "left"):
            d = h * inset
            pts = (f"{x},{y} {x + w},{y + d} {x + w},{y + h - d} {x},{y + h}" if direction == "right"
                   else f"{x},{y + d} {x + w},{y} {x + w},{y + h} {x},{y + h - d}")
        else:
            raise ValueError("direction must be up, down, left or right")
        cid = self.uid("t")
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.add(f'<polygon data-box="{cid}" data-kind="card" points="{pts}" fill="{fill or p.tint}" '
                 f'stroke="{p.accent}" stroke-width="1.1"{dash}/>')
        self._reg(cid, x, y, w, h)
        if s:
            self.text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, color=p.deep, anchor="middle", box=cid,
                      family=family, italic=italic)
        return cid

    def bracket(self, x, y, w, h, color=INK, sw=1.1, tick=5):
        """Square brackets around a region (a vector or a grouped output)."""
        self.add(f'<path d="M{x + tick} {y}H{x}V{y + h}H{x + tick}M{x + w - tick} {y}H{x + w}V{y + h}H{x + w - tick}" '
                 f'fill="none" stroke="{color}" stroke-width="{sw}"/>')

    def cylinder(self, x, y, w, h, role, s=None, size=None, family="serif", italic=True):
        """Stored data (history, dataset)."""
        size = size or self.fs("label")
        p = PAL[role]
        ry = min(6.0, h * 0.18)
        cid = self.uid("cy")
        self.add(
            f'<path data-box="{cid}" data-kind="card" d="M{x} {y + ry}A{w / 2} {ry} 0 0 1 {x + w} {y + ry}V{y + h - ry}'
            f'A{w / 2} {ry} 0 0 1 {x} {y + h - ry}Z" fill="{p.tint}" stroke="{p.accent}" stroke-width="1"/>'
        )
        self.add(f'<path d="M{x} {y + ry}A{w / 2} {ry} 0 0 0 {x + w} {y + ry}" fill="none" stroke="{p.accent}" stroke-width="1"/>')
        self._reg(cid, x, y, w, h)
        if s:
            self.text(x + w / 2, y + h / 2 + ry / 2 + size * 0.36, s, size=size, color=p.deep, anchor="middle", box=cid,
                      family=family, italic=italic)
        return cid

    def block_arrow(self, x1, y1, x2, y2, width=12, color="#D6D2C4", head=None):
        """Thick chevron arrow for a stage transition (horizontal or vertical)."""
        head = head or width * 1.25
        if abs(y2 - y1) < 1e-6:
            s = 1 if x2 > x1 else -1
            xb = x2 - s * head
            pts = f"{x1},{y1 - width / 2} {xb},{y1 - width / 2} {xb},{y1 - width} {x2},{y1} {xb},{y1 + width} {xb},{y1 + width / 2} {x1},{y1 + width / 2}"
        else:
            s = 1 if y2 > y1 else -1
            yb = y2 - s * head
            pts = f"{x1 - width / 2},{y1} {x1 - width / 2},{yb} {x1 - width},{yb} {x1},{y2} {x1 + width},{yb} {x1 + width / 2},{yb} {x1 + width / 2},{y1}"
        self.add(f'<polygon points="{pts}" fill="{color}"/>')

    def bubble(self, x, y, w, h, s, role="gray", size=None, tail="left", family="serif", italic=True, weight=400, stroke=None):
        """Speech bubble for language (instructions, model utterances). Its text counts as example content."""
        size = size or self.fs("label")
        p = PAL[role]
        r = 8
        cy = y + h / 2
        if tail == "left":
            d = (f"M{x + r} {y}H{x + w - r}A{r} {r} 0 0 1 {x + w} {y + r}V{y + h - r}A{r} {r} 0 0 1 {x + w - r} {y + h}"
                 f"H{x + r}A{r} {r} 0 0 1 {x} {y + h - r}V{cy + 5}L{x - 9} {cy}L{x} {cy - 5}V{y + r}A{r} {r} 0 0 1 {x + r} {y}Z")
        else:
            d = (f"M{x + r} {y}H{x + w - r}A{r} {r} 0 0 1 {x + w} {y + r}V{y + h - r}A{r} {r} 0 0 1 {x + w - r} {y + h}"
                 f"H{x + r}A{r} {r} 0 0 1 {x} {y + h - r}V{y + r}A{r} {r} 0 0 1 {x + r} {y}Z")
        cid = self.uid("bb")
        self.add(f'<path data-box="{cid}" data-kind="example" d="{d}" fill="#FFFFFF" stroke="{stroke or p.accent}" stroke-width="1.1"/>')
        self._reg(cid, x, y, w, h)
        self.text(x + w / 2, cy + size * 0.36, s, size=size, color=p.deep, anchor="middle", box=cid, family=family,
                  italic=italic, weight=weight)
        return cid

    # -- glyphs --------------------------------------------------------------
    def icon(self, name, x, y, size, color):
        sid = f"i-{name}"
        if sid not in self.defs:
            self.defs[sid] = f'<symbol id="{sid}" viewBox="0 0 24 24" class="ic">{ICONS[name]}</symbol>'
        self.add(f'<use href="#{sid}" x="{x}" y="{y}" width="{size}" height="{size}" color="{color}"/>')

    def asset(self, path, x, y, size, color=INK, sw=1.5):
        """Place a vendored open-source SVG (see svgicons.py) as a reusable symbol.

        `size` is the display width and the height follows the viewBox. `sw` is the display stroke width,
        so outline icons keep one optical weight at any size; `color` feeds currentColor, and `color=None`
        keeps a colour illustration's own paint (Fluent Emoji). The icon gets a QA box, so it is checked for
        overlaps and counted in coverage. Returns that box id.
        """
        tint = color is not None
        key = (str(Path(path).resolve()), tint)
        if key not in self._assets:
            self._assets[key] = load_svg_asset(path, tint)
        a = self._assets[key]
        sid = f"as-{a.slug}" + ("" if tint else "-c")
        vx, vy, vw, vh = a.view_box
        if sid not in self.defs:
            self.defs[sid] = f'<symbol id="{sid}" viewBox="{vx:g} {vy:g} {vw:g} {vh:g}">{a.body}</symbol>'
        h = size * vh / vw
        self.add(f'<use href="#{sid}" x="{x}" y="{y}" width="{size}" height="{h:.2f}" color="{color or INK}" '
                 f'stroke-width="{sw * vw / size:.3f}"/>')
        bid = self.uid("ic")
        self.add(f'<rect data-box="{bid}" data-kind="icon" x="{x}" y="{y}" width="{size}" height="{h:.2f}" '
                 f'fill="none" stroke="none"/>')
        return bid

    def image(self, path, x, y, w, h, fit="cover", r=3, frame=HAIR):
        """Embed a real render, photo or data crop (PNG, JPEG or WebP) clipped to a rounded frame.

        fit: "cover" fills the frame and crops, "contain" letterboxes, "stretch" ignores the aspect ratio.
        The bytes are inlined as a data URI, so the SVG stays self-contained. Returns the frame's QA box id.
        """
        aspect = {"cover": "xMidYMid slice", "contain": "xMidYMid meet", "stretch": "none"}[fit]
        mime, data = load_raster(path)
        clip = self.uid("clip")
        self.defs[clip] = f'<clipPath id="{clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath>'
        uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        self.add(f'<g clip-path="url(#{clip})"><image href="{uri}" x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'preserveAspectRatio="{aspect}"/></g>')
        cid = self.uid("img")
        stroke = f'stroke="{frame}" stroke-width="1"' if frame else 'stroke="none"'
        self.add(f'<rect data-box="{cid}" data-kind="thumb" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" {stroke}/>')
        return self._reg(cid, x, y, w, h)

    def scene(self, x, y, w, h, frame=None, marker=None):
        if "scene" not in self.defs:
            self.defs["scene"] = SCENE
        cid = self.uid("s")
        clip = self.uid("clip")
        self.defs[clip] = f'<clipPath id="{clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2"/></clipPath>'
        self.add(f'<g clip-path="url(#{clip})"><use href="#scene" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/></g>')
        self.add(f'<rect data-box="{cid}" data-kind="thumb" x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" '
                 f'stroke="{frame or "#B9B4A8"}" stroke-width="1"/>')
        return self._reg(cid, x, y, w, h)

    def arrow(self, d, color=WIRE, sw=1.3, dashed=False, start=False, end=True, head=7.0, opacity=None, open_=False):
        mk = self._marker(color, head, open_)
        attrs = f'd="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'
        if dashed:
            attrs += ' stroke-dasharray="4 3"'
        if end:
            attrs += f' marker-end="url(#{mk})"'
        if start:
            attrs += f' marker-start="url(#{mk})"'
        if opacity is not None:
            attrs += f' stroke-opacity="{opacity}"'
        self.add(f"<path {attrs}/>")

    # -- ports and tidy connectors -------------------------------------------
    def port(self, box, side, t=0.5, out=0.0) -> tuple[float, float]:
        """A point on a container's edge. `side` is left, right, top or bottom, `t` the fraction along that
        edge (0.5 is the middle) and `out` an outward offset, so an arrowhead can stop just off the stroke."""
        x, y, w, h = self.rect(box)
        if side == "left":
            return (x - out, y + h * t)
        if side == "right":
            return (x + w + out, y + h * t)
        if side == "top":
            return (x + w * t, y - out)
        if side == "bottom":
            return (x + w * t, y + h + out)
        raise ValueError("side must be left, right, top or bottom")

    @staticmethod
    def _auto_sides(ra, rb) -> tuple[str, str]:
        ax, ay = ra[0] + ra[2] / 2, ra[1] + ra[3] / 2
        bx, by = rb[0] + rb[2] / 2, rb[1] + rb[3] / 2
        if abs(bx - ax) >= abs(by - ay):
            return ("right", "left") if bx >= ax else ("left", "right")
        return ("bottom", "top") if by >= ay else ("top", "bottom")

    def connect(self, a, b, sides=None, color=WIRE, sw=1.4, dashed=False, open_=False, start=False, head=7.5,
                ta=0.5, tb=0.5, gap=1.0, mid=None, label=None, label_color=None, label_size=None,
                label_family="serif", knockout=False):
        """Arrow between two containers, anchored exactly on their edges.

        Sides are chosen from the boxes' relative position unless given as (from_side, to_side). The path is
        a straight line when the two ports line up, otherwise an orthogonal elbow through `mid` (the shared
        x for a left-right pair, the shared y for a top-bottom pair). `label` is placed beside the longest
        segment, or on it with a white knock-out when knockout=True. Returns the path data.

        `ta=None` (or `tb=None`) leaves that end free so it lines up with the other box's port, which is how
        a tall container connects straight to a short card without hand-computing the fraction.
        """
        ra, rb = self.rect(a), self.rect(b)
        sa, sb = sides or self._auto_sides(ra, rb)
        x0, y0 = self.port(a, sa, 0.5 if ta is None else ta, out=gap)
        x1, y1 = self.port(b, sb, 0.5 if tb is None else tb, out=gap)
        horiz_a, horiz_b = sa in ("left", "right"), sb in ("left", "right")
        if ta is None:
            if horiz_a:
                y0 = min(max(y1, ra[1]), ra[1] + ra[3])
            else:
                x0 = min(max(x1, ra[0]), ra[0] + ra[2])
        if tb is None:
            if horiz_b:
                y1 = min(max(y0, rb[1]), rb[1] + rb[3])
            else:
                x1 = min(max(x0, rb[0]), rb[0] + rb[2])
        if horiz_a and horiz_b:
            if abs(y1 - y0) < 0.75:
                d, seg = f"M{x0:.1f} {y0:.1f}H{x1:.1f}", ((x0, y0), (x1, y0), "h")
            else:
                mx = mid if mid is not None else round((x0 + x1) / 2)
                d = f"M{x0:.1f} {y0:.1f}H{mx:.1f}V{y1:.1f}H{x1:.1f}"
                seg = ((mx, y0), (mx, y1), "v")
        elif not horiz_a and not horiz_b:
            if abs(x1 - x0) < 0.75:
                d, seg = f"M{x0:.1f} {y0:.1f}V{y1:.1f}", ((x0, y0), (x0, y1), "v")
            else:
                my = mid if mid is not None else round((y0 + y1) / 2)
                d = f"M{x0:.1f} {y0:.1f}V{my:.1f}H{x1:.1f}V{y1:.1f}"
                seg = ((x0, my), (x1, my), "h")
        elif horiz_a:
            d, seg = f"M{x0:.1f} {y0:.1f}H{x1:.1f}V{y1:.1f}", ((x0, y0), (x1, y0), "h")
        else:
            d, seg = f"M{x0:.1f} {y0:.1f}V{y1:.1f}H{x1:.1f}", ((x0, y0), (x0, y1), "v")
        self.arrow(d, color, sw, dashed=dashed, open_=open_, start=start, head=head)
        if label:
            (sx, sy), (ex, ey), kind = seg
            size = label_size or self.fs("min")
            mxp, myp = (sx + ex) / 2, (sy + ey) / 2
            if knockout:
                self.pill(mxp, myp, label, size=size, family=label_family, italic=label_family == "serif")
            elif kind == "h":
                self.text(mxp, myp - 7, label, size=size, color=label_color or MUTED, anchor="middle",
                          family=label_family, italic=label_family == "serif")
            else:
                self.text(mxp + 8, myp + size * 0.36, label, size=size, color=label_color or MUTED,
                          family=label_family, italic=label_family == "serif")
        return d

    def bus(self, src, targets, side="bottom", at=None, color=WIRE, sw=1.3, head=6.5, ta=0.5, gap=1.0,
            face=None):
        """Fork one container into several with a shared trunk: a stem, one trunk line, one arrow per target.

        `side` is the side of `src` the stem leaves from; the trunk sits at `at` (a y for bottom/top, an x
        for left/right) or halfway between the boxes. Tidier than one long arrow per target.
        `face` overrides which side of the targets the arrows enter, for a trunk that leaves upwards and
        comes back down into targets that sit level with the source.
        """
        sx, sy = self.port(src, side, ta, out=gap)
        rects = [self.rect(t) for t in targets]
        if side in ("bottom", "top"):
            face = face or ("top" if side == "bottom" else "bottom")
            edge = min(r[1] for r in rects) if side == "bottom" else max(r[1] + r[3] for r in rects)
            trunk = at if at is not None else round((sy + edge) / 2)
            xs = [r[0] + r[2] / 2 for r in rects]
            self.line(f"M{sx:.1f} {sy:.1f}V{trunk:.1f}", color, sw)
            self.line(f"M{min(xs + [sx]):.1f} {trunk:.1f}H{max(xs + [sx]):.1f}", color, sw)
            self.dot(sx, trunk, color, 2.4)
            for t, x in zip(targets, xs):
                tx, ty = self.port(t, face, 0.5, out=gap)
                self.arrow(f"M{x:.1f} {trunk:.1f}V{ty:.1f}", color, sw, head=head)
        else:
            face = face or ("left" if side == "right" else "right")
            edge = min(r[0] for r in rects) if side == "right" else max(r[0] + r[2] for r in rects)
            trunk = at if at is not None else round((sx + edge) / 2)
            ys = [r[1] + r[3] / 2 for r in rects]
            self.line(f"M{sx:.1f} {sy:.1f}H{trunk:.1f}", color, sw)
            self.line(f"M{trunk:.1f} {min(ys + [sy]):.1f}V{max(ys + [sy]):.1f}", color, sw)
            self.dot(trunk, sy, color, 2.4)
            for t, y in zip(targets, ys):
                tx, ty = self.port(t, face, 0.5, out=gap)
                self.arrow(f"M{trunk:.1f} {y:.1f}H{tx:.1f}", color, sw, head=head)
        return trunk

    def curve(self, a, b, ta=0.5, tb=0.5, up=False, color=WIRE, sw=1.3, dashed=False, head=7.0, gap=1.0):
        """Wire between two stacked layers that leaves and enters vertically: a straight line when the ports line
        up, otherwise a smooth S. Use it for many-to-many mappings between rows (nodes to contracts, policies to
        contracts): give a box that takes several wires spread ports (`ta`/`tb` of 0.2, 0.5, 0.8), mirror them
        between the two halves of the figure, and order the boxes so no two wires cross.
        `up=True` runs from the top of `a` to the bottom of `b`. Returns the path data."""
        x0, y0 = self.port(a, "top" if up else "bottom", ta, out=gap)
        x1, y1 = self.port(b, "bottom" if up else "top", tb, out=gap)
        if abs(x0 - x1) < 0.5:
            d = f"M{x0:.1f} {y0:.1f}V{y1:.1f}"
        else:
            ym = (y0 + y1) / 2
            d = f"M{x0:.1f} {y0:.1f}C{x0:.1f} {ym:.1f} {x1:.1f} {ym:.1f} {x1:.1f} {y1:.1f}"
        self.arrow(d, color, sw, dashed=dashed, head=head)
        return d

    def arc(self, a, b, sides=None, bulge=40, color=FAINT, sw=1.3, dashed=True, head=7.0, label=None,
            label_color=None, label_size=None, ta=0.5, tb=0.5, gap=1.0, knockout=False):
        """Curved connector for a feedback or return path: one quadratic bend of `bulge` px, sign picks the side.

        A label sits just outside the bend, never on the curve, and flips to the inside when the outside
        would leave the canvas; `knockout=True` puts it on the apex in a white pill instead.
        """
        ra, rb = self.rect(a), self.rect(b)
        sa, sb = sides or self._auto_sides(ra, rb)
        x0, y0 = self.port(a, sa, ta, out=gap)
        x1, y1 = self.port(b, sb, tb, out=gap)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / length, dx / length
        cx, cy = mx + nx * bulge, my + ny * bulge
        self.arrow(f"M{x0:.1f} {y0:.1f}Q{cx:.1f} {cy:.1f} {x1:.1f} {y1:.1f}", color, sw, dashed=dashed, head=head)
        if label:
            size = label_size or self.fs("min")
            apex = ((mx + cx) / 2, (my + cy) / 2)
            if knockout:
                self.pill(apex[0], apex[1], label, size=size, family="serif", italic=True)
            else:
                out = (size * 0.9 + 4) * (1 if bulge >= 0 else -1)
                lx, ly = apex[0] + nx * out, apex[1] + ny * out
                half = text_w(label, size) / 2
                if lx - half < 4 or lx + half > self.w - 4 or ly - size < 4 or ly + size > self.h - 4:
                    lx, ly = apex[0] - nx * out, apex[1] - ny * out
                self.text(lx, ly + size * 0.36, label, size=size, color=label_color or MUTED, anchor="middle",
                          family="serif", italic=True)
        return (cx, cy)

    def route(self, a, b, lanes, sides=None, color=WIRE, sw=1.4, dashed=False, open_=False, start=False,
              head=7.5, ta=0.5, tb=0.5, gap=1.0, label=None, label_color=None, label_size=None,
              label_family="serif", label_seg=None, label_at=None, knockout=False):
        """Orthogonal connector that wraps around content through explicit lanes.

        `lanes` holds one coordinate per bend, alternating with the direction the wire leaves `a`: a y for a
        top or bottom departure, an x for a left or right one. The wire always arrives perpendicular to b's
        side, so a feedback path around a column is
        `route(last, first, (lane_y, lane_x), sides=("bottom", "left"))`.
        `label_seg` picks which segment carries the label (default the longest) and `label_at` where along it the
        label is centred (an x on a horizontal segment, a y on a vertical one; default its midpoint), so a label on
        a lane that runs under several panels can sit inside one of them instead of across a panel edge.
        Returns the path data.
        """
        ra, rb = self.rect(a), self.rect(b)
        sa, sb = sides or self._auto_sides(ra, rb)
        x0, y0 = self.port(a, sa, ta, out=gap)
        x1, y1 = self.port(b, sb, tb, out=gap)
        cur = [x0, y0]
        parts, segs = [f"M{x0:.1f} {y0:.1f}"], []

        def step(val, vertical):
            if abs(val - cur[1 if vertical else 0]) < 0.4:
                return
            p0 = (cur[0], cur[1])
            if vertical:
                cur[1] = val
                parts.append(f"V{val:.1f}")
            else:
                cur[0] = val
                parts.append(f"H{val:.1f}")
            segs.append((p0, (cur[0], cur[1]), "v" if vertical else "h"))

        vertical = sa in ("top", "bottom")
        for i, v in enumerate(lanes):
            step(v, vertical if i % 2 == 0 else not vertical)
        if sb in ("left", "right"):
            step(y1, True)
            step(x1, False)
        else:
            step(x1, False)
            step(y1, True)
        d = "".join(parts)
        self.arrow(d, color, sw, dashed=dashed, open_=open_, start=start, head=head)
        if label and segs:
            i = label_seg if label_seg is not None else max(
                range(len(segs)), key=lambda k: abs(segs[k][1][0] - segs[k][0][0]) + abs(segs[k][1][1] - segs[k][0][1]))
            (sx, sy), (ex, ey), kind = segs[i]
            size = label_size or self.fs("min")
            mxp, myp = (sx + ex) / 2, (sy + ey) / 2
            if label_at is not None:
                lo, hi = sorted((sx, ex) if kind == "h" else (sy, ey))
                if not lo <= label_at <= hi:
                    raise ValueError(f"label_at={label_at:g} is off the labelled segment ({lo:g} to {hi:g})")
                mxp, myp = (label_at, myp) if kind == "h" else (mxp, label_at)
            if knockout:
                self.pill(mxp, myp, label, size=size, family=label_family, italic=label_family == "serif")
            elif kind == "h":
                self.text(mxp, myp - 7, label, size=size, color=label_color or MUTED, anchor="middle",
                          family=label_family, italic=label_family == "serif")
            else:
                self.text(mxp + 8, myp + size * 0.36, label, size=size, color=label_color or MUTED,
                          family=label_family, italic=label_family == "serif")
        return d

    def dot(self, x, y, color=WIRE, r=2.4):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def brace(self, x, y1, y2, color=FAINT, depth=12, sw=1.2):
        """Right-pointing curly brace spanning y1..y2 at x (opens to the left)."""
        m = (y1 + y2) / 2
        d = depth / 2
        self.add(
            f'<path d="M{x} {y1}C{x + d} {y1} {x + d} {y1 + 6} {x + d} {y1 + 14}V{m - 10}C{x + d} {m - 3} {x + depth} {m} {x + depth} {m}'
            f'C{x + depth} {m} {x + d} {m + 3} {x + d} {m + 10}V{y2 - 14}C{x + d} {y2 - 6} {x + d} {y2} {x} {y2}" '
            f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    def line(self, d, color=HAIR, sw=1.0, dashed=False):
        dash = ' stroke-dasharray="3 3"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash}/>')

    # -- data sketches -------------------------------------------------------
    def bars(self, x, y, w, h, values, color, gap=2.0, base="#B9B4A8", box=None):
        n = len(values)
        bw = (w - gap * (n - 1)) / n
        self.line(f"M{x} {y + h}H{x + w}", base, 1)
        for i, v in enumerate(values):
            bh = max(1.5, v * h)
            self.add(f'<rect x="{x + i * (bw + gap):.1f}" y="{y + h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                     f'fill="{color}" opacity="{0.5 + 0.4 * v:.2f}"/>')

    def curves(self, x, y, w, h, colors, seed=1.0, grid=True, n=60, sw=1.4):
        if grid:
            for k in range(1, 4):
                self.line(f"M{x} {y + h * k / 4:.1f}H{x + w}", "#ECEAE4", 0.8)
            self.line(f"M{x} {y}V{y + h}H{x + w}", "#9C968A", 0.9)
        for j, c in enumerate(colors):
            pts = []
            for i in range(n + 1):
                t = i / n
                v = 0.5 + 0.28 * math.sin(2.2 * t * math.pi + seed * (j + 1) * 1.7) + 0.12 * math.sin(5.3 * t * math.pi + j * 0.9 + seed)
                pts.append(f"{x + t * w:.1f} {y + (1 - v) * h:.1f}")
            self.add(f'<path d="M{"L".join(pts)}" fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round"/>')

    def strip(self, x, y, w, h, n, lit, color, off="#ECEAE4", gap=1.0):
        cw = (w - gap * (n - 1)) / n
        for i in range(n):
            fill = color if i < lit else off
            self.add(f'<rect x="{x + i * (cw + gap):.2f}" y="{y}" width="{cw:.2f}" height="{h}" rx="0.6" fill="{fill}"/>')
