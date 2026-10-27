#!/usr/bin/env python3
"""Build main.md / main.tex / S*_Text.{md,tex} / S1_Table.{md,tex} from the *.src.md sources.

Source syntax (a small Markdown subset):
  # title, ## section, ### subsection, #### subsubsection
  paragraphs, "- " bullet lists, "1. " numbered lists, ``` code blocks, pipe tables
  **bold**, *italic*, `code`, citations [@key; @key2]
  @@AUTHORS@@, @@BOX@@ ... @@ENDBOX@@, @@FIG name@@, @@TABLE name@@, @@SUPP ...@@, @@REFERENCES@@
Usage: python3 build.py [--no-embed]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from refs import REFS, vancouver, bibtex  # noqa: E402

EMBED = "--no-embed" not in sys.argv

AUTHOR_PLACEHOLDER = "[Author list to be completed by the submitting authors]"
AI_STATEMENT = (
    "**AI-contribution statement.** All content of this article (study design, code, data curation, "
    "analyses, figures and text) was produced by AI agents: a principal-investigator agent and the "
    "sub-agents that it launched (Claude Code; Claude Opus 5.5 and Claude Sonnet 5.5; Anthropic). "
    "The human authors set up the computing environment, wrote the initial prompt, granted one time "
    "extension, asked three follow-up questions, requested this manuscript, and take responsibility "
    "for its submission. See the box below, Methods and S2 Text."
)

# --------------------------------------------------------------------------- parsing helpers

def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def parse_blocks(text, kind):
    """@@@KIND name\nkey: value\n...\n<body>\n@@@END -> {name: {fields..., 'body': str}}"""
    out = {}
    for m in re.finditer(r"@@@" + kind + r" (\S+)\n(.*?)\n@@@END", text, re.S):
        name, raw = m.group(1), m.group(2)
        fields, body = {}, []
        for line in raw.split("\n"):
            mm = re.match(r"^(title|caption|colspec|note|text):\s?(.*)$", line)
            if mm and not body:
                fields[mm.group(1)] = mm.group(2)
            else:
                body.append(line)
        fields["body"] = "\n".join(body).strip()
        out[name] = fields
    return out


def load_captions():
    """Figure captions: manuscript/figures/CAPTIONS.md if present (parsed leniently), else the draft."""
    caps = parse_blocks(read(os.path.join(HERE, "captions_draft.src.md")), "CAPTION")
    p = os.path.join(MS, "figures", "CAPTIONS.md")
    override = os.path.join(HERE, "captions_final.src.md")
    if os.path.exists(override):
        caps.update(parse_blocks(read(override), "CAPTION"))
    return caps


def fig_file(name):
    for ext in (".pdf", ".png", ".jpg"):
        p = os.path.join(MS, "figures", name + ext)
        if os.path.exists(p):
            return "figures/" + name + ext
    return None


# --------------------------------------------------------------------------- citations

class Citer:
    def __init__(self):
        self.order = []

    def number(self, key):
        if key not in REFS:
            raise KeyError(f"unknown reference key {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    @staticmethod
    def keys(s):
        return [k.strip().lstrip("@") for k in s.split(";")]

    def md(self, text):
        def rep(m):
            nums = sorted(self.number(k) for k in self.keys(m.group(1)))
            parts, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
                i = j + 1
            return "[" + ", ".join(parts) + "]"
        return re.sub(r"\[(@[^\]]+)\]", rep, text)


CIT_RE = re.compile(r"\[(@[^\]]+)\]")

# --------------------------------------------------------------------------- inline TeX

UNI = {
    "≤": r"$\le$", "≥": r"$\ge$", "−": r"$-$", "×": r"$\times$", "≈": r"$\approx$", "±": r"$\pm$",
    "→": r"$\rightarrow$", "Å": r"\AA{}", "³": r"$^{3}$", "²": r"$^{2}$", "⁻": r"$^{-}$",
    "β": r"$\beta$", "λ": r"$\lambda$", "τ": r"$\tau$", "ρ": r"$\rho$", "Δ": r"$\Delta$", "α": r"$\alpha$",
    "μ": r"$\mu$", "µ": r"$\mu$", "–": "--", "—": "---", "…": r"\ldots{}", "<": r"\textless{}",
    ">": r"\textgreater{}", "|": r"\textbar{}", "·": r"$\cdot$", "∪": r"$\cup$", "°": r"$^\circ$",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    out = []
    for ch in s:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNI:
            out.append(UNI[ch])
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"\$\^\{-\}\$\$\^\{3\}\$", r"$^{-3}$", s)  # 10⁻³
    return s


def tex_inline(s):
    keep = []

    def stash(v):
        keep.append(v)
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + esc(m.group(1)) + "}"), s)
    s = CIT_RE.sub(lambda m: stash("~\\cite{" + ",".join(Citer.keys(m.group(1))) + "}"), s)
    s = re.sub(r"(https?://\S+?)([.,;)]?)(?=\s|$)", lambda m: stash(r"\url{" + m.group(1) + "}") + m.group(2), s)
    s = re.sub(r'"([^"]+)"', lambda m: "\x01" + m.group(1) + "\x02", s)
    s = esc(s)
    s = s.replace("\x01", "``").replace("\x02", "''")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r" ~\\cite", r"~\\cite", s)
    for _ in range(3):
        s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


# --------------------------------------------------------------------------- table rendering

def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_pipe_table(body):
    lines = [l for l in body.split("\n") if l.strip().startswith("|")]
    header = split_row(lines[0])
    rows = [split_row(l) for l in lines[2:]]
    return header, rows


def table_tex(num, t, label):
    header, rows = parse_pipe_table(t["body"])
    colspec = t.get("colspec") or ("l" * len(header))
    out = [r"\begin{table}[!ht]", r"\begin{adjustwidth}{-0.0in}{0in}", r"\centering",
           r"\caption{{\bf " + tex_inline(t["title"]) + "} " + tex_inline(t.get("caption", "")) + "}",
           r"\label{" + label + "}", r"{\footnotesize", r"\begin{tabular}{" + colspec + "}", r"\hline",
           " & ".join(r"\textbf{" + tex_inline(h) + "}" for h in header) + r" \\ \thickhline"]
    for r in rows:
        out.append(" & ".join(tex_inline(c) for c in r) + r" \\ \hline")
    out += [r"\end{tabular}}", r"\end{adjustwidth}"]
    if t.get("note"):
        out.append(r"\begin{flushleft}" + tex_inline(t["note"]) + r"\end{flushleft}")
    out.append(r"\end{table}")
    return "\n".join(out)


def table_md(num, t, citer, prefix="Table"):
    head = f"**{prefix} {num}. {t['title']}** {t.get('caption', '')}".strip()
    body = t["body"]
    s = head + "\n\n" + body + ("\n\n" + t["note"] if t.get("note") else "")
    return citer.md(s)


# --------------------------------------------------------------------------- document conversion

FIG_ORDER = ["fig1_pipeline", "fig2_ligysis", "fig3_prereg", "fig4_composition_tradeoff", "fig5_speed", "fig6_af_dimers"]
TABLE_ORDER = ["origins", "benchmarks", "prereg", "agents", "decisions"]


def convert(src, tables, caps, citer, kind="main"):
    """Return (md, tex_body)."""
    lines = src.split("\n")
    md, tex = [], []
    i = 0
    para = []
    title = None
    in_box = False

    def flush():
        nonlocal para
        if para:
            text = " ".join(l.strip() for l in para)
            md.append(citer.md(text) + "\n")
            tex.append(tex_inline(text) + "\n")
            para = []

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            md.append("```\n" + "\n".join(block) + "\n```\n")
            tex.append("\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\footnotesize]\n" + "\n".join(block) + "\n\\end{Verbatim}\n")
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("# ") and title is None:
            flush()
            title = s[2:]
            md.append("# " + title + "\n")
            i += 1
            continue
        m = re.match(r"^(#{2,4}) (.*)$", s)
        if m:
            flush()
            level, head = len(m.group(1)), m.group(2)
            md.append(m.group(1) + " " + head + "\n")
            cmd = {2: "section*", 3: "subsection*", 4: "subsubsection*"}[level]
            if head == "References":
                pass
            elif kind == "main" and head == "Introduction":
                tex.append("\\clearpage\n\\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}\n\\linenumbers\n")
                tex.append(f"\\{cmd}{{{tex_inline(head)}}}\n")
            elif head in ("Abstract", "Author summary") and kind == "main":
                tex.append(f"\\{cmd}{{{tex_inline(head)}}}\n")
            else:
                tex.append(f"\\{cmd}{{{tex_inline(head)}}}\n")
            i += 1
            continue
        if s == "@@AUTHORS@@":
            flush()
            md.append(AUTHOR_PLACEHOLDER + "\n")
            md.append(AI_STATEMENT + "\n")
            tex.append("\\newline\n\\\\\\relax\n" + tex_inline(AUTHOR_PLACEHOLDER) + "\n\\\\\n\\bigskip\n" + tex_inline(AI_STATEMENT) + "\n\\end{flushleft}\n")
            i += 1
            continue
        if s == "@@BOX@@":
            flush()
            in_box = True
            tex.append("\\bigskip\n\\noindent\\fcolorbox{black}{boxgray}{\\begin{minipage}{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule\\relax}\\small\n")
            i += 1
            continue
        if s == "@@ENDBOX@@":
            # box content is collected as a paragraph
            if para:
                text = " ".join(l.strip() for l in para)
                md.append("> " + citer.md(text) + "\n")
                tex.append(tex_inline(text) + "\n")
                para = []
            tex.append("\\end{minipage}}\n\\bigskip\n")
            in_box = False
            i += 1
            continue
        m = re.match(r"^@@FIG (\S+)@@$", s)
        if m:
            flush()
            name = m.group(1)
            n = FIG_ORDER.index(name) + 1
            c = caps[name]
            f = fig_file(name)
            md.append(citer.md(f"> **Fig {n}. {c['title']}** {c['text']}") + "\n")
            fpng = "figures/" + name + ".png"
            if os.path.exists(os.path.join(MS, fpng)):
                md.append(f"> ![Fig {n}]({fpng})\n")
            fig = ["\\begin{figure}[!h]"]
            if f:
                fig.append("\\ifembedfigures\\centering\\includegraphics[width=\\linewidth,height=0.55\\textheight,keepaspectratio]{" + f + "}\\fi")
            fig.append("\\caption{{\\bf " + tex_inline(c["title"]) + "}\n" + tex_inline(c["text"]) + "}")
            fig.append("\\label{" + name + "}\n\\end{figure}\n")
            tex.append("\n".join(fig))
            i += 1
            continue
        m = re.match(r"^@@TABLE (\S+)@@$", s)
        if m:
            flush()
            name = m.group(1)
            t = tables[name]
            if name in TABLE_ORDER:
                n = TABLE_ORDER.index(name) + 1
                md.append(table_md(n, t, citer) + "\n")
                tex.append(table_tex(n, t, "table_" + name) + "\n")
            else:
                header, rows = parse_pipe_table(t["body"])
                md.append(citer.md(f"**{t['title']}** {t.get('caption', '')}\n\n" + t["body"]) + "\n")
                tex.append("\\noindent{\\bf " + tex_inline(t["title"]) + "} " + tex_inline(t.get("caption", "")) + "\n\n"
                           + "{\\footnotesize\\begin{tabular}{" + t["colspec"] + "}\\hline\n"
                           + " & ".join("\\textbf{" + tex_inline(h) + "}" for h in header) + " \\\\ \\hline\n"
                           + "\n".join(" & ".join(tex_inline(c) for c in r) + " \\\\ \\hline" for r in rows)
                           + "\n\\end{tabular}}\n")
            i += 1
            continue
        m = re.match(r"^@@SUPP ([^|]+)\|([^|]+)\|([^|]+)\|(.*)@@$", s)
        if m:
            flush()
            lab, head, ttl, text = m.groups()
            cm = re.match(r"@@CAPTION (\S+)@@", text)
            if cm:
                c = caps[cm.group(1)]
                text = c["text"]
            md.append(citer.md(f"**{head}** **{ttl}** {text}") + "\n")
            tex.append(f"\\paragraph*{{{head}}}\n\\label{{{lab}}}\n{{\\bf {tex_inline(ttl)}}} {tex_inline(text)}\n")
            i += 1
            continue
        if s == "@@REFERENCES@@":
            flush()
            md.append("@@REFLIST@@\n")
            tex.append("\\nolinenumbers\n\\bibliography{references}\n")
            i += 1
            continue
        mb = re.match(r"^(- |\d+\. )(.*)$", s)
        if mb and not para:
            flush()
            numbered = mb.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                mm = re.match(r"^(- |\d+\. )(.*)$", lines[i].strip())
                if mm:
                    items.append(mm.group(2))
                elif lines[i].strip() and items and lines[i].startswith("  "):
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            for k, it in enumerate(items):
                md.append(("%d. " % (k + 1) if numbered else "- ") + citer.md(it))
            md.append("")
            env = "enumerate" if numbered else "itemize"
            tex.append(f"\\begin{{{env}}}\n" + "\n".join("\\item " + tex_inline(it) for it in items) + f"\n\\end{{{env}}}\n")
            continue
        if s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            t = {"title": "", "body": "\n".join(block)}
            header, rows = parse_pipe_table(t["body"])
            md.append(citer.md("\n".join(block)) + "\n")
            tex.append("{\\footnotesize\\begin{tabular}{" + "p{%.1fcm}" % (15.5 / len(header)) * len(header) + "}\\hline\n"
                       + " & ".join("\\textbf{" + tex_inline(h) + "}" for h in header) + " \\\\ \\hline\n"
                       + "\n".join(" & ".join(tex_inline(c) for c in r) + " \\\\ \\hline" for r in rows)
                       + "\n\\end{tabular}}\n")
            continue
        para.append(line)
        i += 1
    flush()
    return title, "\n".join(md), "\n".join(tex)


PREAMBLE = r"""%% Generated by manuscript/build/build.py from main.src.md -- edit the source, not this file.
%% PLOS LaTeX template (journals.plos.org/ploscompbiol/s/latex), adapted for XeTeX/Tectonic.
\documentclass[10pt,letterpaper]{article}
\usepackage[top=0.85in,left=2.75in,footskip=0.75in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{changepage}
\usepackage{textcomp,marvosym}
\usepackage{cite}
\usepackage{nameref,hyperref}
\usepackage[right]{lineno}
\usepackage{iftex}
\usepackage[nopatch=eqnum]{microtype}
\ifpdftex\DisableLigatures[f]{encoding = *, family = * }\fi
\usepackage[table]{xcolor}
\definecolor{boxgray}{gray}{0.94}
\usepackage{array}
\usepackage{fvextra}
\usepackage{url}
\newcolumntype{+}{!{\vrule width 2pt}}
\newlength\savedwidth
\newcommand\thickcline[1]{%
  \noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
  \cline{#1}%
  \noalign{\vskip\arrayrulewidth}%
  \noalign{\global\arrayrulewidth\savedwidth}%
}
\newcommand\thickhline{\noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
\hline
\noalign{\global\arrayrulewidth\savedwidth}}
\raggedright
\setlength{\parindent}{0.5cm}
\textwidth 5.25in
\textheight 8.75in
\usepackage[aboveskip=1pt,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=off]{caption}
\renewcommand{\figurename}{Fig}
\bibliographystyle{plos2025}
\makeatletter
\renewcommand{\@biblabel}[1]{\quad#1.}
\makeatother
\usepackage{lastpage,fancyhdr,graphicx}
\pagestyle{fancy}
\fancyhf{}
\rfoot{\thepage/\pageref{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrule}{\hrule height 2pt \vspace{2mm}}
\fancyheadoffset[L]{2.25in}
\fancyfootoffset[L]{2.25in}
\lfoot{\today}
%% PLOS asks for figures NOT to be embedded in the submission file; set \embedfiguresfalse for submission.
\newif\ifembedfigures
\embedfigures@@EMBED@@
"""


def write(p, s):
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(s)


def reflist_md(citer):
    return "\n".join(f"{n}. {vancouver(REFS[k])}" for n, k in enumerate(citer.order, 1))


def build_main():
    src = read(os.path.join(HERE, "main.src.md"))
    tables = parse_blocks(read(os.path.join(HERE, "tables.src.md")), "TABLE")
    caps = load_captions()
    citer = Citer()
    title, md, body = convert(src, tables, caps, citer, "main")
    md = md.replace("@@REFLIST@@", reflist_md(citer))
    write(os.path.join(MS, "main.md"), md + "\n")
    tex = PREAMBLE.replace("@@EMBED@@", "true" if EMBED else "false")
    tex += "\\begin{document}\n\\vspace*{0.2in}\n\\begin{flushleft}\n{\\Large\n\\textbf{" + tex_inline(title) + "}\n}\n"
    tex += body + "\n\\end{document}\n"
    write(os.path.join(MS, "main.tex"), tex)
    bib = "".join(bibtex(k, REFS[k]) + "\n" for k in citer.order)
    write(os.path.join(MS, "references.bib"), bib)
    return citer


SUPP_PRE = PREAMBLE.replace("%% Generated by manuscript/build/build.py from main.src.md", "%% Generated by manuscript/build/build.py")


def build_supp(name):
    src = read(os.path.join(HERE, name + ".src.md"))
    tables = parse_blocks(read(os.path.join(HERE, "tables.src.md")), "TABLE")
    tables.update(parse_blocks(src, "TABLE"))
    src = re.sub(r"@@@TABLE .*?@@@END\n?", "", src, flags=re.S)
    caps = load_captions()
    citer = Citer()
    title, md, body = convert(src, tables, caps, citer, "supp")
    write(os.path.join(MS, name + ".md"), md + "\n")
    tex = SUPP_PRE.replace("@@EMBED@@", "false")
    tex = tex.replace("left=2.75in", "left=1in,right=1in").replace("\\textwidth 5.25in\n", "").replace(
        "\\fancyheadoffset[L]{2.25in}\n\\fancyfootoffset[L]{2.25in}\n", "")
    tex += "\\begin{document}\n\\begin{flushleft}\n{\\Large\\textbf{" + tex_inline(title) + "}}\n\\end{flushleft}\n"
    tex += body + "\n\\end{document}\n"
    write(os.path.join(MS, name + ".tex"), tex)


# S1 Table: table blocks named in S1_Table.src.md are rendered without the "Table N" prefix logic
TABLE_ORDER_SUPP = {"roster": "S1 Table"}


if __name__ == "__main__":
    citer = build_main()
    for n in ("S1_Text", "S2_Text", "S1_Table"):
        if os.path.exists(os.path.join(HERE, n + ".src.md")):
            build_supp(n)
    print("references cited:", len(citer.order))
