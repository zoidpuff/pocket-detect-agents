# Open issues for the submitting authors

## References (manuscript/build/refs.py → references.bib)
Items 1–13 of the earlier list (missing initials/titles for zPocket/PocketWatch, HOTPocket, Zillmer & Walther, AF2BIND; unverified author lists/DOIs for PUResNetV2.0, GrASP, VN-EGNN, IF-SitePred, Schmidtke 2010, CryptoBench, BioLiP2, PocketMiner, PRANK) were resolved on 2026-10-27 against Crossref, PubMed/Europe PMC and arXiv. See REFERENCES_CHECK.md for the status of every reference and three wording notes on in-text claims (P2Rank 52.6% vs published 51.9%; LIGYSIS 6,721 vs 6,882 sites; Zillmer & Walther also filter on PAE).

Added beyond LITERATURE.md (standard tools/methods, verifiable): LightGBM (Ke 2017), MMseqs2 (Steinegger & Söding 2017), Foldseek (van Kempen 2024), gemmi (Wojdyr 2022), Numba (Lam 2015), PyTorch (Paszke 2019), Triton (Tillet 2019), NumPy (Harris 2020), SciPy (Virtanen 2020), PDB (Berman 2000), AlphaFold DB (Varadi 2022), AlphaFold2 (Jumper 2021), Holm 1979 (statistics), and Clark, Martin & Mahowald 2026 (arXiv:2610.29162, the source of the writing guide; cited from manuscript/skill/SKILL.md). Please confirm that these are acceptable.

Not cited (no reference available in LITERATURE.md): Claude Code / Claude models (named in text only); weighted contact number (defined explicitly instead of cited); Fibonacci sphere directions; Hungarian matching; Kabsch superposition. Add citations if the journal requires them.

## Content
14. **Follow-up prompts are not verbatim.** S2 Text §9 summarises the human's follow-up messages from AGENT_SETUP.md and RESEARCH_TREE D19/D20; the verbatim text must be pasted from the session transcript. Agent briefs were also not archived in the repository.
15. **Figure-maker model** in S1 Table (row 27) is marked "[to be confirmed]".
16. **Author list, affiliations, corresponding author, funding, competing interests and author contributions** are placeholders. The AI-contribution statement under the author line and the box should be checked against journal policy on AI authorship/disclosure (AI agents cannot be listed as authors).
17. **Repository URL** is the placeholder https://github.com/<to-be-published>/burrow (main text, Data and code availability). The release repository (release/burrow) was still a skeleton when this text was written; check that it contains the models, predictions and scripts that the availability statement promises.
18. **Speed numbers** come from a shared node (±30%); REPORT §6.7 recommends repeating them on an exclusive node before publication.
19. **"More than 130 git commits"** is from `git log` at the time of writing (136 commits, including the first manuscript commit); update if needed.
20. The main text (Introduction–Discussion) is ≈4,060 words and Methods ≈2,710 words (total ≈6,770, excluding tables, captions and references); abstract 295 words; author summary 196 words.

## Format / build
21. `main.tex` follows the PLOS LaTeX template (downloaded from journals.plos.org, plos2025.bst included) adapted for XeTeX/Tectonic (`\DisableLigatures` only under pdfTeX; fvextra, url, iftex added). Figures are embedded for the review PDF via `\embedfigurestrue`; set `\embedfiguresfalse` (or run `python3 build/build.py --no-embed`) for submission, where PLOS wants figures as separate files. PLOS also asks for the .bbl content to be pasted into the .tex at submission; references.bib + plos2025.bst currently generate it.
22. All outputs are generated: edit `build/*.src.md` (text, tables, captions) and `build/refs.py` (references), then run `python3 manuscript/build/build.py` and compile with Tectonic (`tectonic -X compile main.tex`). Hand edits to main.md/main.tex are overwritten.
23. Tables 1, 2, 4 and 5 are long; PLOS may require tables as cell-based (no p{} paragraphs) or may move Tables 4–5 to supporting information.
24. Minor overfull boxes remain in the S1 Table PDF (long file paths, ≤ 9 pt).
