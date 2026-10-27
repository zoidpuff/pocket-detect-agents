# Open issues for the submitting authors

## References
References were verified against Crossref/PubMed by the reference-verifier agent (REFERENCES_CHECK.md) and corrected in build/refs.py. Added in revision 1 and checked on Crossref by the manuscript writer: Lin et al. 2008 (weighted contact number, doi:10.1002/prot.21983), Chernozhukov et al. 2018 (cross-fitting, doi:10.1111/ectj.12097), Dürr et al. 2023 (Metal3D, doi:10.1038/s41467-023-37870-6). LightGBM and PyTorch page ranges come from printed proceedings (not re-verified). No citation is given for Claude Code, Fibonacci sphere directions, Hungarian matching or Kabsch superposition.

## Content
14. Follow-up messages are now verbatim in S2 Text §9 (from FOLLOWUP_PROMPTS.md). Agent briefs and session transcripts were not archived; this is stated in Methods and S2 Text. Claude Code version and token usage were not recorded.
15. **Fig 3 (figure file):** the dashed "+2" bar still extends through the C1-strict rows (referee M1); the caption is fixed, but make_figures.py should shorten the line to the C1 rows. S2 Fig C shows 10 test rows (adds AF-set DCA4) while the text counts 9 (REPORT's definition); S2 Fig A3 labels overlap (referee). Fig 4 legend "P2Rank (published)" is explained in the caption.
16. **Placeholders for the human authors:** author list, affiliations, corresponding author, author contributions (CRediT), funding, competing interests, what the human authors verified (AI-contribution statement), repository URL and archive DOI. The AI-contribution statement under the author line and the box should be checked against journal policy on AI authorship/disclosure (AI agents cannot be listed as authors).
17. **Repository URL** is the placeholder https://github.com/<to-be-published>/burrow (main text, Data and code availability). The release repository (release/burrow) was still a skeleton when this text was written; check that it contains the models, predictions and scripts that the availability statement promises.
18. **Speed numbers** come from a shared node (±30%); REPORT §6.7 recommends repeating them on an exclusive node before publication.
19. **"More than 130 git commits"** is from `git log` at the time of writing (136 commits, including the first manuscript commit); update if needed.
20. Revision 1: Introduction–Discussion ≈4,720 words, Methods ≈3,120 (total ≈7,840 excluding tables, captions and references; above the ~6,500 target, mainly from referee-requested caveats and agent-setup details); abstract 296 words; author summary 197 words. The referee's COULD-item C1 (shortening) was not applied.
25. The "How this work was produced" box was kept at the top as the user requested, although PLOS research articles have no boxes (referee format note); it can be moved into the Introduction at submission.

## Format / build
21. `main.tex` follows the PLOS LaTeX template (downloaded from journals.plos.org, plos2025.bst included) adapted for XeTeX/Tectonic (`\DisableLigatures` only under pdfTeX; fvextra, url, iftex added). Figures are embedded for the review PDF via `\embedfigurestrue`; set `\embedfiguresfalse` (or run `python3 build/build.py --no-embed`) for submission, where PLOS wants figures as separate files. PLOS also asks for the .bbl content to be pasted into the .tex at submission; references.bib + plos2025.bst currently generate it.
22. All outputs are generated: edit `build/*.src.md` (text, tables, captions) and `build/refs.py` (references), then run `python3 manuscript/build/build.py` and compile with Tectonic (`tectonic -X compile main.tex`). Hand edits to main.md/main.tex are overwritten.
23. Tables 1, 2, 4 and 5 are long; PLOS may require tables as cell-based (no p{} paragraphs) or may move Tables 4–5 to supporting information.
24. Minor overfull boxes remain in the S1 Table PDF (long file paths, ≤ 9 pt).
