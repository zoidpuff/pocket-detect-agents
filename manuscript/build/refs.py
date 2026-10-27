"""Reference data for the BURROW manuscript.

Sources: LITERATURE.md (venue, volume, DOI or PMC id as listed there) and standard
software/statistics references. Fields marked with TODO in `flag` are incomplete or
need checking; see manuscript/OPEN_ISSUES.md. Nothing here may be guessed: missing
initials or titles are left as explicit placeholders.
"""

REFS = {
    "hendlich1997": dict(type="article", authors=["Hendlich M", "Rippmann F", "Barnickel G"],
        title="LIGSITE: automatic and efficient detection of potential small molecule-binding sites in proteins",
        journal="J Mol Graph Model", year="1997", volume="15", pages="359-363",
        doi="10.1016/S1093-3263(98)00002-3"),
    "weisel2007": dict(type="article", authors=["Weisel M", "Proschak E", "Schneider G"],
        title="PocketPicker: analysis of ligand binding-sites with shape descriptors",
        journal="Chem Cent J", year="2007", volume="1", pages="7", doi="10.1186/1752-153X-1-7"),
    "leguilloux2009": dict(type="article", authors=["Le Guilloux V", "Schmidtke P", "Tufféry P"],
        title="Fpocket: an open source platform for ligand pocket detection",
        journal="BMC Bioinformatics", year="2009", volume="10", pages="168", doi="10.1186/1471-2105-10-168"),
    "krivak2015": dict(type="article", authors=["Krivák R", "Hoksza D"],
        title="Improving protein-ligand binding site prediction accuracy by classification of inner pocket points using local features",
        journal="J Cheminform", year="2015", volume="7", pages="12", note="PMC4414931"),
    "krivak2018": dict(type="article", authors=["Krivák R", "Hoksza D"],
        title="P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure",
        journal="J Cheminform", year="2018", volume="10", pages="39", doi="10.1186/s13321-018-0285-8"),
    "aggarwal2022": dict(type="article", authors=["Aggarwal R", "Gupta A", "Chelur V", "Jawahar CV", "Priyakumar UD"],
        title="DeepPocket: ligand binding site detection and segmentation using 3D convolutional neural networks",
        journal="J Chem Inf Model", year="2022", volume="62", pages="5069", doi="10.1021/acs.jcim.1c00799"),
    "kandel2024": dict(type="article", authors=["Kandel J"], etal=True,
        title="PUResNetV2.0: a deep learning model leveraging sparse representation for improved ligand binding site prediction",
        journal="J Cheminform", year="2024", volume="16", pages="66", doi="10.1186/s13321-024-00865-6",
        flag="TODO verify title and co-authors"),
    "smith2024": dict(type="article", authors=["Smith Z", "Strobel M", "Vani BP", "Tiwary P"],
        title="Graph attention site prediction (GrASP): identifying druggable binding sites using graph neural networks with attention",
        journal="J Chem Inf Model", year="2024", volume="64", pages="2637", doi="10.1021/acs.jcim.3c01698",
        flag="TODO verify title and author list"),
    "sestak2025": dict(type="article", authors=["Sestak F", "Schneckenreiter L", "Brandstetter J", "Hochreiter S", "Mayr A", "Klambauer G"],
        title="VN-EGNN: E(3)-equivariant graph neural networks with virtual nodes enhance protein binding site identification",
        journal="J Cheminform", year="2026", volume="18", pages="11", doi="10.1186/s13321-025-01127-9",
        flag="TODO verify title and author list (LITERATURE.md gives first author, venue, DOI)"),
    "carbery2024": dict(type="article", authors=["Carbery A", "Buttenschoen M", "Skyner R", "von Delft F", "Deane CM"],
        title="Learnt representations of proteins can be used for accurate prediction of small molecule binding sites on experimentally determined and predicted protein structures",
        journal="J Cheminform", year="2024", volume="16", pages="32", doi="10.1186/s13321-024-00821-4",
        flag="TODO verify title and author list"),
    "utges2024": dict(type="article", authors=["Utgés JS", "Barton GJ"],
        title="Comparative evaluation of methods for the prediction of protein-ligand binding sites",
        journal="J Cheminform", year="2024", volume="16", pages="126", doi="10.1186/s13321-024-00923-z"),
    "yang2013": dict(type="article", authors=["Yang J", "Roy A", "Zhang Y"],
        title="Protein-ligand binding site recognition using complementary binding-specific substructure comparison and sequence profile alignment",
        journal="Bioinformatics", year="2013", volume="29", pages="2588-2595"),
    "schmidtke2010": dict(type="article", authors=["Schmidtke P", "Souaille C", "Estienne F", "Baurin N", "Kroemer RT"],
        title="Large-scale comparison of four binding site detection algorithms",
        journal="J Chem Inf Model", year="2010", volume="50", pages="2191-2200",
        flag="TODO verify author list (LITERATURE.md gives 'Schmidtke et al. 2010, JCIM 50:2191')"),
    "skrhak2025": dict(type="article", authors=["Škrhák V", "Novotný M", "Feidakis CP", "Krivák R", "Hoksza D"],
        title="CryptoBench: cryptic protein-ligand binding sites dataset and benchmark",
        journal="Bioinformatics", year="2025", volume="41", pages="btae745", note="PMC11725321",
        flag="TODO verify initials of Novotný and Feidakis"),
    "portnoi2026": dict(type="misc", authors=["Portnoi", "Keys", "DiFrancesco", "Gruver"],
        title="[Title of the zPocket/PocketWatch preprint to be completed]",
        journal="arXiv", year="2026", pages="2610.17696",
        flag="TODO add initials and title"),
    "durairaj2024": dict(type="misc", authors=["Durairaj J"], etal=True,
        title="PLINDER: the protein-ligand interactions dataset and evaluation resource",
        journal="bioRxiv", year="2024", doi="10.1101/2024.07.17.603955"),
    "jumper2021": dict(type="article", authors=["Jumper J", "Evans R", "Pritzel A", "Green T", "Figurnov M", "Ronneberger O"], etal=True,
        title="Highly accurate protein structure prediction with AlphaFold",
        journal="Nature", year="2021", volume="596", pages="583-589", doi="10.1038/s41586-021-03819-2"),
    "varadi2022": dict(type="article", authors=["Varadi M", "Anyango S", "Deshpande M", "Nair S", "Natassia C", "Yordanova G"], etal=True,
        title="AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models",
        journal="Nucleic Acids Res", year="2022", volume="50", pages="D439-D444", doi="10.1093/nar/gkab1061"),
    "carpenter2025": dict(type="article", authors=["Carpenter", "Altman RB"],
        title="[Title of the HOTPocket article to be completed]",
        journal="J Cheminform", year="2025", volume="17", pages="180", doi="10.1186/s13321-025-01125-x",
        flag="TODO add first author's initials and title"),
    "zillmer2025": dict(type="article", authors=["Zillmer", "Walther D"],
        title="[Title to be completed]",
        journal="PLoS Comput Biol", year="2025", note="PMC12324681",
        flag="TODO add initials, title, volume, article number and DOI"),
    "gazizov2026": dict(type="article", authors=["Gazizov"], etal=True,
        title="[Title of the AF2BIND article to be completed]",
        journal="Nat Methods", year="2026", volume="23", pages="626", doi="10.1038/s41592-026-03011-2",
        flag="TODO add full author list (LITERATURE.md: Gazizov, ..., Ovchinnikov, Polizzi) and title"),
    "ke2017": dict(type="inproceedings", authors=["Ke G", "Meng Q", "Finley T", "Wang T", "Chen W", "Ma W", "Ye Q", "Liu TY"],
        title="LightGBM: a highly efficient gradient boosting decision tree",
        booktitle="Advances in Neural Information Processing Systems 30", year="2017", pages="3146-3154"),
    "zhang2024": dict(type="article", authors=["Zhang C", "Zhang X", "Freddolino PL", "Zhang Y"],
        title="BioLiP2: an updated structure database for biologically relevant ligand-protein interactions",
        journal="Nucleic Acids Res", year="2024", volume="52", pages="D404-D412", note="PMC10767969",
        flag="TODO verify author list and title"),
    "holm1979": dict(type="article", authors=["Holm S"],
        title="A simple sequentially rejective multiple test procedure",
        journal="Scand J Stat", year="1979", volume="6", pages="65-70",
        flag="standard statistics reference, not in LITERATURE.md"),
    "meller2023": dict(type="article", authors=["Meller A", "Ward M", "Borowsky J", "Kshirsagar M", "Lotthammer JM", "Oviedo F"], etal=True,
        title="Predicting locations of cryptic pockets from single protein structures using the PocketMiner graph neural network",
        journal="Nat Commun", year="2023", volume="14", pages="1177", doi="10.1038/s41467-023-36699-3",
        flag="TODO verify author list"),
    "vankempen2024": dict(type="article", authors=["van Kempen M", "Kim SS", "Tumescheit C", "Mirdita M", "Lee J", "Gilchrist CLM"], etal=True,
        title="Fast and accurate protein structure search with Foldseek",
        journal="Nat Biotechnol", year="2024", volume="42", pages="243-246", doi="10.1038/s41587-023-01773-0"),
    "berman2000": dict(type="article", authors=["Berman HM", "Westbrook J", "Feng Z", "Gilliland G", "Bhat TN", "Weissig H"], etal=True,
        title="The Protein Data Bank",
        journal="Nucleic Acids Res", year="2000", volume="28", pages="235-242", doi="10.1093/nar/28.1.235"),
    "clark2026": dict(type="misc", authors=["Clark TH", "Martin MB", "Mahowald K"],
        title="Why is AI-generated language so hard to understand?",
        journal="arXiv", year="2026", pages="2610.29162",
        flag="cited from manuscript/skill/SKILL.md, not LITERATURE.md"),
    "wojdyr2022": dict(type="article", authors=["Wojdyr M"],
        title="GEMMI: a library for structural biology",
        journal="J Open Source Softw", year="2022", volume="7", pages="4200", doi="10.21105/joss.04200"),
    "harris2020": dict(type="article", authors=["Harris CR", "Millman KJ", "van der Walt SJ", "Gommers R", "Virtanen P", "Cournapeau D"], etal=True,
        title="Array programming with NumPy",
        journal="Nature", year="2020", volume="585", pages="357-362", doi="10.1038/s41586-020-2649-2"),
    "virtanen2020": dict(type="article", authors=["Virtanen P", "Gommers R", "Oliphant TE", "Haberland M", "Reddy T", "Cournapeau D"], etal=True,
        title="SciPy 1.0: fundamental algorithms for scientific computing in Python",
        journal="Nat Methods", year="2020", volume="17", pages="261-272", doi="10.1038/s41592-019-0686-2"),
    "lam2015": dict(type="inproceedings", authors=["Lam SK", "Pitrou A", "Seibert S"],
        title="Numba: a LLVM-based Python JIT compiler",
        booktitle="Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC", year="2015", pages="1-6",
        doi="10.1145/2833157.2833162"),
    "tillet2019": dict(type="inproceedings", authors=["Tillet P", "Kung HT", "Cox D"],
        title="Triton: an intermediate language and compiler for tiled neural network computations",
        booktitle="Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages",
        year="2019", pages="10-19", doi="10.1145/3315508.3329973"),
    "paszke2019": dict(type="inproceedings", authors=["Paszke A", "Gross S", "Massa F", "Lerer A", "Bradbury J", "Chanan G"], etal=True,
        title="PyTorch: an imperative style, high-performance deep learning library",
        booktitle="Advances in Neural Information Processing Systems 32", year="2019", pages="8024-8035"),
    "steinegger2017": dict(type="article", authors=["Steinegger M", "Söding J"],
        title="MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets",
        journal="Nat Biotechnol", year="2017", volume="35", pages="1026-1028", doi="10.1038/nbt.3988"),
    "desaphy2015": dict(type="article", authors=["Desaphy J", "Bret G", "Rognan D", "Kellenberger E"],
        title="sc-PDB: a 3D-database of ligandable binding sites - 10 years on",
        journal="Nucleic Acids Res", year="2015", volume="43", pages="D399-D404", doi="10.1093/nar/gku928"),
    "huang2006": dict(type="article", authors=["Huang B", "Schroeder M"],
        title="LIGSITEcsc: predicting ligand binding sites using the Connolly surface and degree of conservation",
        journal="BMC Struct Biol", year="2006", volume="6", pages="19", doi="10.1186/1472-6807-6-19"),
    "kawabata2010": dict(type="article", authors=["Kawabata T"],
        title="Detection of multiscale pockets on protein surfaces using mathematical morphology",
        journal="Proteins", year="2010", volume="78", pages="1195-1211", doi="10.1002/prot.22639"),
    "halgren2009": dict(type="article", authors=["Halgren TA"],
        title="Identifying and characterizing binding sites and assessing druggability",
        journal="J Chem Inf Model", year="2009", volume="49", pages="377-389", doi="10.1021/ci800324m"),
    "schmidtke2010b": dict(type="article", authors=["Schmidtke P", "Barril X"],
        title="Understanding and predicting druggability. A high-throughput method for detection of drug binding sites",
        journal="J Med Chem", year="2010", volume="53", pages="5858-5867", doi="10.1021/jm100574m"),
}


def vancouver(r):
    auth = ", ".join(r["authors"]) + (", et al" if r.get("etal") else "")
    title = r["title"].rstrip(".")
    end = "?" if title.endswith("?") else "."
    title = title.rstrip("?")
    s = f"{auth}. {title}{end} "
    if r["type"] == "inproceedings":
        s += f"In: {r['booktitle']}; {r['year']}."
        if r.get("pages"):
            s += f" p. {r['pages'].replace('-', '–')}."
    elif r["journal"] in ("arXiv", "bioRxiv"):
        s += f"{r['journal']}. {r['year']}."
        if r["journal"] == "arXiv" and r.get("pages"):
            s += f" arXiv:{r['pages']}."
    else:
        s += f"{r['journal']}. {r['year']}"
        if r.get("volume"):
            s += f";{r['volume']}"
        if r.get("pages"):
            s += f":{r['pages'].replace('-', '–')}"
        s += "."
    if r.get("doi"):
        s += f" doi:{r['doi']}"
    elif r.get("note"):
        s += f" {r['note']}"
    return s.strip()


def _bib_author(a):
    parts = a.split(" ")
    if len(parts) == 1:
        return "{" + a + "}"
    initials = parts[-1]
    last = " ".join(parts[:-1])
    if initials.isupper() and len(initials) <= 3:
        return "{" + last + "}, " + " ".join(c + "." for c in initials)
    return "{" + a + "}"


def bibtex(key, r):
    authors = " and ".join(_bib_author(a) for a in r["authors"])
    if r.get("etal"):
        authors += " and others"
    f = {"author": authors, "title": "{" + r["title"] + "}", "year": r["year"]}
    if r["type"] == "inproceedings":
        f["booktitle"] = r["booktitle"]
    elif r["journal"] in ("arXiv", "bioRxiv"):
        f["journal"] = r["journal"] if r["journal"] == "bioRxiv" else "arXiv"
        if r["journal"] == "arXiv":
            f["volume"] = "arXiv:" + r["pages"]
    else:
        f["journal"] = r["journal"]
        if r.get("volume"):
            f["volume"] = r["volume"]
    if r.get("pages") and r["journal" if "journal" in r else "type"] not in ("arXiv",):
        if not (r.get("journal") == "arXiv"):
            f["pages"] = r["pages"].replace("-", "--")
    if r.get("doi"):
        f["doi"] = r["doi"]
    if r.get("note") and not r.get("doi"):
        f["note"] = r["note"]
    btype = "inproceedings" if r["type"] == "inproceedings" else "article"
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
    return f"@{btype}{{{key},\n{body}\n}}\n"
