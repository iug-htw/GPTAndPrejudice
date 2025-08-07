import re
from itertools import chain

novel_titles = [
    "PERSUASION", "NORTHANGER ABBEY", "MANSFIELD PARK", "EMMA",
    "LOVE AND FREINDSHIP AND OTHER EARLY WORKS", "PRIDE AND PREJUDICE",
    "SENSE & SENSIBILITY", "ORMOND", "HARRINGTON", "Title: Agnes Grey",
    "Title: The Tenant of Wildfell Hall", "JANE EYRE", "WIVES AND DAUGHTERS.",
    "THE PROFESSOR", "VILLETTE.", "CRANFORD", "MARY BARTON", "CECILIA",
    "MY LADY LUDLOW", "Title: North and South", "RUTH", "SYLVIA'S LOVERS.",
    "THE MOORLAND COTTAGE.", "Title: Wuthering Heights", "SHIRLEY",
    "Title: Camilla; or, A Picture of Youth", "CECILIA, Volume 1 (of 3)", 
    "Title: Evelina, Or, the History of a Young Lady's Entrance into the World",
    "Title: The Wanderer; or, Female Difficulties (Volume 1 of 5)",
    "Title: The Wanderer; or, Female Difficulties (Volume 2 of 5)",
    "Title: The Wanderer; or, Female Difficulties (Volume 3 of 5)",
    "Title: The Wanderer; or, Female Difficulties (Volume 4 of 5)",
    "Title: The Wanderer; or, Female Difficulties (Volume 5 of 5)",
    "Title: Middlemarch", "Title: The Mill on the Floss", "BELINDA.",
    "CASTLE RACKRENT", "HELEN", "Title: Leonora", "PATRONAGE.", "THE ABSENTEE",
    "TO-MORROW", "LODORE.", "MARRIAGE."
]


def find_novel_positions(text, titles):
    """Finds the second occurrence of each title (or first if only one exists)"""
    positions = {}
    for title in titles:
        matches = [m.start() for m in re.finditer(rf"\n\b{title}\b\n", text)]
        if len(matches) > 1:
            positions[title] = matches[1]  # Second occurrence
        elif len(matches) == 1:
            positions[title] = matches[0]  # First occurrence if only one exists
    return positions

def is_trivial_block(block):
    """Return True if block contains only one short sentence or junk."""
    block = block.strip()
    if not block or re.fullmatch(r"[\W_]+", block):
        return True

    sentences = re.split(r"[.!?]\s+", block)
    return all(len(re.findall(r"\w+", s)) <= 4 for s in sentences if s.strip())


def clean_text_blocks(text, eot_symbol):
    def truncate_after(text, regex, ignore_case=False):
        flags = re.IGNORECASE if ignore_case else 0
        return re.split(regex, text, flags=flags)[0]

    patterns = [
        (r"INTRODUCTION", False),
        (r"PREFACE", False),
        (r"GLOSSARY\b", False),
        (r"THE\s+END", False),
        (r"End\s+of\s+Project\s+Gutenberg[’']s\s+", True),
        (r"END\s+OF\s+", False),
        (r"^\bFinis\b", True),
        (r"START:\s+FULL\s+LICENSE", False),
        (r"\[*Transcriber[’']s\s+note", True),
        (r"A\s+NOTE\s+ON\s+THE\s+TEXT", False),
        (r"PRINTED\s+BY", False),
    ]

    blocks = [b.strip() for b in text.split(eot_symbol)]
    cleaned = []

    for block in blocks:
        if re.fullmatch(r"[\W_]+", block):
            continue
        for pattern, ic in patterns:
            block = truncate_after(block, pattern, ignore_case=ic)
        block = block.strip()
        if block and not is_trivial_block(block):
            cleaned.append(block)

    return cleaned

def clean_gutenberg_text(file_path, eot_symbol="<|endoftext|>"):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    text = re.sub(
        r"LADY SUSAN[\s\S]*?(?=PRIDE AND PREJUDICE)",
        "",
        text
    )

    positions = find_novel_positions(text, novel_titles)
    if positions == {}:
        positions["_"] = 0

    sorted_titles = sorted(positions, key=positions.get)
    split_texts = {}

    for i, title in enumerate(sorted_titles):
        start = positions[title]
        end = positions[sorted_titles[i + 1]] if i + 1 < len(sorted_titles) else len(text)
        split_texts[title] = text[start:end].strip()

    def clean_novel_text(text, title, eot_symbol):
        """Cleans each novel separately"""

        # 1. Find the start of the first chapter in the text
        # (anything before this will be removed)

        chapter_markers = [
            r"^\s*CHAPTER\s+(?:[.ivxlcdm]+|\d+)(?=\s*\.?\s*(\n|--?\s*[A-Z]))",
            r"\[Illustration:\s*Chapter\s+1:[^\]]*?\]",
            r"^\s*LETTER\s+(?:[ivxlcdmj]+|\d+)\.?(?:\s*\[.*?\])?\s*$",
        ]

        chapter_i_matches = list(chain.from_iterable(
            re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for pattern in chapter_markers
        ))

        for match in chapter_i_matches:
            # Look ahead at the next ~200 characters
            lookahead = text[match.end():match.end() + 200]

            # If CHAPTER [X] appears shortly after, it's probably a TOC — skip it
            if re.search(r"CHAPTER\s+(?:[.IVXLCDM]+|\d+)\b", lookahead, re.IGNORECASE):
                continue

            # Otherwise, this is our starting point
            text = text[match.start():]
            break

        for pattern in chapter_markers: 
            text = re.sub(
                pattern,
                f"\n\n{eot_symbol}\n\n",
                text,
                flags=re.IGNORECASE | re.MULTILINE
            )

        # 3. Text cleanups

        # Remove "VOLUME [X]" or "BOOK [X]"
        text = re.sub(r"\b(VOLUME|BOOK)\s+[IVXLCDM]+\b", "", text, flags=re.IGNORECASE)

        # Remove "(End of volume one.)"
        text = re.sub(r"\(End of volume one\.\)", "", text, flags=re.IGNORECASE)

        text = re.sub(
            rf"{re.escape(eot_symbol)}\s*\n\s*[A-Z\s,.'\-]+\.?\s*(?=\n)",
            f"\n\n{eot_symbol}\n\n",
            text,
            flags=re.MULTILINE
        )

        # Clean each block
        cleaned_blocks = clean_text_blocks(text, eot_symbol)
        text = f"\n\n{eot_symbol}\n\n".join(cleaned_blocks)

        text = re.sub(f"{title}", "", text)
        text = re.sub(r"(\s*\*\s*){3,}", "", text)
        text = re.sub(r"(\s*-\s*){3,}", "", text)
        text = re.sub("_", "", text, flags=re.DOTALL | re.IGNORECASE) # Remove underscores used for italics
        text = re.sub(r"“|”", "\"", text)
        text = re.sub(r"‘|’", "\'", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) # Replace single '\n' with space
        text = re.sub(r"\n\s*\n+", "\n", text) # Replace two or more consecutive '\n' with a single '\n'
        text = re.sub(r"\[\d{1,3}\]", "", text) # Remove glossary references [X]
        text = re.sub(r"^\s{2,}\{\d{1,3}\}.*$", "", text, flags=re.MULTILINE) # Remove glossary references {X}
        text = re.sub(r"\{\d{1,3}\}", "", text)
        text = re.sub(r"\[\s*Illustration.*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*Picture:.*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*Footnote.*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[End.*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*\.\s*\n", "\n", text)

        return text.strip()

    cleaned_texts = {title: clean_novel_text(text, title, eot_symbol) for title, text in split_texts.items()}

    # Combine cleaned novels back into one text
    final_text = f"\n{eot_symbol}\n".join(cleaned_texts.values())

    return final_text.strip()