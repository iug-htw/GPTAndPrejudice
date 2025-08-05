import re

def is_trivial_block(block):
    """Return True if block contains only one short sentence or junk."""
    # Remove surrounding whitespace
    block = block.strip()

    # Reject if completely empty or just symbols
    if not block or re.fullmatch(r"[\W_]+", block):
        return True

    # Skip blocks where every sentence is very short (3–4 words max)
    sentences = re.split(r"[.!?]\s+", block)
    if all(len(re.findall(r"\w+", s)) <= 4 for s in sentences if s.strip()):
        return True

    return False

def clean_end_of_text_blocks(text, eot_symbol):
    def truncate_after_phrase(text, phrase_regex, ignore_case=False):
        """Truncate text at the first match of the regex pattern (case-insensitive)."""
        return re.split(phrase_regex, text, flags=re.IGNORECASE if ignore_case else 0)[0]
    
    # Split into blocks on the eot_symbol
    blocks = [b.strip() for b in text.split(eot_symbol)]

    cleaned_blocks = []
    for block in blocks:
        block = block.strip()

        # Skip blocks that are just punctuation or noise (like ".")
        if re.fullmatch(r"[\W_]+", block):
            continue

        # Truncate block after known non-narrative phrases
        block = truncate_after_phrase(block, r"INTRODUCTION")
        block = truncate_after_phrase(block, r"PREFACE")
        block = truncate_after_phrase(block, r"THE\s+END")
        block = truncate_after_phrase(block, r"END\s+OF\s+PROJECT\s+GUTENBERG[’']?S")
        block = truncate_after_phrase(block, r"END\s+OF\s+")
        block = truncate_after_phrase(block, r"End\s+of\s+Project\s+Gutenberg[’']s\s+")
        block = truncate_after_phrase(block, r"START:\s+FULL\s+LICENSE")
        block = truncate_after_phrase(block, r"\[*Transcriber[’']s\s+note", ignore_case=True)
        block = truncate_after_phrase(block, r"A\s+NOTE\s+ON\s+THE\s+TEXT")

        # Final cleanup and triviality check
        block = block.strip()
        if not block or is_trivial_block(block):
            continue

        cleaned_blocks.append(block)

    return cleaned_blocks

def clean_gutenberg_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Define the list of novels
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

    text = re.sub(
        r"LADY SUSAN[\s\S]*?(?=PRIDE AND PREJUDICE)",
        "",
        text
    )

    # Find the second occurrence of each title (or first if only one exists)
    def find_novel_positions(text, titles):
        positions = {}
        for title in titles:
            matches = [m.start() for m in re.finditer(rf"\n\b{title}\b\n", text)]
            if len(matches) > 1:
                positions[title] = matches[1]  # Second occurrence
            elif len(matches) == 1:
                positions[title] = matches[0]  # First occurrence if only one exists
        return positions

    positions = find_novel_positions(text, novel_titles)
    if positions == {}:
        positions["_"] = 0

    # Sort positions and split the text
    sorted_titles = sorted(positions, key=positions.get)
    split_texts = {}

    for i, title in enumerate(sorted_titles):
        start = positions[title]
        end = positions[sorted_titles[i + 1]] if i + 1 < len(sorted_titles) else len(text)
        split_texts[title] = text[start:end].strip()

    # Clean each novel separately
    def clean_text(text, title, eot_symbol="<|endoftext|>"):
        # Remove "VOLUME [X]" or "BOOK [X]"
        text = re.sub(r"\b(VOLUME|BOOK)\s+[IVXLCDM]+\b", "", text, flags=re.IGNORECASE)

        # Remove "(End of volume one.)"
        text = re.sub(r"\(End of volume one\.\)", "", text, flags=re.IGNORECASE)

        # Match CHAPTER [X] and optional next-line title
        chapter_i_matches = list(re.finditer(
            r"^\s*CHAPTER\s+(?:[.ivxlcdm]+|\d+)(?=\s*\.?\s*(\n|--?\s*[A-Z]))",
            text,
            re.IGNORECASE | re.MULTILINE
        ))

        chapter_i_matches = chapter_i_matches + list(re.finditer(
            r"\[Illustration:\s*Chapter\s+1:[^\]]*?\]",
            text,
            re.IGNORECASE | re.MULTILINE
        ))

        chapter_i_matches = chapter_i_matches + list(re.finditer(
            r"^\s*LETTER\s+(?:[IVXLCDM]+|\d+)\.?\s*(?=\s*\.?\s*(\n|--?\s*[A-Z]))?",
            text,
            re.MULTILINE
        ))

        for match in chapter_i_matches:
            # Look ahead at the next ~300 characters
            lookahead = text[match.end():match.end() + 200]

            # If CHAPTER [X] appears shortly after, it's probably a TOC — skip it
            if re.search(r"CHAPTER\s+(?:[.IVXLCDM]+|\d+)\b", lookahead, re.IGNORECASE):
                continue

            # Otherwise, this is our starting point
            text = text[match.start():]
            break

        # Replace chapter headers with <|endoftext|>
        text = re.sub(
            r"\n\s*CHAPTER\s+(?:[.IVXLCDM]+|\d+)\b",
            f"\n\n{eot_symbol}\n\n",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\[Illustration:\s*Chapter\s+\d:[^\]]*?\]",
            f"\n\n{eot_symbol}\n\n",
            text,
            flags=re.IGNORECASE
        )
        # remove illusterations
        text = re.sub(
            r"\n\s*LETTER\s+(?:[.IVXLCDM]+|\d+)\b",
            "",
            text
        )

        text = re.sub(
            rf"{re.escape(eot_symbol)}\s*\n\s*[A-Z\s,.'\-]+\.?\s*(?=\n)",
            f"\n\n{eot_symbol}\n\n",
            text,
            flags=re.MULTILINE
        )

        cleaned_blocks = clean_end_of_text_blocks(text, eot_symbol)
        # Join cleaned blocks
        text = f"\n\n{eot_symbol}\n\n".join(cleaned_blocks)

        # Remove star formations like "       *       *       *       *       *"
        text = re.sub(r"(\s*\*\s*){3,}", "", text)
        # Remove star formations like "-------"
        text = re.sub(r"(\s*-\s*){3,}", "", text)

        # Remove underscores used for italics
        text = re.sub("_", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Replace single '\n' with space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Replace two or more consecutive '\n' with a single '\n'
        text = re.sub(r"\n\s*\n+", "\n", text)

        # Remove lines that contain only a single "."
        text = re.sub(r"\n\s*\.\s*\n", "\n", text)

        # if title == "CECILIA":
        #     print(text[:500])
        #     print(60*"-")

        return text.strip()

    cleaned_texts = {title: clean_text(text, title) for title, text in split_texts.items()}

    # Combine cleaned novels back into one text
    final_text = "\n<|endoftext|>\n".join(cleaned_texts.values())

    return final_text.strip()