import re

def clean_gutenberg_collected_work(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Define the list of novels
    novel_titles = [
        "PERSUASION", "NORTHANGER ABBEY", "MANSFIELD PARK", "EMMA",
        "LADY SUSAN", "LOVE AND FREINDSHIP AND OTHER EARLY WORKS", 
        "PRIDE AND PREJUDICE", "SENSE & SENSIBILITY",
    ]

    # Find the second occurrence of each title (or first if only one exists)
    def find_novel_positions(text, titles):
        positions = {}
        for title in titles:
            matches = [m.start() for m in re.finditer(rf"\b{title}\b", text)]
            if len(matches) > 1:
                positions[title] = matches[1]  # Second occurrence
            elif len(matches) == 1:
                positions[title] = matches[0]  # First occurrence if only one exists
        return positions

    positions = find_novel_positions(text, novel_titles)

    # Sort positions and split the text
    sorted_titles = sorted(positions, key=positions.get)
    split_texts = {}

    for i, title in enumerate(sorted_titles):
        if i == 5:
            continue
        start = positions[title]
        end = positions[sorted_titles[i + 1]] if i + 1 < len(sorted_titles) else len(text)
        split_texts[title] = text[start:end].strip()

    # Clean each novel separately
    def clean_text(text, title):
        match = re.search(r"\n\s*\n(CHAPTER\s+I\b)", text, re.IGNORECASE)
        if match:
            text = text[match.start():]

        match = re.search(r"\n\s*\n(Chapter\s+1\b)", text, re.IGNORECASE)
        if match:
            text = text[match.start():]

        match = re.search(r"\n\s*\n(A NOTE ON THE TEXT\b)", text, re.IGNORECASE)
        if match:
            text = text[:match.start()]

        
        # Replace chapter headers with <|endoftext|>
        text = re.sub(r"\n\s*CHAPTER\s+[.IVXLCDM]+\b", f"\n\n<|endoftext|>\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*CHAPTER\s+[.1234567890]+\b", f"\n\n<|endoftext|>\n\n", text, flags=re.IGNORECASE)
        
        # Remove star formations like "       *       *       *       *       *"
        text = re.sub(r"(\s*\*\s*){3,}", "", text)
        
        # remove "_" from words (it is originally used to indicate italic styling)
        text = re.sub("_", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove "finis" and "The end" from the end of the text
        text = re.sub(r"\nFinis.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\nTHE END.*", "", text, flags=re.DOTALL) 

        #  Remove illustrations and volume headers
        text = re.sub(r"\[Illustration: .*?\]", "", text, flags=re.DOTALL)
        text = re.sub(r"\[End volume.*?\]", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\(End of volume.*?\)", "", text, flags=re.DOTALL |re.IGNORECASE)
        text = re.sub(r"\n\s*VOLUME\s+[.IVXLCDM]+\b", "", text, flags=re.DOTALL)

        
        # Remove "Transcriber's note:" and everything after it if it appears at the end of the file
        note = text.find(r"Transcriber's note", 0)
        text = text[:note]
            
        # Replace single '\n' with space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Replace two or more consecutive '\n' with a single '\n'
        text = re.sub(r"\n\s*\n+", "\n", text)
        
        # Remove lines that contain only a single "."
        text = re.sub(r"\n\s*\.\s*\n", "\n", text)
        
        # Remove the first line before returning (The first removed chapter header)
        text = "\n".join(text.split("\n")[2:])

        return text.strip()

    cleaned_texts = {title: clean_text(text, title) for title, text in split_texts.items()}

    # Combine cleaned novels back into one text
    final_text = "\n<|endoftext|>\n".join(cleaned_texts.values())

    return final_text.strip()
