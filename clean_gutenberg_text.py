import re

def clean_gutenberg_text(file_path, eot_symbol="<|endoftext|>"):
    """ Cleans a Project Gutenberg book text file by removing non-novel content dynamically. """

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # # Find start and end markers dynamically
    # start_idx, end_idx = None, None

    # for i, line in enumerate(lines):
    #     if re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK", line):
    #         start_idx = i + 1  # Skip the start line itself
    #     if re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", line):
    #         end_idx = i  # Stop before the end line

    # # Keep only content between start and end markers
    # if start_idx is not None and end_idx is not None:
    #     lines = lines[start_idx:end_idx]
    # elif start_idx is not None:
    #     lines = lines[start_idx:]  # If no end marker, keep until the end
    # else:
    #     lines = lines  # If no markers, keep everything (fallback)

    # Convert list back to text for easier regex processing
    text = "".join(lines)
    
    # Remove text from "END OF VOL. X" until the next chapter
    text = re.sub(r"END OF VOL\. [IVXLCDM]+.*?(?=\n\s*CHAPTER\s+[IVXLCDM]+\b)", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove text from "END OF THE SECOND VOLUME." until the next chapter
    text = re.sub(r"END OF THE SECOND VOLUME\..*?(?=\n\s*CHAPTER\s+[IVXLCDM]+\b)", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove "(End of volume one.)"
    text = re.sub(r"\(End of volume one\.\)", "", text, flags=re.IGNORECASE)

    # Find the first occurrence of "CHAPTER I" that has an empty line before it
    match = re.search(r"\n\s*\n(CHAPTER\s+I\b)", text, re.IGNORECASE)

    if match:
        text = text[match.start():]  # Keep content from first valid "CHAPTER I"
    
    # Replace chapter headers with <|endoftext|>
    text = re.sub(r"\n\s*CHAPTER\s+[.IVXLCDM]+\b", f"\n\n{eot_symbol}\n\n", text, flags=re.IGNORECASE)
    
    # Remove star formations like "       *       *       *       *       *"
    text = re.sub(r"(\s*\*\s*){3,}", "", text)
    
    # remove "_" from words (it is originally used to indicate italic styling)
    text = re.sub("_", "", text, flags=re.DOTALL | re.IGNORECASE)
    
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