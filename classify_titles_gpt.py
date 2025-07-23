import pandas as pd
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

def classify_titles_with_gpt():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Load existing labeled file if it exists
    if os.path.exists("unique_title_author_labeled.csv"):
        df = pd.read_csv("unique_title_author_labeled.csv")
        print("Loaded existing labeled titles.")
    else:
        df = pd.read_csv("unique_titles.csv")
        df["is_novel"] = None
        print("Starting fresh from unique_titles.csv")

    # Loop through titles
    for i, row in df.iterrows():
        title = row["title"]
        author = row["author"]

        # Skip if already labeled
        if pd.notnull(row.get("is_novel")):
            continue

        prompt = f"Is the following book title a novel? Answer only with True or False.\n\nTitle: {title}"

        if pd.notnull(author):
            prompt += f"\nAuthor: {author}"

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a classifier of book genres."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0,
            )

            # Interpret response
            answer = completion.choices[0].message.content.strip()
            if "true" in answer.lower():
                df.at[i, "is_novel"] = True
            elif "false" in answer.lower():
                df.at[i, "is_novel"] = False
            else:
                print(f"Unclear response for '{title}':", answer)

        except Exception as e:
            print(f"Error at index {i} for title '{title}':", e)
            time.sleep(10)
            continue

        time.sleep(5)

        # Save checkpoint
        if i % 10 == 0:
            df.to_csv("unique_title_author_labeled.csv", index=False)
            print(f"Checkpoint saved at index {i}")

    # Final save
    df.to_csv("unique_title_author_labeled.csv", index=False)
    print("All done.")

if __name__ == "__main__":
    classify_titles_with_gpt()
