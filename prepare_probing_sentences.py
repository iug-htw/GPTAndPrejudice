from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import os


def prepare_probing_sentences():
    # Set your OpenAI API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Load the reviewed CSV file
    df = pd.read_csv("Reviewed_Probing_Sentences.csv")

    # Create a new column for GPT-edited sentences
    edited_sentences = []

    # Iterate over rows
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row['sentence']
        assessment = row['assessment']
        suggestion = row['edit_suggestion']

        prompt = f"""
    You are a helpful editor. Here is a sentence from a Jane Austen novel:

    "{sentence}"

    This sentence id {assessment}. Edit or paraphrase it according to this suggestion: "{suggestion}"
    Only output the revised sentence. if multiple sentences are present, concatenate them with a $ sign.
    """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a careful sentence rewriter and editor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            edited = response.choices[0].message.content.strip()
            sentences = edited.split("$")
        except Exception as e:
            edited = f"[ERROR: {e}]"
            sentences = [edited]

        edited_sentences.append(sentences)

    # Add the edited sentences to the DataFrame
    df['gpt_edited_sentence'] = edited_sentences

    # Save the updated CSV
    df.to_csv("probing_sentences_gpt4o_edited.csv", index=False)
    print("✅ Done! Output saved to probing_sentences_gpt4o_edited.csv")

if __name__ == "__main__":
    prepare_probing_sentences()