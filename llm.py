from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

def get_llm_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions ONLY using the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=500,
        timeout=30,
    )
    return response.choices[0].message.content.strip()
