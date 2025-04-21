import openai
import os
from dotenv import load_dotenv
load_dotenv()
def test(client, prompt):
    try:
    
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an helpful assistant to answer the user quaries."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract and parse the response
        response= response.choices[0].message.content
    
        return response

    except Exception as e:
        raise e

#key= os.getenv("OPENAI_API_KEY")
key = "sk-proj-"
print(key)
client = openai.OpenAI(api_key=key)


print(test(client=client, prompt= "Give me a joke" ))