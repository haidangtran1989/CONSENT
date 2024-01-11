import time
import openai

openai.api_key = ""


def generate_text_from_gpt4(prompt):
    ret = ""
    for i in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0
            )
            choice = response.choices[0]
            ret = choice["message"]["content"]
            time.sleep(0.34)
            break
        except:
            print("error calling OpenAI API")
            time.sleep(5)
            continue
    return ret
