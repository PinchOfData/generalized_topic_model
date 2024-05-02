from openai import OpenAI
client = OpenAI(api_key='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')

def translate_text(text):
  response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Translate English into Chinese:\n\n" + text,
  )
  return response.choices[0].text.strip()


result = translate_text("Hello, how are you?")
print(result)
