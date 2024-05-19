from os import environ

from google.cloud import translate

PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

def print_supported_languages(display_language_code: str):
  client = translate.TranslationServiceClient()

  response = client.get_supported_languages(
    parent=PARENT,
    display_language_code=display_language_code,
  )

  languages = response.languages
  print(f" Languages: {len(languages)} ".center(60, "-"))
  for language in languages:
    language_code = language.language_code
    display_name = language.display_name
    print(f"{language_code:10}{display_name}")

print_supported_languages("en")
print_supported_languages("fr")


def translate_text(text: str, target_language_code: str) -> translate.Translation:
  client = translate.TranslationServiceClient()

  response = client.translate_text(
    parent=PARENT,
    contents=[text],
    target_language_code=target_language_code,
  )

  return response.translations[0]

text = "Hello World!"
target_languages = ["tr", "de", "es", "it", "el", "zh", "ja", "ko"]

print(f" {text} ".center(50, "-"))
for target_language in target_languages:
    translation = translate_text(text, target_language)
    source_language = translation.detected_language_code
    translated_text = translation.translated_text
    print(f"{source_language} → {target_language} : {translated_text}")
