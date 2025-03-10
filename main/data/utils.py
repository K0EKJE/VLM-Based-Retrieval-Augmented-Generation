import json
from openai import OpenAI
from tqdm import tqdm

def format_data():
    input_filename = "questions.json"

    with open(input_filename, "r", encoding="utf-8") as file:
        input_json = json.load(file)

    output_json = {}
    for index, question_data in enumerate(input_json["questions"], start=1):
        output_json[str(index)] = {
            "question": question_data["question"],
            "options": question_data["options"],
            "correct_answer": question_data["correct_answer"],
        }

    output_filename = "formatted_question.json"
    with open(output_filename, "w", encoding="utf-8") as file:
        json.dump(output_json, file, indent=4, ensure_ascii=False)


def change_language():
    client = OpenAI()
    with open("formatted_question.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    translated_data = {}

    for key, value in tqdm(list(data.items())):

        translation_prompt = (
            f"""
            Translate the following Chinese multiple-choice question to English:\n\n
            {value}
            """
            +
            """
            Provide the output in the following JSON format:\n\n
            {
                "question":"english version",
                "options": [
                    "A. ",
                    "B. ",
                    "C. ",
                    "D. "
                ],
                "correct_answer": "only one word"
            }
            """
        )


        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": translation_prompt}
            ]
        )
        translated_text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        # print(translated_text)
        translated_data[key] = json.loads(translated_text)

        with open("translated_questions.json", "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print("save to translated_questions.json")


if __name__ == '__main__':
    change_language()