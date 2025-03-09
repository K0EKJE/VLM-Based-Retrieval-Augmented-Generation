import base64
import io
import json
from openai import OpenAI
from tqdm import tqdm
from preprocessor import PDFToImageConverter, ImageOCR
from datasets import load_dataset


class ImageQuestionGenerator:
    def __init__(self, client, pdf_path, output_file="./data/image_question.json", dpi=200, model="gpt-4o-mini"):
        self.pdf_path = pdf_path
        self.output_file = output_file
        self.converter = PDFToImageConverter(dpi=dpi)
        self.client = client
        self.model = model
        self.prompt = '''
        Given the input image, generate a multiple-choice question related to the content of the image. Avoid abstract or interpretive questions like "What is this image about?" The question should test understanding or observation of the image. Provide exactly four answer choices, with only one correct answer among them. Ensure the answer choices are diverse and not overly obvious, offering a reasonable level of challenge. The question and the answer choices should be provided separately.

        Finally, return the output in the following JSON format:
        {
            "question": "",
            "options": [
                "A",
                "B",
                "C",
                "D"
            ],
            "correct_answer": "only an option from [A,B,C,D]"
        }
        Make sure the question is clear and concise, options is an array of exactly four strings, and correct_answer matches exactly one of the provided options."
        '''

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_questions(self, max_pages=2):
        images = self.converter.extract_images(self.pdf_path)
        json_dataset = {}

        for i, image in enumerate(tqdm(images[:max_pages])):
            base64_image = self.encode_image(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
            )

            response_content = response.choices[0].message.content
            json_data = json.loads(
                response_content.replace("```json", "").replace("```", "")
            )
            json_data['page'] = i + 1
            json_dataset[i + 1] = json_data

        self.save_to_file(json_dataset)

    def save_to_file(self, data):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class QAGenerator:
    def __init__(self, client, model="gpt-4o-mini", retry_delay=5):
        self.client = client
        self.model = model
        self.retry_delay = retry_delay

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def expand_query(self, query):
        expansion_prompt = (
            f"""
            You are an AI assistant that helps improve document retrieval by expanding queries.

            Given an input query: {query}, generate five semantically similar queries that rephrase the original question while maintaining its intent. The expanded queries should include variations in wording, synonyms, and domain-specific terminology to maximize recall during retrieval.
            
            Return the output in **JSON format** with the key `"queries"` containing a list of five expanded queries.
            
            ### Example:
            **Input Query:** "What are the requirements for getting a driver's license?"  
            **Output:**
            """
            +
            """
            ```json
            {
              "queries": [
                "What documents do I need to apply for a driver's license?",
                "What are the age and identification criteria for obtaining a driver's license?",
                "How can I qualify for a driver's license?",
                "What are the necessary steps to apply for a driver's license?",
                "What is required to be eligible for a driver's license?"
              ]
            }
            ```
            """
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": expansion_prompt}]}
                ],
            )
            expanded_query = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            expanded_query = json.loads(expanded_query)
            return expanded_query
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query

    def check_image_relevance(self, query, image):
        base64_image = self.encode_image(image)

        check_prompt = (
            f"The user has asked the following query: '{query}'.\n"
            "You have been provided with an image. Analyze whether this image is useful for answering the query.\n"
            "Additionally, determine whether more images are needed to provide a complete answer.\n"
            "Your response should be in JSON format as follows:\n"
            "{\n"
            '    "need_additional_image": "yes" or "no",\n'
            '    "reason": "Explain why the given image is sufficient or not.",\n'
            '    "additional images description": "If additional images are needed, describe what kind of images are required. Otherwise, leave this empty."\n'
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": check_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
            )
            result = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")

            try:
                result_json = json.loads(result)
                return result_json
            except json.JSONDecodeError:
                print(f"Failed to parse response JSON: {result}")
                return {
                    "need_additional_image": "no",
                    "reason": "Unable to determine, defaulting to 'no'.",
                    "additional images description": ""
                }
        except Exception as e:
            print(f"Image relevance check failed: {e}")
            return {
                "need_additional_image": "no",
                "reason": "An error occurred while processing the image.",
                "additional images description": ""
            }

    def response(self, query, image=None, options=None, ocr_content=None):
        # query = self.expand_query(query)

        cot_prompt = (
            "Think step by step before answering. If it's a multiple-choice question, "
            "analyze each option before selecting the correct answer."
        )

        prompt = f"{query}\n\n{cot_prompt}\n"

        if options:
            prompt += f"{options}\n\nplease only give an option letter"
        if ocr_content:
            prompt += f"\n\nYou should answer the question based on the following content:\n{ocr_content}\n"
        if image:
            # print("use image")
            base64_image = self.encode_image(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
            )
        else:
            # print("no image")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        return response.choices[0].message.content


if __name__ == '__main__':
    client = OpenAI()
    # generator = ImageQuestionGenerator(
    #     client=client,
    #     pdf_path="/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/dmv.pdf",
    #     output_file="./data/image_question_1.json"
    # )
    # generator.generate_questions(max_pages=90)

    dataset = load_dataset('parquet',
                           data_files='/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/vidore/arxivqa_test_subsampled/test-00000-of-00001.parquet')
    print(dataset)
    images = dataset['train']['image']
    queries = dataset['train']['query']
    options = dataset['train']['options']
    answers = dataset['train']['answer']
    # images[0].show()
    # print(queries[0])
    # print(options[0])
    # print(answers[0])

    QAGenerator = QAGenerator(client=client, model="gpt-4o-mini")
    expanded_query = QAGenerator.expand_query(queries[0])
    print(expanded_query)
    # ocr = ImageOCR(lang='eng')
    #
    # acc = 0
    # n = 2
    # for i in tqdm(range(n)):
    #     ocr_content = ocr.extract_text(images[i])
    #     response = QAGenerator.response(
    #         query=queries[i],
    #         # image=images[i],
    #         options=options[i],
    #         ocr_content=ocr_content,
    #     )
    #     if len(answers[i]) == 1 and answers[i] in response:
    #         acc += 1
    #     else:
    #         if len(response) == 1 and response in answers[i]:
    #             acc += 1
    # acc = acc / n * 100
    # print(f"Accuracy: {acc:.2f}%")
