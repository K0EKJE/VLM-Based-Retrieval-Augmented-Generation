import numpy as np
from datasets import load_dataset
from openai import OpenAI
import json
from tqdm import tqdm
from preprocessor import PDFToImageConverter, ImageOCR
from preprocessor import (
    create_model,
    ImageTextEmbedder,
)
from dbManager import (
    save_db,
    load_db,
    compute_similarity_score,
    stitch_images_vertically
)
from gen import QAGenerator


def retrieve_image(embedder, queries, data_dic, embeddings, top_n=3):
    query_embeddings = []
    for query in queries:
        query_embeddings.append(embedder.get_embedding(query, mod="text")[0].cpu())

    sim_mat = np.zeros((len(query_embeddings), len(embeddings)))
    for i in range(len(query_embeddings)):
        for j in range(len(embeddings)):
            sim_mat[i, j] = compute_similarity_score(query_embeddings[i], embeddings[j])

    max_indices = np.argsort(sim_mat, axis=1)[:, -top_n:]
    # print("Top 3 indices of maximum similarity values per row:", max_indices)
    results = []
    for idx in max_indices:
        tmp = []
        for i in idx:
            tmp.append(data_dic['images'][i])
        img = stitch_images_vertically(tmp)
        results.append(img)

    # print(results)
    return results


def retrieve_ocr_text(embedder, queries, data_dic, embeddings, top_n=3):
    query_embeddings = []
    for query in queries:
        query_embeddings.append(embedder.get_embedding(query, mod="text")[0].cpu())

    sim_mat = np.zeros((len(query_embeddings), len(embeddings)))
    for i in range(len(query_embeddings)):
        for j in range(len(embeddings)):
            sim_mat[i, j] = compute_similarity_score(query_embeddings[i], embeddings[j])

    max_indices = np.argsort(sim_mat, axis=1)[:, -top_n:]
    # print("Top 3 indices of maximum similarity values per row:", max_indices)
    results = []
    for idx in max_indices:
        tmp = []
        for i in idx:
            tmp.append(data_dic['texts'][i])
        text = '\n'.join(tmp)
        results.append(text)

    # print(results)
    return results


def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


if __name__ == '__main__':
    data_file = "/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/dmv.pdf"
    json_data_file = "./data/image_question_0.json"

    model, processor = create_model(model_type="colSmol")
    embedder = ImageTextEmbedder(model, processor)
    n_data = 10

    output_file = f'../db/{data_file.split("/")[-2]}.pickle'
    filetype = data_file.split(".")[-1].strip()

    # data_dic = save_db(
    #     embedder=embedder,
    #     data_file=data_file,
    #     output_file=output_file,
    #     filetype=filetype,
    #     n_data=n_data
    # )

    image_embeddings, text_embeddings, data_dic = load_db(output_file)

    client = OpenAI()
    QAGenerator = QAGenerator(client=client, model="gpt-4o-mini")
    dataset = load_json_dataset(json_data_file)

    n = 0
    acc = 0
    for key in tqdm(list(dataset.keys())[:100]):
        n += 1
        query = dataset[key]["question"]
        option = dataset[key]["options"]
        answer = dataset[key]["correct_answer"]

        image_results = retrieve_image(
            embedder=embedder,
            queries=[query],
            data_dic=data_dic,
            embeddings=image_embeddings,
            top_n=3
        )

        text_results = retrieve_ocr_text(
            embedder=embedder,
            queries=[query],
            data_dic=data_dic,
            embeddings=text_embeddings,
            top_n=3
        )

        response = QAGenerator.response(
            query=query,
            image=image_results[0],
            options=option,
            # ocr_content=text_results[0],
        )
        if len(answer) == 1 and answer in response:
            acc += 1
        else:
            if len(response) == 1 and response in answer:
                acc += 1
    acc = acc / n * 100
    print(f"Accuracy: {acc:.2f}%")
