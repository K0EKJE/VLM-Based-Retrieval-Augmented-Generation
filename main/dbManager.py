import numpy as np
from tqdm import tqdm
import torch

from preprocessor import (
    get_parquet_dataset,
    create_model,
    ImageTextEmbedder,
    PDFToImageConverter,
    get_pdf_dataset,
    compute_similarity_score,
)


def save_db(embedder, data_file, output_file, filetype, n_data=10):
    data_dic = {}
    if filetype == "parquet":
        data_dic = get_parquet_dataset(data_file, n_data)
    if filetype == "pdf":
        data_dic = get_pdf_dataset(data_file, n_data)

    images = data_dic["images"]
    texts = data_dic["texts"]
    queries = data_dic["queries"]

    image_embeddings, text_embeddings, query_embeddings = [], [], []
    for image, text, query in tqdm(zip(images, texts, queries), total=len(images), desc="Processing embeddings"):
        image_embeddings.append(embedder.get_embedding(image, mod="image")[0].cpu())
        text_embeddings.append(embedder.get_embedding(text, mod="text")[0].cpu())
        query_embeddings.append(embedder.get_embedding(query, mod="text")[0].cpu())

    torch.save(
        {"image_embeddings": image_embeddings,
         "text_embeddings": text_embeddings,
         "query_embeddings": query_embeddings},
        output_file)

    print(f"Embeddings saved to {output_file}")


def load_db(data_file_path):
    data = torch.load(data_file_path, weights_only=False)
    image_embeddings = data.get("image_embeddings", [])
    text_embeddings = data.get("text_embeddings", [])
    query_embeddings = data.get("query_embeddings", [])

    print(f"Successfully loaded {len(image_embeddings)} image embeddings and {len(text_embeddings)} text embeddings")

    return image_embeddings, text_embeddings, query_embeddings


if __name__ == "__main__":
    data_file = '/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/vidore/arxivqa_test_subsampled/test-00000-of-00001.parquet'
    # data_file = "/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/dmv.pdf"
    output_file = f'../db/{data_file.split("/")[-2]}.pickle'
    n_data = 10

    filetype = data_file.split(".")[-1].strip()
    data_dic = get_parquet_dataset(data_file, n_data)

    # colSmol / qwen
    model, processor = create_model(model_type="qwen")
    embedder = ImageTextEmbedder(model, processor)

    save_db(
        embedder=embedder,
        data_file=data_file,
        output_file=output_file,
        filetype=filetype,
        n_data=n_data
    )

    image_embeddings, text_embeddings, query_embeddings = load_db(output_file)
    sim_mat = np.zeros((len(image_embeddings), len(query_embeddings)))
    for i in range(len(image_embeddings)):
        for j in range(len(query_embeddings)):
            sim_mat[i, j] = compute_similarity_score(image_embeddings[j], query_embeddings[i])

    # print(sim_mat)

    max_indices = np.argsort(sim_mat, axis=1)[:, -3:]
    print("Top 3 indices of maximum similarity values per row:", max_indices)

    target = np.array([i for i in range(10)])
    recall_at_3 = np.mean([target in max_indices[i] for i, target in enumerate(target)])
    print("Recall@3:", recall_at_3)
