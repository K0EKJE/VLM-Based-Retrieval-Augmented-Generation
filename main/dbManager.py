import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
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
    print(filetype)

    # images = data_dic["images"]
    # texts = data_dic["texts"]
    # queries = data_dic["queries"]
    #
    # image_embeddings, text_embeddings, query_embeddings = [], [], []
    # for image, text, query in tqdm(zip(images, texts, queries), total=len(images), desc="Processing embeddings"):
    #     image_embeddings.append(embedder.get_embedding(image, mod="image")[0].cpu())
    #     text_embeddings.append(embedder.get_embedding(text, mod="text")[0].cpu())
    #     query_embeddings.append(embedder.get_embedding(query, mod="text")[0].cpu())
    #
    # torch.save(
    #     {"image_embeddings": image_embeddings,
    #      "text_embeddings": text_embeddings,
    #      "query_embeddings": query_embeddings},
    #     output_file)

    images = data_dic["images"]
    texts = data_dic["texts"]

    image_embeddings, text_embeddings = [], []
    for image, text in tqdm(zip(images, texts), total=len(images), desc="Processing embeddings"):
        image_embeddings.append(embedder.get_embedding(image, mod="image")[0].cpu())
        text_embeddings.append(embedder.get_embedding(text, mod="text")[0].cpu())

    torch.save(
        {"image_embeddings": image_embeddings,
         "text_embeddings": text_embeddings,
        "data_dic": data_dic},
        output_file)

    print(f"Embeddings saved to {output_file}")
    return data_dic


def load_db(data_file_path):
    data = torch.load(data_file_path, weights_only=False)
    image_embeddings = data.get("image_embeddings", [])
    text_embeddings = data.get("text_embeddings", [])
    data_dic = data.get("data_dic", {})
    # query_embeddings = data.get("query_embeddings", [])

    print(f"Successfully loaded {len(image_embeddings)} image embeddings and {len(text_embeddings)} text embeddings")

    return image_embeddings, text_embeddings, data_dic  # , query_embeddings


def stitch_images_vertically(image_list):
    width = image_list[0].width
    total_height = sum(img.height for img in image_list)

    stitched_image = Image.new('RGB', (width, total_height))

    y_offset = 0
    for img in image_list:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height

    return stitched_image


if __name__ == "__main__":
    # data_file = '/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/vidore/arxivqa_test_subsampled/test-00000-of-00001.parquet'
    data_file = "/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/dmv.pdf"
    output_file = f'../db/{data_file.split("/")[-2]}.pickle'
    n_data = 10
    top_n = 3

    filetype = data_file.split(".")[-1].strip()

    # colSmol / qwen
    model, processor = create_model(model_type="colSmol")
    embedder = ImageTextEmbedder(model, processor)

    data_dic = save_db(
        embedder=embedder,
        data_file=data_file,
        output_file=output_file,
        filetype=filetype,
        n_data=n_data
    )

    image_embeddings, text_embeddings, data_dic = load_db(output_file)
    queries = ["What type of driver's license is primarily needed for driving noncommercial vehicles in California?"]
    query_embeddings = []
    for query in queries:
        query_embeddings.append(embedder.get_embedding(query, mod="text")[0].cpu())

    sim_mat = np.zeros((len(query_embeddings), len(image_embeddings)))
    for i in range(len(query_embeddings)):
        for j in range(len(image_embeddings)):
            sim_mat[i, j] = compute_similarity_score(query_embeddings[i], image_embeddings[j])

    # print(sim_mat)

    max_indices = np.argsort(sim_mat, axis=1)[:, -top_n:]
    print("Top 3 indices of maximum similarity values per row:", max_indices)
    rag_docs = []
    for idx in max_indices:
        tmp = []
        for i in idx:
            tmp.append(data_dic['images'][i])
        img = stitch_images_vertically(tmp)
        rag_docs.append(img)
    print(rag_docs)
    # rag_docs[0][0].show()
    # target = np.array([i for i in range(n_data)])
    # recall_at_3 = np.mean([target in max_indices[i] for i, target in enumerate(target)])
    # print("Recall@3:", recall_at_3)
