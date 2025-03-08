from tqdm import tqdm
from pdf2image import convert_from_path
import cv2
import torch
import numpy as np
from PIL import Image
import pytesseract
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from datasets import load_dataset


class PDFToImageConverter:
    def __init__(self, dpi=200):
        self.dpi = dpi

    def extract_images(self, pdf_path: str):
        images = convert_from_path(pdf_path, dpi=self.dpi)
        return images


class ImageOCR:
    def __init__(self, lang='eng'):
        self.lang = lang

    def extract_text(self, image: Image.Image):
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text.strip()

    def batch_extract_text(self, images):
        contexts = []
        for image in tqdm(images):
            contexts.append(self.extract_text(image))
        return contexts


class ImageTextEmbedder:
    def __init__(self, model, processor, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"use {self.device}")

        cache_dir = "./models/colqwen2-v1.0"

        self.processor = processor
        self.model = model

    def get_embedding(self, X, mod):
        if mod == "text":
            batch = self.processor.process_queries([X]).to(self.device)
        else:
            batch = self.processor.process_images([X]).to(self.device)
        with torch.no_grad():
            features = self.model(**batch)
        # features = features / features.norm(p=2, dim=-1, keepdim=True)
        # embeddings = features[0].to(torch.float32).cpu().numpy()
        return features

    def batch_embedding(self, X_list, mod):
        if mod == "text":
            batch = self.processor.process_queries(X_list).to(self.device)
        else:
            batch = self.processor.process_images(X_list).to(self.device)
        with torch.no_grad():
            features = self.model(**batch)
        # features = features / features.norm(p=2, dim=-1, keepdim=True)
        # embeddings = features[0].to(torch.float32).cpu().numpy()
        # return [emb for emb in embeddings]

        return features


def create_model(model_type="qwen"):
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if model_type == "qwen":
        model_name = "vidore/colqwen2-v1.0"
        cache_dir = "./models/colqwen2-v1.0"

        # model_name = "tsystems/colqwen2-2b-v1.0"
        # cache_dir = "./models/colqwen2-2b-v1.0"

        model = ColQwen2.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        processor = ColQwen2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        return model, processor

    # if model_type == "qwen2.5":
    #     model_name = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
    #     cache_dir = "./models/ColQwen2.5-3b-multilingual-v1.0"
    #     model = ColQwen2_5.from_pretrained(
    #         model_name,
    #         cache_dir=cache_dir,
    #         torch_dtype=torch.bfloat16,
    #         device_map=device
    #     ).eval()
    #
    #     processor = ColQwen2_5_Processor.from_pretrained(model_name, cache_dir=cache_dir)
    #     return model, processor

    if model_type == "colSmol":
        model_name = "vidore/colSmol-500M"
        cache_dir = "./models/colSmol-500M"

        model = ColIdefics3.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColIdefics3Processor.from_pretrained(model_name)

        return model, processor


def compute_similarity_score(text_embeddings, image_embeddings):
    emb1 = text_embeddings.unsqueeze(0)
    emb2 = image_embeddings.unsqueeze(0)
    similarity_matrix = torch.einsum("bnd,csd->bcns", emb1, emb2)
    max_sim_scores = similarity_matrix.max(dim=3)[0]
    similarity_score = max_sim_scores.sum(dim=2)

    return similarity_score


def explore_pdf():
    pdf_path = "/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/dmv.pdf"
    converter = PDFToImageConverter(dpi=200)

    images = converter.extract_images(pdf_path)
    print(len(images))

    ocr = ImageOCR(lang='eng')

    image_list = images[:2]
    text_list = ocr.batch_extract_text(image_list)

    model, processor = create_model()
    embedder = ImageTextEmbedder(model, processor)

    image_embeddings = embedder.batch_embedding(image_list, mod="image")
    print("image embedding dim:", image_embeddings[0].shape)

    ocr_text_embeddings = embedder.batch_embedding(text_list, mod="text")
    print("text embedding dim:", ocr_text_embeddings[0].shape)

    # similarity_score1 = compute_similarity_score(ocr_text_embeddings[0], image_embeddings[0])
    # similarity_score2 = compute_similarity_score(ocr_text_embeddings[1], image_embeddings[0])
    #
    # print(f"sim score1: {similarity_score1.item()}")
    # print(f"sim score2: {similarity_score2.item()}")

    scores = processor.score_multi_vector(ocr_text_embeddings, image_embeddings)
    print(scores)


def explore_dataset():
    dataset = load_dataset('parquet',
                           data_files='/Users/hanboyu/Desktop/winter2025/cs224n/RAGSystem/data/vidore/arxivqa_test_subsampled/test-00000-of-00001.parquet')
    print(dataset)
    images = dataset['train']['image']
    queries = dataset['train']['query']
    options = dataset['train']['options']
    answers = dataset['train']['answer']

    print(len(images))

    ocr = ImageOCR(lang='eng')
    images[5].show()

    # image_list = images[:2]
    # text_list = ocr.batch_extract_text(image_list)
    #
    # model, processor = create_model()
    # embedder = ImageTextEmbedder(model, processor)
    #
    # image_embeddings = embedder.batch_embedding(image_list, mod="image")
    # print("image embedding dim:", image_embeddings[0].shape)
    #
    # ocr_text_embeddings = embedder.batch_embedding(text_list, mod="text")
    # print("text embedding dim:", ocr_text_embeddings[0].shape)
    #
    # scores = processor.score_multi_vector(ocr_text_embeddings, image_embeddings)
    # print(scores)


def get_parquet_dataset(data_files, n_data):
    dataset = load_dataset('parquet', data_files=data_files)
    print(dataset)

    images = dataset['train']['image'][:n_data]
    queries = dataset['train']['query'][:n_data]
    options = dataset['train']['options'][:n_data]
    answers = dataset['train']['answer'][:n_data]
    ocr = ImageOCR(lang='eng')
    texts = ocr.batch_extract_text(images)

    print(f"num of data: {len(images)}")

    dic = {
        "images": images,
        "texts": texts,
        "queries": queries,
        "options": options,
        "answers": answers
    }

    return dic


def get_pdf_dataset(pdf_path, n_data):
    converter = PDFToImageConverter(dpi=200)

    images = converter.extract_images(pdf_path)[:n_data]
    print(len(images))

    ocr = ImageOCR(lang='eng')
    texts = ocr.batch_extract_text(images)

    print(f"num of data: {len(images)}")

    dic = {
        "images": images,
        "texts": texts,
    }

    return dic


if __name__ == "__main__":
    explore_pdf()
