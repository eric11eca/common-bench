import torch
import logging
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from simcse import SimCSE
from sentence_transformers import SentenceTransformer

from common_bench.dataset import CommonDataset
from common_bench.utils.py_io import *

embeddings = {
    'nli_mean': 'roberta-large-nli-mean-tokens',
    'simcse': 'princeton-nlp/sup-simcse-roberta-large',
}

util_logger = logging.getLogger(
    'common_bench.dynamic_icl'
)


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def decode(args, data_loader, tok=None, model=None):
    embeddings = []

    if args.encoder_name == 'roberta-base' or args.encoder_name == 'roberta-large':
        print("Using non Sentence Transformer models")
        for corpus_tmp in tqdm(data_loader):
            encoding = tok.batch_encode_plus(
                corpus_tmp, padding=True, truncation=True)
            sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
            sentence_batch, attn_mask = torch.LongTensor(
                sentence_batch), torch.LongTensor(attn_mask)

            with torch.no_grad():
                embedding_output_batch = model(sentence_batch, attn_mask)
                if args.embed_type == 'mean':
                    sentence_embeddings = mean_pooling(
                        embedding_output_batch, attn_mask)
                elif args.embed_type == 'CLS':
                    sentence_embeddings = embedding_output_batch[0][:, 0, :]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())
            del sentence_batch, attn_mask, embedding_output_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Using Sentence Transformer models")
        for corpus_tmp in tqdm(data_loader):
            sentence_embeddings = model.encode(corpus_tmp)
            embeddings.append(sentence_embeddings)

    return np.concatenate(embeddings, axis=0)


def demonstration_search(args, examples, problems, encoder, reversed=False):
    demos = [x["question"] for x in examples]
    inputs = [x["question"] for x in problems]

    demo_loader = DataLoader(demos, batch_size=16, shuffle=False)
    input_loader = DataLoader(inputs, batch_size=16, shuffle=False)

    emb_train = decode(args, demo_loader, model=encoder)
    emb_dev = decode(args, input_loader, model=encoder)

    if args.metric == "euclidean":
        nbrs = NearestNeighbors(
            n_neighbors=args.num_neighbors,
            algorithm='ball_tree',
            n_jobs=-1
        ).fit(emb_train)
        _, indices = nbrs.kneighbors(emb_dev)
    elif args.metric == "cosine":
        dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
        if reversed:
            _, indices = torch.topk(-torch.from_numpy(dist_matrix),
                                    k=args.num_neighbors, dim=-1)
        else:
            _, indices = torch.topk(torch.from_numpy(
                dist_matrix), k=args.num_neighbors, dim=-1)
        indices = indices.numpy()

    return indices


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="tomi")
    parser.add_argument("--task", default="tomi")
    parser.add_argument("--encoder_name", default="simcse")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--num_neighbors", default=16, type=int)
    args = parser.parse_args()
    args.model_name_or_path = "t5"

    encoder_repo = embeddings[args.encoder_name]
    if args.encoder_name == 'simcse':
        print("Using SimCSE models")
        encoder = SimCSE(encoder_repo)
    else:
        print("Using Sentence Transformer models")
        encoder = SentenceTransformer(encoder_repo)

    tokenizer = encoder.tokenizer
    exmple_pool = CommonDataset(
        util_logger,
        args,
        tokenizer,
        args.data_dir,
        data_type="train",
        is_training=False,
        ic_examples=[]
    )

    example_dict = {}
    for example in exmple_pool.data:
        example_dict[example["guid"]] = example["question"]

    test_data = CommonDataset(
        util_logger,
        args,
        tokenizer,
        args.data_dir,
        data_type="test",
        is_training=False,
        ic_examples=[]
    )

    examples_selected = demonstration_search(
        args, exmple_pool.data, test_data.data, encoder
    )

    data_w_examples = []
    for i, problem in enumerate(test_data.data):
        examples = [exmple_pool.data[j]["guid"] for j in examples_selected[i]]
        for e in examples:
            assert e in example_dict
        examples.reverse()
        data_w_examples.append({
            "guid": problem["guid"],
            "examples": examples
        })
    write_jsonl(data_w_examples,
                f"data/{args.dataset}/train_{args.encoder_name}.jsonl")
    print(
        "Generated examples to data/{args.dataset}/train_{args.encoder_name}.jsonl")
