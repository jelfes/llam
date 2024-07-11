import os
import re
import json
import torch
import logging
import argparse
import transformers

from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import datetime
from llam.utils import extract_grouped_actantial_dict, ensure_directory
from llam.dataset import NewsDatasetSQL
from transformers import AutoTokenizer
from torch.backends import mps
from llam.constants import MODELS, TEMPLATES
from llam.config_local import DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--template",
    type=str,
    help="Templates to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="",
    help="Model to use during inference. Provide either a path to huggingface or local model.",
)
parser.add_argument(
    "--query",
    type=str,
    default="",
    help="Query to sql database",
)


args = vars(parser.parse_args())


# set device for training
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\n\nyeah, cuda-time!\n\n")
elif mps.is_available():
    device = torch.device("mps")
    print("\n\nMPS is the BEST!\n\n")
else:
    device = torch.device("cpu")
    print("\n\nNo GPU, you mad?\n\n")

model = args["model_path"]


# logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_root = f"{model}_{args['template_group']}_{timestamp}"

print(f"Logging to: {path_root}.log")

logging.basicConfig(
    filename=f"logs/{path_root}.log",
    encoding="utf-8",
    format="%(asctime)s %(message)s",
    level=logging.INFO,
)
logging.captureWarnings(True)
logging.info("Start logging")
logging.info(f"Summary:")
logging.info(f"Model:\t{model}")
logging.info(f"Prompt:\t{args['template']}")
logging.info(f"Query:\n{args['query']}")


# dtype
if device == torch.device("cpu"):
    # half precision not implemented on cpu
    data_type = torch.float32
else:
    # faster on gpu
    data_type = torch.float16


# model
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=data_type,
    device=device,
    trust_remote_code=True,
)

# prompt template
environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template(args["template"])

# data
data = NewsDatasetSQL(args["query"])


if __name__ == "__main__":

    # output
    MODEL_DIR = Path(DATA_DIR, "actantial_models", args["model"])
    PROMPT_DIR = Path(MODEL_DIR, args["template_group"])
    RUN_DIR = Path(PROMPT_DIR, timestamp)

    ensure_directory(MODEL_DIR)
    ensure_directory(PROMPT_DIR)
    ensure_directory(RUN_DIR)

    logging.info("Start inference")
    for row in tqdm(data.table.itertuples(), total=len(data)):
        logging.info(f"---------- Article {row.hash_id} ----------\n")
        actant_collection = {}

        prompt = template.render(article=row.full_text)

        sequences = pipeline(
            prompt,
            do_sample=False,
            temperature=1,
            top_p=1,
            num_return_sequences=1,
            max_new_tokens=300,
            return_full_text=False,
        )
        logging.info(f"Output:\n{sequences[0]['generated_text']}")

        actant_dict = extract_grouped_actantial_dict(sequences[0]["generated_text"])

        for actant, name in actant_dict.items():
            actant_collection[actant] = name

        if device == torch.device("mps"):
            torch.mps.empty_cache()
        elif device == torch.device("cuda:0"):
            torch.cuda.empty_cache()

        file_path = Path(RUN_DIR, f"{row.hash_id}.txt")

        logging.info(f"Writing output to: {file_path}")

        with open(file_path, "w") as output:
            json.dump(actant_collection, output)

        logging.info("##############################################################\n")
