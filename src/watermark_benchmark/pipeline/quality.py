import copy
import multiprocessing
import os
import re
import signal
import sys
from dataclasses import replace
import torch
import openai
from openai import OpenAI
from tqdm import tqdm

from watermark_benchmark.utils import (
    get_input_file,
    get_output_file,
    load_config,
    setup_randomness,
)
from watermark_benchmark.utils.classes import Generation


def writer_process(queue, config, w_count):
    from watermark_benchmark.utils import get_output_file, setup_randomness

    setup_randomness(config)
    outfilepath = get_output_file(config)

    for _ in tqdm(range(w_count), total=w_count, desc="Rating"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        with open(outfilepath, "a") as outfile:
            outfile.write("\n".join(str(gen) for gen in task) + "\n")


def rating_process_openai(config, generations, writer_queue):
    """
    Runs the model on the given generations and calculates the rating for each generation.

    Args:
        config (Config): Configuration object.
        generations (List[Generation]): List of generations to rate.
        writer_queue (Queue): Queue to write the rated generations to.
        device (int): Index of the GPU device to use.

    Returns:
        None
    """
    outputs = []
    client = OpenAI(api_key=config.openai_key)
    for generation in tqdm(generations, total=len(generations), desc="Rating"):
        prompt = generation.prompt.replace("[/INST]", "").replace("[INST]", "").replace("<<SYS>>", "").replace(
            "<</SYS>>", "").replace(
            "You are a helpful assistant. Always answer in the most accurate way.",
            "",
        ).strip()
        response = generation.response
        response = response[:len(response)//2]
        # cut off the response if it is too long
        messages = [{"role": "system",
                     "content": "You are given a prompt and a response, and you need grade the response out of 100 based on: Accuracy (20 points) - correctness and relevance to the prompt; Detail (20 points) - comprehensiveness and depth; Grammar and Typing (30 points) - grammatical and typographical accuracy; Vocabulary (30 points) - appropriateness and richness. Deduct points for shortcomings in each category. Give a total grade at the first line of the response."},
                    {"role": "user", "content": f"Prompt:{prompt}\nResponse:{response}"}]
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=8
            )
            grade = completion.choices[0].message.content
        except Exception:
            grade = "0"
            print(f"message: {messages}")
        outputs.append(grade)
        print(f"grade: {grade}")

    num_regex = re.compile("([0-9]+\.*[0-9]*)")

    # Parse outputs
    for idx, gen in enumerate(outputs):
        raw = 0.0
        matches = re.findall(num_regex, gen)[:2]
        if len(matches) >= 2:
            try:
                raw = float(matches[0]) / float(matches[1])
            except Exception:
                raw = 0
        elif len(matches) == 1:
            raw = float(matches[0]) / 100
        raw = max(min(raw, 1), 0)

        if idx >= len(generations):
            print(
                "Warning: Received more outputs than generations ({} vs {})".format(
                    len(outputs), len(generations)
                )
            )
            break
        generations[idx] = replace(generations[idx], rating=raw)

    # Write to file
    writer_queue.put(generations)


def rating_process(config_input, generations, writer_queue, device):
    """
    Runs the model on the given generations and calculates the rating for each generation.

    Args:
        config (Config): Configuration object.
        generations (List[Generation]): List of generations to rate.
        writer_queue (Queue): Queue to write the rated generations to.
        device (int): Index of the GPU device to use.

    Returns:
        None
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Imports
    import torch

    from watermark_benchmark.servers import get_model
    from watermark_benchmark.utils import get_server_args, setup_randomness

    torch.set_num_threads(1)
    config = copy.deepcopy(config_input)
    setup_randomness(config)

    # Setup server
    config.model = "meta-llama/Llama-2-7b-chat-hf"
    config.max_new_tokens = 8
    config.dtype = "float16"
    config.num_return_sequences = 1
    inference_engine = config.engine
    server = get_model(inference_engine, config, **get_server_args(config))
    tokenizer = server.tokenizer
    # tokenizer.pad_token = tokenizer.eos_token
    # pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    tokenizer.pad_token = '[PAD]'
    tasks = []
    for generation in generations:
        tasks.append(
            prompt.format(
                generation.prompt.replace("[/INST]", "")
                .replace("[INST]", "")
                .replace("<<SYS>>", "")
                .replace("<</SYS>>", "")
                .replace(
                    "You are a helpful assistant. Always answer in the most accurate way.",
                    "",
                )
                .strip(),
                generation.response,
            )
        )

    # Clip sequences that are too long
    max_token_length = 4000
    for i in tqdm(range(len(tasks)), total=len(tasks), desc="Encoding"):
        task = tasks[i]
        if len(task) > max_token_length:
            encoded_task = tokenizer(task)["input_ids"]
            if len(encoded_task) > max_token_length:
                print(
                    "Warning: Task too long ({} tokens), clipping to {} tokens".format(
                        len(encoded_task), max_token_length
                    )
                )
                task = tokenizer.decode(encoded_task[:max_token_length])
        tasks[i] = task

    print("Encoding done. Ready for rating.")

    # Run model
    outputs = server.run(tasks, config, 0.0, use_tqdm=True)
    num_regex = re.compile("([0-9]+\.*[0-9]*)")

    # Parse outputs
    for idx, gen in enumerate(outputs):
        raw = 0.0
        matches = re.findall(num_regex, gen.response)[:2]
        if len(matches) >= 2:
            try:
                raw = float(matches[0]) / float(matches[1])
            except Exception:
                raw = 0
        elif len(matches) == 1:
            raw = float(matches[0]) / 100
        raw = max(min(raw, 1), 0)

        if idx >= len(generations):
            print(
                "Warning: Received more outputs than generations ({} vs {})".format(
                    len(outputs), len(generations)
                )
            )
            break
        generations[idx] = replace(generations[idx], rating=raw)

    # Write to file
    writer_queue.put(generations)


def run(config_file, generations=None):
    # load config
    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    setup_randomness(config)

    # load generations
    generations = (
        Generation.from_file(get_input_file(config))
        if not generations
        else generations
    )
    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)
    existing = {
        str(
            (
                g.watermark.to_dict(True, True)
                if g.watermark is not None
                else g.temp,
                g.id,
                g.attack,
            )
        )
        for g in Generation.from_file(outfilepath)
    }
    tasks = [
        g
        for g in generations
        if str(
            (
                g.watermark.to_dict(True, True)
                if g.watermark is not None
                else g.temp,
                g.id,
                g.attack,
            )
        )
           not in existing
    ]

    if not len(tasks):
        return

    ct = 1 + (len(tasks) // len(config.get_devices()))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()

    for idx, device in enumerate(config.get_devices()):
        local_gens = tasks[idx * ct: (idx + 1) * ct]
        if config.openai_quality:
            processes.append(
                multiprocessing.Process(
                    target=rating_process_openai,
                    args=(config, local_gens, writer_queue),
                )
            )
        else:
            processes.append(
                multiprocessing.Process(
                    target=rating_process,
                    args=(config, local_gens, writer_queue, device),
                )
            )
        processes[-1].start()

    writer = multiprocessing.Process(
        target=writer_process,
        args=(writer_queue, config, len(config.get_devices())),
    )
    writer.start()

    # Setup signal handler
    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in processes:
            p.terminate()
        writer.terminate()
        exit()

    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in processes:
        p.terminate()

    return Generation.from_file(get_output_file(config))


def main():
    multiprocessing.set_start_method("spawn")
    run(sys.argv[1])


prompt = "[INST] <<SYS>> You are given a prompt and a response, and you need grade the response out of 100 based on: Accuracy (20 points) - correctness and relevance to the prompt; Detail (20 points) - comprehensiveness and depth; Grammar and Typing (30 points) - grammatical and typographical accuracy; Vocabulary (30 points) - appropriateness and richness. Deduct points for shortcomings in each category. Give a total grade at the first line of the response. <</SYS>> Prompt: {}\nResponse: {}[/INST] Grade out of 100: "



def rate(config_file, generations):
    """
    Standalone perturb procedure.

    Args:
        config_file (str or ConfigSpec): Config file or path to config file
        generations (list of Generation or None): List of generations to perturb.

    If config does not contain a results directory, it will be created.
    This procedure sets the appropriate input and output files for the generation procedure.

    Return:
        generations (list): A list of generations.
        config (ConfigSpec): The updated configuration object.
    """
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn")

    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    config.input_file = config.results + "/perturbed{}.tsv".format(
        "_val" if config.validation else ""
    )
    config.output_file = config.results + "/rated{}.tsv".format(
        "_val" if config.validation else ""
    )

    if generations is None:
        generations = Generation.from_file(get_input_file(config))

    return run(config, generations), config

def rate_existing(config_file, generations):
    """
    Standalone perturb procedure.

    Args:
        config_file (str or ConfigSpec): Config file or path to config file
        generations (list of Generation or None): List of generations to perturb.

    If config does not contain a results directory, it will be created.
    This procedure sets the appropriate input and output files for the generation procedure.

    Return:
        generations (list): A list of generations.
        config (ConfigSpec): The updated configuration object.
    """

    #multiprocessing.set_start_method("spawn")

    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    config.input_file = config.results + "/detect{}.tsv".format(
        "_val" if config.validation else ""
    )
    config.output_file = config.results + "/rate_new_new{}.tsv".format(
        "_val" if config.validation else ""
    )

    if generations is None:
        generations = Generation.from_file(get_input_file(config))

    return run(config, generations), config

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    rate_existing(sys.argv[1], None)