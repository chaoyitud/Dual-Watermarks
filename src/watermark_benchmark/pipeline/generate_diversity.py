import json
import multiprocessing
import os
import random
import signal
import sys
from dataclasses import replace

from tqdm import tqdm

from watermark_benchmark import WatermarkSpec
from watermark_benchmark.utils import (
    get_output_file,
    get_server_args,
    load_config,
    setup_randomness,
)
from watermark_benchmark.utils.generation_prompts import raw_prompts


def writer_process(queue, config, w_count, data):
    """
    This function is a process that writes the generated outputs to a file.

    Args:
        queue (multiprocessing.Queue): A queue containing the generated outputs.
        config (Config): The configuration object.
        w_count (int): The number of watermark generations to write to the file.
    """

    outfilepath = get_output_file(config)
    for _ in tqdm(range(w_count), total=w_count, desc="Generations"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return
        json_temp = []
        for i, gen in enumerate(task):
            # get the prompt
            prompt = gen.prompt
            # get the generated text
            text = gen.response

            # get the watermark
            watermark = gen.watermark
            reference = data[i]['reference_text']
            assert prompt == data[i]['prefix_text']

            try:
                parts = [
                    watermark.generator, watermark.temp, watermark.beta, watermark.alpha, watermark.gamma
                ]
            except AttributeError:
                parts = [None, None, None, None, None]
            filename = '_'.join([str(p) for p in parts])

            outfilepath_new = outfilepath + f'/{filename}.json'
            # write to file
            json_temp.append({"prefix_text": prompt, "reference_text": reference, "generated_result": {"0": text}})
        with open(outfilepath_new, "a", encoding="utf-8") as outfile:
            json.dump(json_temp, outfile, indent=4)
            outfile.write("\n")


def gen_process(
    config, tasks, writer_queue, device, prompts, custom_builder=None
):
    """
    This function is a process that generates watermarked text.

    Args:
        config (Config): The configuration object.
        tasks (list): A list of tuples containing the watermark, keys, and temperature.
        writer_queue (multiprocessing.Queue): A queue to store the generated outputs.
        device (int): The device to use for generating the watermarked text.
        prompts (list): A list of prompts to use for generating the watermarked text.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Imports
    import torch

    from watermark_benchmark.servers import get_model
    from watermark_benchmark.utils.bit_tokenizer import Binarization
    from watermark_benchmark.watermark import get_watermark

    setup_randomness(config)

    # Setup server
    server = get_model(config.engine, config, **get_server_args(config))
    tokenizer = server.tokenizer
    binarizer = Binarization(
        tokenizer,
        server.devices,
        use_huffman_coding=config.huffman_coding is not None,
        huffman_coding_path=config.huffman_coding,
    )

    def run_instance(watermark, keys, temp):
        # Setup watermark
        print("Generating watermark: {}".format(watermark))
        setup_randomness(config)

        print("Loading tokenizer {}...".format(config.model))
        watermark_engine = (
            get_watermark(
                watermark,
                tokenizer,
                binarizer,
                server.devices,
                keys,
                builder=custom_builder,
            )
            if watermark is not None
            else None
        )

        # Install and run
        print("Installing watermark engine...")
        server.install(watermark_engine, watermark_spec=watermark)
        print("Running server...")
        outputs = server.run(prompts, config, temp, keys, watermark)

        writer_queue.put(outputs)

        # Reset server
        server.install(None)
        torch.cuda.empty_cache()

    for t in tasks:
        print("Running task: {}".format(t))
        run_instance(*t)


def run(config_file, watermarks=None, custom_builder=None):
    """
    This function runs the watermark generation process.

    Args:
        config_file (str): The path to the configuration file.
        watermarks (list): A list of watermarks to use for generating the watermarked text.
    """
    from watermark_benchmark.utils.classes import Generation, WatermarkSpec
    from watermark_benchmark.utils.standardize import standardize

    # Load config
    if isinstance(config_file, str):
        config = load_config(config_file)
    else:
        config = config_file
    setup_randomness(config)

    # Setup watermarks
    if not watermarks:
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(line.strip()), tokenizer=config.model
                )
                for line in infile.read().split("\n")
                if len(line)
            ]

    # Generate tasks
    with open('diversity.json', 'r') as file:
        data = json.load(file)
    print(data[0])
    data = data[:config.prompt_size]
    prompts = [data[i]['prefix_text'] for i in range(len(data))]


    print("Generating {} watermarks...".format(len(watermarks)))
    print("Generating {} prompts...".format(len(prompts)))

    unique_temps, tasks = set(), []
    for watermark in watermarks:
        # Randomly sample key if needed
        if watermark.randomize:
            keys = [random.randint(0, 1000000) for _ in prompts]
        else:
            keys = [watermark.secret_key for _ in prompts]

        # Add task
        tasks.append((watermark, keys))
        unique_temps.add(watermark.temp)


    all_tasks = [(watermark, keys, watermark.temp) for watermark, keys in tasks]
    if config.baseline:
        all_tasks.extend([(None, None, temp) for temp in unique_temps])


    filtered_tasks = []
    for w, k, t in all_tasks: # w: watermark, k: keys, t: temp
        if w is not None and str(w.to_dict(True, True)):
            filtered_tasks.append((w, k, t))
        elif w is None and str(t):
            filtered_tasks.append((w, k, t))

    if not len(filtered_tasks):
        return

    # Setup processes
    ct = 1 + (len(filtered_tasks) // len(config.get_devices()))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()
    #random.shuffle(filtered_tasks)

    print("Generating {} tasks...".format(len(filtered_tasks)))

    for idx, device in enumerate(config.get_devices()):
        local = filtered_tasks[idx * ct : (idx + 1) * ct]
        processes.append(
            multiprocessing.Process(
                target=gen_process,
                args=(
                    config,
                    local,
                    writer_queue,
                    device,
                    prompts,
                    custom_builder,
                ),
            )
        )
        processes[-1].start()

    writer = multiprocessing.Process(
        target=writer_process, args=(writer_queue, config, len(filtered_tasks), data)
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

    return


def main():
    config = load_config(sys.argv[1])
    with open(config.watermark, encoding="utf-8") as infile:
        watermarks = [
            replace(WatermarkSpec.from_str(l.strip()), tokenizer=config.model)
            for l in infile.read().split("\n")
            if len(l)
        ]
    generate(sys.argv[1], watermarks)


def generate(config_file, watermarks, custom_builder=None):
    """
    Standalone generation procedure.

    Args:
        config_file (str or ConfigSpec): Config file or path to config file
        watermarks (list): A list of watermark specs to use for generating the watermarked text.
        custom_builder (function): A custom builder function to use for generating the watermarks.
        Set to none if not using custom watermarks.

    If config does not contain a results directory, it will be created.
    This procedure sets the appropriate input and output files for the generation procedure.

    Return:
        generations (list): A list of generations.
        config (ConfigSpec): The updated configuration object.
    """
    #if multiprocessing.get_start_method() != "spawn":
    #    multiprocessing.set_start_method("spawn")
    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    config.input_file = None
    config.output_file = config.results
    try:
        os.mkdir(config.results)
    except Exception:
        pass
    return run(config, watermarks, custom_builder), config

if __name__ == "__main__":
    main()

