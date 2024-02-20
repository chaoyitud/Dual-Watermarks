""" VLLM Server """

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)

from watermark_benchmark.utils.classes import (
    ConfigSpec,
    Generation,
    WatermarkSpec,
)
from watermark_benchmark.utils.stats import Stats

from watermark_benchmark.servers.server import Server


class HFServer(Server, LogitsProcessor):
    """
    A Hugging Face based watermarking server
    """

    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
        """
        Initializes the HF server.

        Args:
        - config (Dict[str, Any]): A dictionary containing the configuration of the model.
        - **kwargs: Additional keyword arguments.
        """
        model = config.model
        print(f"Loading model {model}")
        self.server = AutoModelForCausalLM.from_pretrained(model, device_map="auto",
                                                           torch_dtype=torch.bfloat16,
                                                           offload_folder="offload", )

        print(f"Loading tokenizer {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"
        self.devices = [[i for i in range(torch.cuda.device_count())][0]]
        self.server = self.server.to(self.devices[0])
        self.watermark_engine = None
        self.batch_size = config.hf_batch_size

        print("batch size is ", self.batch_size)
        self.current_batch = 0
        self.current_offset = 0
        self.watermark_spec = None

    def install(self, watermark_engine, watermark_spec: Optional[WatermarkSpec] = None) -> None:
        """
        Installs the watermark engine.

        Args:
        - watermark_engine (Any): The watermark engine.
        """
        self.watermark_engine = watermark_engine

        if watermark_spec is not None and watermark_spec.generator.startswith('contrastive'):
            self.batch_size = 1
            self.watermark_engine.install_model(self.server)
        self.watermark_spec = watermark_spec

    def __call__(self, input_ids, scores):
        # Apply watermarking
        ids = [
            self.current_offset + (self.current_batch * self.batch_size) + i
            for i in range(self.batch_size)
        ]
        self.stats.update(scores, ids)
        if self.watermark_engine is not None:
            scores = self.watermark_engine.process(scores, input_ids, ids)
        return scores

    def run(
            self,
            inputs: List[str],
            config: ConfigSpec,
            temp: float,
            keys: Optional[List[int]] = None,
            watermark_spec: Optional[WatermarkSpec] = None,
            use_tqdm=True,
    ) -> List[Generation]:
        """
        Runs the server.

        Args:
        - inputs (List[str]): A list of input strings.
        - config (ConfigSpec): The configuration.
        - temp (float): The temperature.
        - keys (Optional[List[int]]): A list of keys.
        - watermark_spec (Optional[WatermarkSpec]): The watermark specification.
        - use_tqdm (bool): A boolean indicating whether to use tqdm.

        Returns:
        - List[Generation]: A list of generations.
        """
        # Setup logit processor
        processors = LogitsProcessorList()
        processors.append(self)

        # Run
        generations = []
        self.stats = Stats(len(inputs), temp)
        print("Number of inputs in this task: ", len(inputs))
        while True:
            try:
                self.current_offset = len(generations)
                for i in tqdm(
                        range(0, (len(inputs) - len(generations)) // self.batch_size),
                        total=(len(inputs) - len(generations)) // self.batch_size,
                        disable=not use_tqdm,
                ):
                    self.current_batch = i
                    batch_inputs = inputs[
                                   self.current_offset
                                   + (i * self.batch_size): self.current_offset
                                                            + ((i + 1) * self.batch_size)
                                   ]
                    # show batch_inputs size
                    batch = self.tokenizer(
                        batch_inputs,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.devices[0])

                    # transform to half precision

                    if self.watermark_spec is not None and self.watermark_spec.generator.startswith('contrastive'):

                        outputs = self.watermark_engine.generate(batch, max_gen_len=config.max_new_tokens,
                                                                 ids=[i for i in range(self.current_offset
                                   + (i * self.batch_size), self.current_offset
                                                            + ((i + 1) * self.batch_size))])
                    else:
                        outputs = self.server.generate(
                            **batch,
                            temperature=temp,
                            max_length=config.max_new_tokens + batch.input_ids.shape[1],
                            num_return_sequences=config.num_return_sequences,
                            do_sample=(temp > 0),
                            logits_processor=processors,
                        )

                        # decode to string
                        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                        # outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                        # remove the prompt
                        outputs = [output[len(prompt):] for output, prompt in zip(outputs, batch_inputs)]

                    for j, output in enumerate(outputs):
                        prompt = batch_inputs[j]
                        result = output
                        # print("prompt: ", prompt)
                        # print("result: ", result)
                        generations.append(
                            # watermark_specm, key, rng, idx, prompt, result, verifier, stats, stats2, stats3, temp
                            Generation(
                                watermark_spec
                                if watermark_spec is not None
                                else None,
                                keys[
                                    self.current_offset
                                    + (i * self.batch_size)
                                    + j
                                    ]
                                if keys is not None
                                else None,
                                None,
                                self.current_offset + (i * self.batch_size) + j,
                                prompt,
                                result,
                                None,
                                None,
                                None,
                                *self.stats[
                                    self.current_offset
                                    + (i * self.batch_size)
                                    + j
                                    ],
                                temp,
                                None
                            )
                        )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.batch_size > 1:
                    torch.cuda.empty_cache()
                    self.batch_size = self.batch_size // 2
                    continue
                else:
                    raise e
            break

        self.current_batch = 0
        self.current_offset = 0
        return generations

    def tokenizer(self):
        """
        Returns the tokenizer.
        """
        return self.tokenizer
