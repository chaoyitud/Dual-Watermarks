# Duwak

Welcome to the official repository for the paper "Duwak: Dual Watermarks in Large Language Models."

## Installation

To install the necessary dependencies, run the following command:

```bash
./install.sh
```

## Usage

To use Duwak for embedding dual watermarks in large language models, follow these steps:

1. **Prepare Watermark Specifications:**
   - Create a file with the specifications for the watermarks you wish to embed. Each line should contain a JSON object that describes the parameters of a watermark.
   - Example watermark specifications can be found in `./run/watermark_specs`. 
   - For a complete definition of the WatermarkSpec object, refer to `./src/watermark-benchmark/utils/classes.py`.

2. **Configure Settings:**
   - Set up your configuration file based on the provided example in `./run/config.yml`.

3. **Set Python Path:**
   - Include the watermark_benchmark in your Python path with the following command:
     ```bash
     export PYTHONPATH="path_to_src:$PYTHONPATH"
     ```

4. **Run Benchmark:**
   - Execute the benchmark script with the following command:
     ```bash
     python3 -m ../src/watermark-benchmark/pipeline/run_all.py config.yml
     ```

## Our Watermarks

- Duwak watermarking scheme is implemented in `./src/watermark-benchmark/watermark/schemes/contrastive_distri.py`.
- This codebase is an extension of the MarkMyWords project. The original codebase is available at [MarkMyWords GitHub Repository](https://github.com/wagner-group/MarkMyWords).

 