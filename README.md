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
     cd run
     python3 -m ../src/watermark-benchmark/pipeline/run_all.py config.yml
     ```

## Our Watermarks

- Duwak watermarking scheme is implemented in `./src/watermark-benchmark/watermark/schemes/contrastive_distri.py`.
- This codebase is an extension of the MarkMyWords project. The original codebase is available at [MarkMyWords GitHub Repository](https://github.com/wagner-group/MarkMyWords).
## Docker Support

### Pull Image from Docker Hub and Tag (Optional):
To pull the pre-built image from Docker Hub and tag it as `duwak`, use the following commands:

```bash
docker pull ctrlmybgm/markword:1.0
docker tag ctrlmybgm/markword:1.0 duwak
```
### Build Image (Optional):
To build the Docker image from the source code, use the following command:

```bash
docker build -t duwak .
```

### Run Container:
To run the Docker container with your source code, replace `/path_to_your_source` with the actual path to your source code. Use the following command:

```bash
cd run
docker run -it --rm \\
    --gpus all \\
    -v /path_to_your_src:/app/src \\
    -v $(pwd):/project \\
    -w /project \\
    -e PYTHONPATH="/app/src:$PYTHONPATH" \\
    -e HF_TOKEN=your_HF_TOKEN \\
    duwak \\
    bash -c "python3 ./src/watermark_benchmark/pipeline/run_all.py ./run/config.yml"
```

This setup allows you to easily use the `duwak` image for Docker container operations. 