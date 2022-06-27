# FLYNC - FLY Non-Coding gene discovery & classification

FLYNC is an end-to-end software pipeline that takes reads from transcriptomic experiments and outputs a curated list of new **non-coding** genes as classified by a pre-trained **Machine-Learning model**.  

## Pipeline overview
1. **Gather required information** for running the software.
2. **Read alignment** to reference genome.
3. **Transcriptome assembly**.
4. Assessment of **coding-probability** for new transcripts.
5. Determination of **non-coding gene types** (antisense, intronic-overlap, intergenic).
6. **Differential transcript expression**, if applicable to dataset.
7. Extract **genomic and sequence-level features** for the candidate non-coding genes.
8. **Machine-learning classification** of candidate non-coding transcripts.

## Installation

### Docker
Docker images of the pipeline are available at rfcdsantos/flync repository.

```
docker pull rfcdsantos/flync
```
### Conda
You can also clone the repository and install all the required environments to run locally using Anaconda. Just clone the repo and run the script the bash script `conda-env`.

```
git clone https://github.com/homemlab/flync.git
cd flync
bash conda-env
```
## Usage
You can see the CLI FLYNC help by running `flync --help` or subcommand specific help with `flync <subcommand> --help`, either with Docker or Conda.

### Docker
To use the `docker` image image be sure to map a local folder to keep the results in. FLYNC allows (and encourages) to set an output folder for the results.  
Example:
```
docker run --rm -v $PWD:/data rfcdsantos/flync flync sra -l test/list.txt -m test/metadata.csv -o /data/test_run
```
### Conda
To use locally with `conda` run `flync` from the cloned directory with the required arguments.  
Similarly as in with `docker`:
```
./flync sra -l test/list.txt -m test/metadata.csv -o ./test_run
```


(...) In development