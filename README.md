# FLYNC - FLY Non-Coding gene discovery & classification

FLYNC is an end-to-end software pipeline that takes reads from transcriptomic experiments and outputs a curated list of new **non-coding** genes as classified by a pre-trained **Machine-Learning model**.  

## Pipeline overview
1. **Gather required information** for running the software.
2. Read alignment to reference genome.
3. **Transcriptome assembly**.
4. Assessment of **coding-probability** for new transcripts.
5. Determination of **non-coding gene types** (antisense, intronic-overlap, intergenic).
6. **Differential transcript expression**, if applicable to dataset.
7. Extract **genomic and sequence-level features** for the candidate non-coding genes.
8. **Machine-learning classification** of candidate non-coding transcripts.

(...) In development