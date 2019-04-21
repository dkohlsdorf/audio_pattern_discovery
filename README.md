# Audio Pattern Discovery

Pattern Discovery In Audio Collections in Rust

## Method

From each file we extract the cepstrum [1] in the following manner:
+ 0. Extract Sliding Window
+ 1. Compute DFT for each window
+ 2. Convolve DFT with triangular window with a stride of half the filter
+ 3. Compute log of filtered window
+ 4. Compute Cepstrum by computing the discrete cosine transform

The parameters needed so far are:
+ dft window
+ dft step
+ triangular window size

We then find slices where something `interesting` happens:
+ 0. For each cepstrum frame compute it's variance
+ 1. Smooth the variances in each sequence using a moving average
+ 2. Extract long sequences of high variances

The parameters needed for the `interesting` detector are:
+ percentile of variance to find variance threshold
+ min size of subsequence

Now we can also reduce the dimensionality further, by adding
an auto encoder. The one used here only has one hidden layer.

We then cluster all sequences using dynamic time warping window.
The window can be restricted by a `Sakoe-Chiba` band [2]. Furthermore,
we can weigh the errors `INSERTION`, `DELETION` and `MATCH` with
separate weights [3]. We also stop clustering using a threshold
estimated by a percentage. 

We cluster using agglomerative clustering with average linkage
also known as UPGMA[4]. For each alignment we compute we also save the
alignment path. This information is then used to build hidden markov models [7].
In each cluster, for each sequence we construct a simple hidden markov model. 
It is a chain of states, one state for each frame connected only to the next.
We then get all of the pairwise alignments in the cluster. We can now reference
each state with a frame number of a sequence. Using the alignment paths, we
have an association between frames or states in one sequence and frames or states
in the other. We merge states if their frames in the sequences match and if
the distance between the frames is small [5]. We also merge frames
in a sequence if the distance between two consequitive states is small.
In the end we delete all states from the model that have no self transition
since they represent only one frame.

During clustering and model merging, we keep track of the merges
using a structure called Union-Find or Connected Components [6].

After this we generate an audio file for each cluster which contains all instances
of the cluster. A latex document with the dendograms of the clustering, the hidden markov
models and a classification experiment showing that the models for each cluster model
the data. The output of the tool is summarised in a result html page.

## Usage

In order to generate the report and all the clusters run:

```
./generate_report.sh APPLY_ONLY FOLDER
```
APPLY_ONLY is true or false and determins if we load existing HMMs and decode the data or if we start a new clustering round. The folder should contain wav files, it will be searched recursively.
In order to configure the program use the file in `project/config`.
In order to change the latex templates use the `project/templates` 
folder.

## Source Code and Project
+ `aligned_model_merging.rs` Model Merging From Alignments and State Deletion
+ `audio.rs` Read and Write Audio                     
+ `discovery.rs` Discovery Parameters
+ `main.rs` Tying it all together              
+ `reporting.rs` Latex/HTML/GraphViz templating
+ `alignments.rs` DTW code with back tracking and alignment path information 
+ `clustering.rs` Hierarchical Clustering                  
+ `hidden_markov_model.rs` Hidden Markov Model 
+ `numerics.rs` All numerics methods
+ `spectrogram.rs` Implements spectrogram and slicing

## Requirements
+ Latex
+ GraphViz
+ Rust and Cargo

## Reference
+ [1 Cepstrum Wikipedia](https://de.wikipedia.org/wiki/Mel_Frequency_Cepstral_Coefficients)
+ [2 Sakoe Chiba](https://ieeexplore.ieee.org/document/1163055)
+ [3 DTW and Weights](https://www.amazon.com/Speech-Synthesis-Recognition-Wendy-Holmes/dp/0748408576)
+ [4 UPGMA](https://en.wikipedia.org/wiki/UPGMA)
+ [5 Model Merging](https://papers.nips.cc/paper/669-hidden-markov-model-induction-by-bayesian-model-merging.pdf)
+ [6 Algorithms Sedgewick](https://www.amazon.com/Algorithms-Robert-Sedgewick/dp/032157351X/ref=asc_df_032157351X/?tag=googshopde-21&linkCode=df0&hvadid=310624385211&hvpos=1o1&hvnetw=g&hvrand=16551751797611632310&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9044109&hvtargid=pla-432981821009&psc=1&th=1&psc=1&tag=&ref=&adgrpid=64736366074&hvpone=&hvptwo=&hvadid=310624385211&hvpos=1o1&hvnetw=g&hvrand=16551751797611632310&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9044109&hvtargid=pla-432981821009)
+ [7 HMM](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
