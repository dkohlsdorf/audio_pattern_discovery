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
also known as UPGMA[4]. 

After this we generate an audio file for each cluster which contains all instances
of the cluster. A latex document with the dendograms of the clusterin and
a classification experiment showing that the models for each cluster model
the data. The output of the tool is summarised in a result html page.

## Usage

In order to generate the report and all the clusters run:

```
./generate_report.sh FOLDER
```
The folder should contain wav files, it will be searched recursively.
In order to configure the program use the file in `project/config`.
In order to change the latex templates use the `project/templates` 
folder.

## Source Code and Project
+ `audio.rs` Read and Write Audio                     
+ `discovery.rs` Discovery Parameters
+ `main.rs` Tying it all together              
+ `reporting.rs` Latex/HTML/GraphViz templating
+ `alignments.rs` DTW code with back tracking and alignment path information 
+ `clustering.rs` Hierarchical Clustering                  
+ `numerics.rs` All numerics methods
+ `spectrogram.rs` Implements spectrogram and slicing
+ `neural.rs` Implements a one layer autoencoder

## Requirements
+ Latex
+ Rust and Cargo

## Reference
+ [1 Cepstrum Wikipedia](https://de.wikipedia.org/wiki/Mel_Frequency_Cepstral_Coefficients)
+ [2 Sakoe Chiba](https://ieeexplore.ieee.org/document/1163055)
+ [3 DTW and Weights](https://www.amazon.com/Speech-Synthesis-Recognition-Wendy-Holmes/dp/0748408576)
+ [4 UPGMA](https://en.wikipedia.org/wiki/UPGMA)
