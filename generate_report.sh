#!/bin/bash

# make project
rm -rf output
mkdir output
mkdir output/img
mkdir output/docs
mkdir output/audio

# Run tool
cargo run > output/log.txt
# Generate reports
cd output
dot -Tpng -o img/markov.png docs/markov.dot 
pdflatex -output-directory docs/ docs/results.tex 
cd ..