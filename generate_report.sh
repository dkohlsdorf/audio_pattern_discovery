#!/bin/bash

# Run tool
cargo run
# Generate reports
cd output
dot -Tpng -o markov.png markov.dot 
pdflatex results.tex 
cd ..