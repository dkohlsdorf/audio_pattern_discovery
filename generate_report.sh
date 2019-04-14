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
for i in docs/*.dot
do
    if test -f "$i" 
    then
        echo "Creating Markov Chain: $i"
        filename=$(basename -- "$i")
        extension="${filename##*.}"
        filename="${filename%.*}"
        dot -Tpng -o img/$filename.png docs/$filename.dot 
    fi
done
pdflatex -output-directory docs/ docs/results.tex 
cd ..