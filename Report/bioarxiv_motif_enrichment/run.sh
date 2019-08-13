#!/usr/bin/env bash

pdflatex document.tex
pdflatex document.tex
bibtex document
bibtex document
pdflatex document.tex
pdflatex document.tex

rm document.aux
rm document.bbl
rm document.blg
rm document.cb
rm document.cb2
rm document.log

