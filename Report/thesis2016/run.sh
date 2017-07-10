#!/usr/bin/env bash

pdflatex thesis.tex
pdflatex thesis.tex
bibtex thesis
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
makeindex thesis.nlo -s nomencl.ist -o thesis.nls
makeindex thesis.nlo -s nomencl.ist -o thesis.nls
pdflatex thesis.tex
pdflatex thesis.tex

rm thesis.aux
rm thesis.bbl
rm thesis.blg
rm thesis.ilg
rm thesis.lof
rm thesis.log
rm thesis.lot
rm thesis.nlo
rm thesis.nls
rm thesis.out
rm thesis.toc
rm preamble/cover.aux
rm preamble/glossary.aux

