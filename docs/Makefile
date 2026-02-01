.PHONY: all clean

all: thesis.pdf

thesis.pdf: thesis.tex sections/*.tex references.bib
	pdflatex thesis
	bibtex thesis
	pdflatex thesis
	pdflatex thesis

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.synctex.gz
	rm -f sections/*.aux