all: thesis.tex
	pdflatex thesis
	bibtex thesis
	pdflatex thesis
	pdflatex thesis
	pdflatex thesis

compress: thesis.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dNOPAUSE -dQUIET -dBATCH -sOutputFile=tmp.pdf thesis.pdf
	mv tmp.pdf thesis.pdf
