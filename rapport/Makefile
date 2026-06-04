# Makefile — Rapport de stage : Détection de tumeurs cérébrales (MC Dropout)
#
#   make            compile le rapport (deux passes pdflatex)
#   make clean      supprime les fichiers auxiliaires (garde le PDF)
#   make cleanall   supprime aussi le PDF

MAIN = rapport_stage
LATEX = pdflatex -interaction=nonstopmode

all: $(MAIN).pdf

# Le « - » ignore le code de sortie non nul renvoyé par certaines installations
# TeX Live malgré une compilation réussie ; les deux passes sont indispensables
# pour la table des matières, la liste des figures et les renvois croisés.
$(MAIN).pdf: $(MAIN).tex
	-$(LATEX) $(MAIN).tex
	-$(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot

cleanall: clean
	rm -f $(MAIN).pdf

.PHONY: all clean cleanall
