# WACV 2024 Author Kit (Latex)

### Other Latex Build Tools
If you prefer to use an alternative editor/build engine to overleaf, you can download this author kit as a [zip file](https://drive.google.com/file/d/1YaMHj7d_vmmBrnro-8kRv8CctpEizNIH/view?usp=sharing).

There are a variety of latex editing/build tools you can find on the web.

### Command-line (e.g. Linux)
If you plan to generate your paper via the command-line, you may need to install latex packages/fonts (e.g. `texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended`).

To generate the document from the latex file, it is recommended that you use `pdflatex`.  For example, also using `bibtex` to generate references:

```$ pdflatex PaperForReview.tex; bibtex PaperForReview; pdflatex PaperForReview.tex```

### Microsoft Word Template (None Provided)
We are no longer providing a Microsoft Word template, and are actively discouraging the use of Word for preparing papers for WACV.  Latex is used for the vast majority of conferences and journals in our field and Overleaf makes the use of Latex very straightforward, even for first-time users.  Overleaf provides an excellent tutorial titled [Learn Latex in 30 minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes).


## Notes on Review vs. Camera-ready Formats:
The typical `egpaper_for_review.tex` and `egpaper_final.tex` files have been merged into a single file, `PaperForReview.tex`.  The file is initially set up for review submission -- all you need to add is your CMT paper ID and select your track (Applications or Algorithms).

There are important instructions at the top of the combined `PaperForReview.tex` document describing (i) how to toggle between the review and final (camera-ready) formats, (ii) how to set your Paper ID (which is assigned upon creation of your paper submission in CMT) for the review version, and (iii) how to select your track (Applications or Algorithms) in the review copy.
