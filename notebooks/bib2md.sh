if [ -f 5_references.md ]; then
    rm 5_references.md
fi

echo "# References" >> 5_references.md

bibtex2html -o - -s abbrv -q -nodoc references.bib >> 5_references.md

ex -snc '$-4,$d|x' 5_references.md