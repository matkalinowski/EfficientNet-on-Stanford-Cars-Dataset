pandoc -H cover_options.sty -V geometry:margin=1.05in -V fontsize=12pt -V fontfamily:fourier cover_and_summary.md -o cover_and_summary.pdf

cd ..
. bib2md.sh
cat documentation.md 5_references.md >> pdf_conversion/merged.md
sed -e s/5_references.md//g -i pdf_conversion/merged.md
pandoc --toc --pdf-engine=xelatex -H pdf_conversion/main_options.sty -V geometry:margin=1.2in -V fontsize=12pt --top-level-division=chapter pdf_conversion/merged.md -o pdf_conversion/contents.pdf
rm pdf_conversion/merged.md

cd pdf_conversion
pdftk cover_and_summary.pdf contents.pdf cat output thesis.pdf
rm cover_and_summary.pdf contents.pdf
