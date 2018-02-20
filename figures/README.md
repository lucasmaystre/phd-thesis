How to compile the GraphViz graphs:

    neato -n -Tpdf graph-example.dot > graph-example.pdf
    pdfcrop graph-example.pdf

If the figure needs to be reworked in Illustrator, save as EPS instead of PDF.
In order to get the smallest file size when saving from Illustrator,

1. save as EPS (leave default settings)
2. use the `epspdf` tool
3. compress with

        gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
            -dNOPAUSE -dQUIET -dBATCH -sOutputFile=out.pdf in.pdf
