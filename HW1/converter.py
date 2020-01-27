import pdfkit


#pdfkit.from_file('test.html', 'test.pdf')

pdfl = PDFLaTeX.from_texfile('hw1.tex')

pdf = pdfl.create_pdf(keep_pdf_file=True)