from .pdf_parser import PDFParser
import os


def parse_pdf(
    pdf_path: str = "./data/lynkco.pdf",
    out_path: str = "./data/parsed",
    out_file_name: str = "all_text.txt",
):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    parser = PDFParser(pdf_path)

    block_sizes = [512, 1024]
    for size in block_sizes:
        parser.ParseBlock(size)

    all_page_sizes = [256, 512]
    for size in all_page_sizes:
        parser.ParseAllPages(size)

    one_page_sizes = [256, 512]
    for size in one_page_sizes:
        parser.ParseOnePageWithRule(size)

    out = open(os.path.join(out_path, out_file_name, "w"))
    for line in parser.data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()


if __name__ == "__main__":
    parse_pdf()
