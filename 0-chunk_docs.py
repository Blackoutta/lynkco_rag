from pdf import parse_pdf
import os


def main():
    """
    以各维度切割文档
    输入: ./data/lynkco.pdf
    输出: ./data/parsed/all_text.txt
    """
    pdf_path = "./data/lynkco.pdf"
    out_path = "./data/parsed"
    if os.path.exists(os.path.join(out_path, "all_text.txt")):
        print("chunked data already exists")
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    parse_pdf(pdf_path=pdf_path, out_path=out_path, out_file_name="all_text.json")


if __name__ == "__main__":
    main()
