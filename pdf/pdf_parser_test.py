from pdf_parser import *
import os

parsed_path = "./data/parsed"
if not os.path.exists(parsed_path):
    os.makedirs(parsed_path)
pdf_path = "./data/lynkco.pdf"


def test_parse_block():
    # 分块解析: 尽量保证一个小标题+对应文档在一个文档块，其中文档块的长度分别是512和1024。
    target_size = [512, 1024]

    for size in target_size:
        parser = PDFParser(pdf_path)
        parser.ParseBlock(max_seq=size)
        out = open(os.path.join(parsed_path, f"block_{size}.txt"), "w")
        for line in parser.data:
            line = line.strip("\n")
            out.write(line)
            out.write("\n")
        out.close()


def test_parse_all_pages():
    # 滑动窗口解析： 把文档句号分割，然后构建滑动窗口，其中文档块的长度分别是256和512。
    target_size = [256, 512]

    for size in target_size:
        parser = PDFParser(pdf_path)
        parser.ParseAllPages(max_seq=size)
        out = open(os.path.join(parsed_path, f"all_pages_{size}.txt"), "w")
        for line in parser.data:
            line = line.strip("\n")
            out.write(line)
            out.write("\n")
        out.close()


def test_parse_one_page_with_rule():
    # 页级提取，如果页内容过多，降级为句子级提取，其中文档块的长度分别是256和512。
    target_size = [256, 512]

    for size in target_size:
        parser = PDFParser(pdf_path)
        parser.ParseOnePageWithRule(max_seq=size)
        out = open(os.path.join(parsed_path, f"one_page_{size}.txt"), "w")
        for line in parser.data:
            line = line.strip("\n")
            out.write(line)
            out.write("\n")
        out.close()


def test_sliding_window():
    chunks = SlidingWindow(["aaa", "bbb", "ccc", "ddd"], kernel=3)
    assert chunks == ["aaa。bbb。", "bbb。ccc。", "ccc。ddd。"]

    chunks = SlidingWindow(["aaa", "bbb", "ccc", "ddd"], kernel=9)
    assert chunks == ["aaa。bbb。ccc。", "bbb。ccc。ddd。", "ccc。ddd。"]


if __name__ == "__main__":
    test_parse_block()
    test_parse_all_pages()
    test_parse_one_page_with_rule()
    test_sliding_window()
