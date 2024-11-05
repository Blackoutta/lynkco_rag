# coding: utf-8

import pdfplumber
from PyPDF2 import PdfReader
from pdfplumber.page import Page
from typing import List, Tuple, Dict, Any


class PDFParser(object):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    def ParseBlock(self, max_seq=1024):
        """
        按段落块提取，核心思路是利用字体大小的变换来判断文档块的变化
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            pages = pdf.pages
            if len(pages) < 1:
                raise Exception("PDF文件没有包含任何页")

            for i, page in enumerate(pages[:]):
                header = GetPageHeader(page)
                if header is None:
                    continue

                lines = page.extract_words(use_text_flow=True, extra_attrs=["size"])[::]
                sequence = ""
                lastsize = 0

                for idx, line in enumerate(lines):
                    l_text = line["text"]
                    l_size = line["size"]
                    if idx < 1:
                        continue
                    # 页面有时候会出现在第二个位置，此时忽略它
                    if idx == 1 and l_text.isdigit():
                        continue

                    # 这些特殊字符单独出现时，忽略它们
                    if l_text == "□" or l_text == "•":
                        continue
                    # 遇到下面的关键词时，说明这一段sequence可以结束了
                    elif l_text == "警告！" or l_text == "注意！" or l_text == "说明！":
                        if len(sequence) > 0:
                            self._consume_unique_chunks(
                                DataFilter(sequence, max_seq=max_seq)
                            )
                        sequence = ""
                    # 如果当前行的字体大小和上一次的一样，说明这一段sequence还是在讲同样的事
                    elif format(lastsize, ".5f") == format(l_size, ".5f"):
                        if len(sequence) > 0:
                            sequence = sequence + l_text
                        else:
                            sequence = l_text

                    # 如果当前行的字体大小和上一次的不一样，说明:
                    # 1. 现在是正文转标题，上一段sequence可以结束
                    # 2. 现在是标题转正文，上一段sequence继续
                    else:
                        lastsize = l_size

                        # 1-15个字符说明现在sequence里只有标题，这里是标题刚转到正文，正文继续
                        if len(sequence) > 0 and len(sequence) < 15:
                            sequence = sequence + l_text
                        # 走到这里，就是要正文转标题
                        else:
                            # 结束掉当前的正文
                            if len(sequence) > 0:
                                self._consume_unique_chunks(
                                    DataFilter(sequence, max_seq=max_seq)
                                )
                            # 将新标题加入放入sequence
                            if l_size > 9:
                                sequence = "#" + l_text + ":"
                            # 走到这里证明是页面的开头，但是开头不是标题
                            else:
                                sequence = l_text

                if len(sequence) > 0:
                    self._consume_unique_chunks(DataFilter(sequence, max_seq=max_seq))

    def ParseAllPages(self, max_seq=512, min_len=6):
        """
        滑窗法提取段落
        1. 把pdf看做一个整体,作为一个字符串
        2. 利用句号当做分隔符,切分成一个数组
        3. 利用滑窗法对数组进行滑动
        """
        all_content = ""
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if "...................." in text or "目录" in text:
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                page_content = page_content + text
            if len(page_content) < min_len:
                continue
            all_content = all_content + page_content
        sentences = all_content.split("。")
        self._consume_unique_chunks(SlidingWindow(sentences, kernel=max_seq))

    def ParseOnePageWithRule(self, max_seq=512, min_len=6):
        """
        按每页内容提取
        如果一页的内容 < max_seq，则直接提取
        如果一页的内容 > max_seq，则按句号划分成句子，再将句子组合成文档块
        如果文档块 > max_seq，就提取
        """

        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if "...................." in text or "目录" in text:
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                page_content = page_content + text
            if len(page_content) < min_len:
                continue
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if (
                        len(cur + sentence) > max_seq
                        and (cur + sentence) not in self.data
                    ):
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence

    def _consume_unique_chunks(self, blocks: List[str]):
        if blocks is None or len(blocks) < 1:
            return
        for block in blocks:
            if block not in self.data:
                self.data.append(block)


def GetPageHeader(page: Page):
    lines = page.extract_words()
    if len(lines) < 1:
        return None
    for line in lines:
        # line be like:
        # {'text': '首次使用前请仔细、完整地阅读本手册内容，将有助于您更好地了解和使用车辆。', 'x0': 167.244, 'x1': 454.1676, 'top': 171.51313359999995, 'doctop': 171.51313359999995, 'bottom': 179.48323359999995, 'upright': True, 'height': 7.970100000000002, 'width': 286.92359999999996, 'direction': 'ltr'}
        l_text = line["text"]
        w_top = line["top"]  # 单词顶部边界的垂直坐标

        # 如果这一页是目录相关，那么跳过
        if "目录" in l_text or ".........." in l_text:
            return None
        if w_top > 17 and w_top < 20:
            return l_text
    first_line = lines[0]["text"]
    if first_line == "123":
        return None
    return first_line


#  数据过滤，根据当前的文档内容的item划分句子，然后根据max_seq划分文档块。
def DataFilter(line: str, max_seq=1024) -> List[str]:
    sz = len(line)
    if sz < 6:
        return []

    if sz <= max_seq:
        return [line.replace("\n", "").replace(",", "").replace("\t", "")]

    if "■" in line:
        sentences = line.split("■")
    elif "•" in line:
        sentences = line.split("•")
    elif "\t" in line:
        sentences = line.split("\t")
    else:
        sentences = line.split("。")

    results = []
    for s in sentences:
        s = s.replace("\n", "")
        if len(s) < max_seq and len(s) > 5:
            results.append(s.replace(",", "").replace("\n", "").replace("\t", ""))
    return results


def SlidingWindow(sentences, kernel=512, stride=1):
    window = ""
    fast = 0
    slow = 0
    chunks = []
    while fast < len(sentences):
        cur_sentence = sentences[fast]
        slide = window + cur_sentence
        if len(slide) > kernel:
            chunks.append(slide + "。")
            window = window[len(sentences[slow] + "。") :]
            slow = slow + 1

        window = window + cur_sentence + "。"
        fast = fast + 1

    # 滑动到最后如果有剩余，也要加上
    if kernel % len(chunks) != 0:
        chunks.append(window)

    return chunks
