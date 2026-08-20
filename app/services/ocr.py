from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re


def configure_tesseract_command(pytesseract) -> str | None:
    """为 pytesseract 自动配置 tesseract.exe 路径。

    pytesseract 只是 Python 调用层，真正干 OCR 的是 Windows 里安装的
    tesseract.exe。

    如果 tesseract.exe 没有加入系统 Path，直接调用会报：
        TesseractNotFoundError

    所以这里做三层查找：
    1. 优先读环境变量 TESSERACT_CMD；
    2. 再找默认安装路径；
    3. 最后兼容你当前机器上出现的 tessdata/tesseract.exe 路径。

    返回值：
        找到的 tesseract.exe 路径；找不到则返回 None。
    """

    candidates = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tessdata\tesseract.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return None


@dataclass
class OcrResult:
    """OCR 解析结果。

    text:
        从图片或 PDF 中识别出来的文本。

    engine:
        实际使用的解析方式。
        例如 pytesseract-image、text-fallback、unsupported。

    filename:
        用户上传的原始文件名。

    content_type:
        HTTP 上传时携带的文件类型。

    warning:
        当 OCR 引擎未安装或文件格式暂不支持时，
        用这个字段返回清楚的提示。
    """

    text: str
    engine: str
    filename: str
    content_type: str | None = None
    warning: str | None = None


class TranscriptOcrService:
    """成绩单 OCR 解析服务。

    这个类的目标不是一次性做成最强 OCR，
    而是先把“文件上传 -> 后端解析 -> 前端展示 -> 可继续问答”的工程链路跑通。

    当前支持：
        1. 图片文件：
           如果安装了 pytesseract，就调用本地 OCR。
           如果没有安装，就返回安装提示。

        2. 文本文件：
           用 UTF-8 / GBK 尝试读取，方便你用 txt 模拟 OCR 结果。

        3. PDF 文件：
           当前先返回提示。
           后面可以接 MinerU、PyMuPDF、pdfplumber 或 PaddleOCR。

    为什么不直接强依赖 pytesseract？
        因为 pytesseract 除了 Python 包，还需要系统安装 Tesseract-OCR 程序。
        对学习项目来说，如果强制依赖它，很多机器会直接启动失败。
    """

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    text_suffixes = {".txt", ".md", ".csv"}

    def _prepare_image_variants(self, image):
        """
        为 Tesseract 准备多种图片版本。

        成绩单图片通常字小、表格密、有水印。
        直接识别原图时，Tesseract 很容易把中文识别成英文碎片。
        所以这里先做放大、灰度、增强、锐化和二值化，再交给 OCR。
        """

        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size

        target_width = 1800
        scale = min(4.0, max(2.0, target_width / max(width, 1)))
        resized = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )

        gray = ImageOps.grayscale(resized)
        gray = ImageOps.autocontrast(gray)
        contrast = ImageEnhance.Contrast(gray).enhance(1.8)
        sharp = contrast.filter(ImageFilter.SHARPEN)
        sharp = ImageOps.expand(sharp, border=20, fill=255)
        threshold = sharp.point(lambda pixel: 255 if pixel > 175 else 0)

        return [
            ("gray-enhanced", sharp),
            ("binary-threshold", threshold),
        ]

    def _score_ocr_text(self, text: str) -> float:
        """
        给 OCR 结果打一个粗略质量分。

        没有人工标注答案时，我们用启发式判断：
        - 中文字符越多越好；
        - 命中“成绩单、课程、学分、成绩”等关键词越好；
        - 连续英文乱码碎片越多越扣分。
        """

        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        keyword_score = 0

        for keyword in ["成绩单", "课程名称", "课程类别", "学分", "成绩", "学期", "学分总计"]:
            if keyword in text:
                keyword_score += 80

        ascii_noise = len(re.findall(r"\b[a-zA-Z]{5,}\b", text))

        return chinese_count * 2 + keyword_score + len(text) * 0.05 - ascii_noise * 3

    def _image_to_string_with_best_effort(self, pytesseract, image) -> tuple[str, str]:
        """
        使用多种预处理图片和 Tesseract 参数进行识别，并选择质量最高的结果。

        psm 参数可以理解成“页面布局假设”：
        - psm 6：假设是一整块文本；
        - psm 11：稀疏文本，适合被表格切碎的文本；
        - psm 4：多列文本，适合成绩单这类左右分栏页面。
        """

        variants = self._prepare_image_variants(image)
        configs = [
            "--oem 3 --psm 6 -c preserve_interword_spaces=1",
            "--oem 3 --psm 11 -c preserve_interword_spaces=1",
            "--oem 3 --psm 4 -c preserve_interword_spaces=1",
        ]

        best_text = ""
        best_label = "none"
        best_score = float("-inf")

        for variant_label, variant_image in variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        variant_image,
                        lang="chi_sim+eng",
                        config=config,
                    ).strip()
                except Exception:
                    continue

                score = self._score_ocr_text(text)
                if score > best_score:
                    best_text = text
                    best_label = f"{variant_label}; {config}"
                    best_score = score

        return best_text, best_label

    def parse_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> OcrResult:
        """解析上传文件字节。

        FastAPI 上传文件时拿到的是 UploadFile。
        我们在路由层读取 bytes，
        再交给这个方法按文件后缀选择解析策略。
        """

        suffix = Path(filename).suffix.lower()

        if suffix in self.image_suffixes:
            return self._parse_image(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )

        if suffix in self.text_suffixes:
            return self._parse_text(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )

        if suffix == ".pdf":
            return OcrResult(
                text="",
                engine="pdf-not-configured",
                filename=filename,
                content_type=content_type,
                warning=(
                    "当前 OCR 上传接口已接通，但 PDF 解析暂未启用。"
                    "PDF 建议继续使用 MinerU 解析后再入库；"
                    "如果要在接口里直接解析 PDF，可以后续接 PyMuPDF、pdfplumber 或 MinerU 命令。"
                ),
            )

        return OcrResult(
            text="",
            engine="unsupported",
            filename=filename,
            content_type=content_type,
            warning=f"暂不支持 {suffix or '未知'} 格式，请上传图片、txt 或 PDF。",
        )

    def _parse_image(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> OcrResult:
        """解析图片文件。

        图片 OCR 需要两个东西：
        1. pillow：
           用来把上传的 bytes 读取成图片对象。
        2. pytesseract + Tesseract-OCR：
           用来真正识别图片里的文字。
        """

        try:
            from PIL import Image
        except ImportError:
            return OcrResult(
                text="",
                engine="pillow-missing",
                filename=filename,
                content_type=content_type,
                warning="当前环境未安装 pillow，无法读取图片。",
            )

        try:
            image = Image.open(BytesIO(file_bytes))
        except Exception as error:
            return OcrResult(
                text="",
                engine="image-open-failed",
                filename=filename,
                content_type=content_type,
                warning=f"图片打开失败：{type(error).__name__}: {error}",
            )

        try:
            import pytesseract
        except ImportError:
            return OcrResult(
                text="",
                engine="pytesseract-missing",
                filename=filename,
                content_type=content_type,
                warning=(
                    "图片已读取成功，但当前环境未安装 pytesseract。"
                    "如果要真实识别图片文字，需要安装 pytesseract 和系统级 Tesseract-OCR。"
                ),
            )

        tesseract_cmd = configure_tesseract_command(pytesseract)
        if tesseract_cmd is None:
            return OcrResult(
                text="",
                engine="tesseract-command-missing",
                filename=filename,
                content_type=content_type,
                warning=(
                    "图片已读取成功，也安装了 pytesseract，"
                    "但没有找到 tesseract.exe。"
                    "请设置环境变量 TESSERACT_CMD 指向 tesseract.exe。"
                ),
            )

        try:
            text, best_label = self._image_to_string_with_best_effort(
                pytesseract=pytesseract,
                image=image,
            )
        except Exception as error:
            return OcrResult(
                text="",
                engine="pytesseract-error",
                filename=filename,
                content_type=content_type,
                warning=f"pytesseract 调用失败：{type(error).__name__}: {error}",
            )

        quality_warning = None
        width, _height = image.size

        if width < 1000:
            quality_warning = (
                "当前图片分辨率较低，成绩单表格文字较小，OCR 可能出现错字或列错位。"
                "建议优先上传原始 PDF，或使用宽度 1500px 以上的高清截图。"
            )
        elif text and "成绩单" in text and "课程" not in text:
            quality_warning = (
                "已识别到成绩单标题，但课程表格结构不够稳定。"
                "如果要做正式成绩分析，建议上传更清晰的原始文件。"
            )

        return OcrResult(
            text=text,
            engine=f"pytesseract-image-enhanced ({best_label})",
            filename=filename,
            content_type=content_type,
            warning=quality_warning if text else "OCR 未识别到明显文本。",
        )

    def _parse_text(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> OcrResult:
        """解析文本文件。

        这个入口主要用于学习和调试。
        你可以先把一段成绩单文本保存成 txt 上传，
        验证“上传 -> 解析 -> 结果展示”链路。
        """

        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                text = file_bytes.decode(encoding).strip()
                return OcrResult(
                    text=text,
                    engine=f"text-{encoding}",
                    filename=filename,
                    content_type=content_type,
                    warning=None if text else "文本文件为空。",
                )
            except UnicodeDecodeError:
                continue

        return OcrResult(
            text="",
            engine="text-decode-failed",
            filename=filename,
            content_type=content_type,
            warning="文本文件编码无法识别，请保存为 UTF-8 后重试。",
        )
