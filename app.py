import os

import pytesseract
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


load_dotenv()


# 告诉 pytesseract：你的 Tesseract OCR 程序安装在哪里。
# 你提供的路径是 D:\OCR\tesseract.exe。
pytesseract.pytesseract.tesseract_cmd = r"D:\OCR\tesseract.exe"


# DeepSeek 使用 OpenAI SDK 的兼容写法，但需要换成 DeepSeek 的 base_url。
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


CATEGORIES = [
    "学校通知",
    "租房/房东",
    "银行",
    "NHS/医疗",
    "考试",
    "账单/付款",
    "兼职/工作",
    "其他",
]


def extract_text_from_image(image: Image.Image) -> str:
    """使用 Tesseract OCR，把截图里的英文提取成文字。"""
    return pytesseract.image_to_string(image, lang="eng")


def analyze_notice(ocr_text: str, category: str) -> str:
    """把用户修正后的英文通知发给 DeepSeek，并返回中文分析结果。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("请先在 .env 文件中设置 DEEPSEEK_API_KEY。")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    system_prompt = """
你是 NoticePilot，一个帮助中国留学生理解英文邮件和通知的助手。
你的回答必须使用简体中文，清楚、具体、适合初学者理解。
不要编造原文没有的信息；如果没有提到，请写“未提及”。
如果涉及医疗、法律、银行或租房风险，请提醒用户联系官方机构或专业人士确认。
"""

    user_prompt = f"""
请分析下面这份英文通知。

通知类别：{category}

请严格按照以下结构输出：

## 一句话结论

## 用户需要做什么

## 是否需要回复

## 关键日期、金额、地点、联系人

## 风险等级

## 英文回复模板

## 注意事项

英文通知原文：
{ocr_text}
"""

    # 这里使用 DeepSeek 官网示例里的 chat.completions.create 调用方式。
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    return response.choices[0].message.content


st.set_page_config(page_title="NoticePilot", layout="centered")

st.title("NoticePilot")
st.caption("上传英文邮件或通知截图，OCR 识别后可手动修改，再生成中文解读。")

uploaded_file = st.file_uploader(
    "上传截图",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="已上传的截图", use_container_width=True)

    # session_state 可以保存页面刷新后的临时内容。
    # 这样用户点击按钮后，OCR 结果不会马上消失。
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = ""

    # 如果用户上传了另一张图，就清空上一张图的 OCR 结果。
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.ocr_text = ""

    if st.button("提取文字"):
        with st.spinner("正在识别截图中的英文文字..."):
            try:
                st.session_state.ocr_text = extract_text_from_image(image)
            except pytesseract.TesseractNotFoundError:
                st.error("没有找到 Tesseract。请确认 D:\\OCR\\tesseract.exe 是否存在。")
            except Exception as error:
                st.error(f"OCR 识别失败：{error}")

    corrected_text = st.text_area(
        "OCR 结果（可以在这里修改）",
        value=st.session_state.ocr_text,
        height=260,
        placeholder="点击“提取文字”后，识别结果会出现在这里。你也可以直接粘贴英文通知内容。",
    )
    st.session_state.ocr_text = corrected_text

    category = st.selectbox("通知类别", CATEGORIES)

    if st.button("分析通知"):
        if not corrected_text.strip():
            st.warning("请先提取或输入英文通知内容。")
        else:
            with st.spinner("正在调用 DeepSeek 生成中文解读..."):
                try:
                    result = analyze_notice(corrected_text, category)
                    st.markdown(result)
                except Exception as error:
                    st.error(f"分析失败：{error}")
else:
    st.info("请先上传一张英文邮件或通知截图。")
