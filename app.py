import hashlib
import json
import os
from io import BytesIO, StringIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)


# ==================================================
# 기본 설정
# ==================================================

st.set_page_config(
    page_title="개인 지출 분석 대시보드",
    page_icon="💰",
    layout="wide",
)


# ==================================================
# 카테고리 자동 분류
# ==================================================

CATEGORY_KEYWORDS = {
    "카페·간식": [
        "스타벅스", "이디야", "투썸", "메가커피",
        "컴포즈", "카페", "커피", "베이커리",
    ],
    "식비": [
        "식당", "김밥", "치킨", "피자", "버거",
        "배달", "한식", "분식", "점심", "저녁",
    ],
    "장보기·편의점": [
        "마트", "슈퍼", "편의점", "이마트",
        "홈플러스", "코스트코", "gs25", "cu",
        "세븐일레븐",
    ],
    "교통": [
        "택시", "버스", "지하철", "철도",
        "주유", "주차", "하이패스", "교통", "ktx",
    ],
    "쇼핑": [
        "쿠팡", "쇼핑", "백화점", "다이소",
        "무신사", "올리브영", "의류", "신발",
    ],
    "주거·통신": [
        "월세", "관리비", "전기", "수도",
        "가스", "통신", "휴대폰", "인터넷",
    ],
    "건강·의료": [
        "병원", "약국", "의원", "치과",
        "검진", "헬스", "필라테스",
    ],
    "문화·구독": [
        "넷플릭스", "유튜브", "티빙", "디즈니",
        "영화", "도서", "서점", "공연", "구독",
    ],
    "교육": [
        "학원", "강의", "수강", "교육", "클래스",
    ],
}


def classify_category(description):
    text = str(description).casefold()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.casefold() in text for keyword in keywords):
            return category

    return "기타"


# ==================================================
# CSV 관련 함수
# ==================================================

def to_csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8-sig")


def make_sample():
    return pd.DataFrame({
        "날짜": [
            "2026-10-01", "2026-10-03", "2026-10-05",
            "2026-10-10", "2026-10-15", "2026-11-01",
            "2026-11-02", "2026-11-03", "2026-11-04",
            "2026-11-05",
        ],
        "내역": [
            "스타벅스", "점심 식당", "지하철",
            "이마트", "넷플릭스", "메가커피",
            "치킨 배달", "택시", "쿠팡", "약국",
        ],
        "금액": [
            5500, 12000, 1500, 45000, 17000,
            3000, 24000, 16000, 35000, 8000,
        ],
    })


def read_csv(file_bytes):
    text = None

    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(
            "파일을 UTF-8 CSV로 저장한 후 다시 업로드해주세요."
        )

    try:
        data = pd.read_csv(StringIO(text), dtype=str)
    except pd.errors.EmptyDataError:
        raise ValueError("비어 있는 CSV 파일입니다.")
    except pd.errors.ParserError:
        raise ValueError(
            "CSV 형식이 올바르지 않아요. 샘플 파일을 확인해주세요."
        )

    data.columns = data.columns.str.strip()

    if data.columns.duplicated().any():
        raise ValueError(
            "공백을 제거한 뒤 같은 이름이 되는 열이 있어요. "
            "CSV 열 이름을 서로 다르게 바꿔주세요."
        )

    if data.empty:
        raise ValueError("CSV에 거래 내역이 없습니다.")

    return data


def clean_data(raw, date_col, description_col, amount_col, date_format):
    data = pd.DataFrame(index=raw.index)

    date_text = raw[date_col].fillna("").astype(str).str.strip()

    data["날짜"] = pd.to_datetime(
        date_text,
        format=date_format,
        errors="coerce",
    )

    data["내역"] = (
        raw[description_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    amount_text = (
        raw[amount_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"[,₩원\s]", "", regex=True)
    )

    data["금액"] = pd.to_numeric(amount_text, errors="coerce")

    invalid_date = data["날짜"].isna()
    invalid_amount = (
        data["금액"].isna()
        | data["금액"].isin([float("inf"), float("-inf")])
    )

    invalid = invalid_date | invalid_amount

    rejected = raw.loc[invalid].copy()

    rejected["확인 사항"] = [
        " / ".join(
            message
            for condition, message in (
                (invalid_date.loc[i], "날짜 형식 확인"),
                (invalid_amount.loc[i], "금액 형식 확인"),
            )
            if condition
        )
        for i in rejected.index
    ]

    valid = data.loc[~invalid].copy()
    valid["날짜"] = valid["날짜"].dt.normalize()

    return valid, rejected


# ==================================================
# PDF 생성
# ==================================================

def create_pdf(summary_text, analysis_text, font_bytes=None):
    if font_bytes is not None:
        # 업로드한 글꼴마다 고유한 이름을 사용합니다.
        font_hash = hashlib.sha256(font_bytes).hexdigest()[:16]
        font_name = f"Korean_{font_hash}"

        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(
                TTFont(font_name, BytesIO(font_bytes))
            )
    else:
        font_path = (
            Path(os.environ.get("WINDIR", r"C:\Windows"))
            / "Fonts"
            / "malgun.ttf"
        )

        if not font_path.exists():
            raise ValueError(
                "한글 글꼴을 찾지 못했어요. "
                "PDF 설정에서 한글 TTF 글꼴을 업로드해주세요."
            )

        font_name = "KoreanMalgun"

        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(
                TTFont(font_name, str(font_path))
            )

    title_style = ParagraphStyle(
        "ExpenseTitle",
        fontName=font_name,
        fontSize=20,
        leading=30,
        textColor=colors.HexColor("#16324F"),
        spaceAfter=18,
        wordWrap="CJK",
    )

    heading_style = ParagraphStyle(
        "ExpenseHeading",
        fontName=font_name,
        fontSize=13,
        leading=22,
        spaceBefore=16,
        spaceAfter=10,
        wordWrap="CJK",
    )

    body_style = ParagraphStyle(
        "ExpenseBody",
        fontName=font_name,
        fontSize=10,
        leading=18,
        spaceAfter=7,
        wordWrap="CJK",
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45,
        title="개인 지출 분석 리포트",
    )

    story = [
        Paragraph("개인 지출 분석 리포트", title_style),
        Paragraph("1. 지출 집계", heading_style),
    ]

    for line in summary_text.splitlines():
        if line.strip():
            story.append(Paragraph(escape(line), body_style))
        else:
            story.append(Spacer(1, 5))

    story.append(
        Paragraph("2. 분석 및 개선 제안", heading_style)
    )

    for line in analysis_text.splitlines():
        if line.strip():
            story.append(Paragraph(escape(line), body_style))
        else:
            story.append(Spacer(1, 5))

    story.append(Spacer(1, 15))
    story.append(
        Paragraph(
            "선택한 조건에 해당하는 유효 지출만 집계했습니다. "
            "카테고리는 키워드로 추정한 결과입니다. "
            "AI 해석에는 오류가 있을 수 있습니다.",
            body_style,
        )
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 9)
        canvas.drawRightString(
            A4[0] - 45,
            25,
            f"{doc.page} 페이지",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return buffer.getvalue()


# ==================================================
# 화면 시작
# ==================================================

st.title("💰 개인 지출 분석 대시보드")
st.write(
    "CSV를 업로드하면 지출을 분석하고 "
    "AI 리포트와 한글 PDF를 만들 수 있어요."
)

sample = make_sample()

with st.expander("CSV 양식과 샘플 다운로드"):
    st.write(
        "날짜, 내역, 금액이 필요해요. "
        "열 이름이 달라도 업로드 후 직접 선택할 수 있어요."
    )

    st.dataframe(sample, hide_index=True)

    st.download_button(
        "샘플 CSV 다운로드",
        data=to_csv_bytes(sample),
        file_name="sample_expenses.csv",
        mime="text/csv",
    )


uploaded = st.file_uploader(
    "지출 CSV 업로드",
    type=["csv"],
)

if uploaded is None:
    use_sample = st.checkbox("샘플 데이터로 체험하기")

    if not use_sample:
        st.info("CSV를 올리거나 샘플 데이터 체험을 선택해주세요.")
        st.stop()

    raw = sample.astype(str)
    is_sample = True
else:
    try:
        raw = read_csv(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        st.stop()

    is_sample = False

if is_sample:
    st.info("지금은 실제 지출이 아닌 샘플 데이터를 보고 있어요.")


# ==================================================
# 열 선택과 형식 변환
# ==================================================

if len(raw.columns) < 3:
    st.error("날짜, 내역, 금액에 해당하는 열이 최소 3개 필요해요.")
    st.stop()


def guess_column(names, fallback):
    for name in names:
        if name in raw.columns:
            return list(raw.columns).index(name)

    return fallback


st.subheader("1. 데이터 형식 확인")

a, b, c = st.columns(3)

with a:
    date_col = st.selectbox(
        "날짜 열",
        raw.columns,
        index=guess_column(
            ["날짜", "거래일자", "이용일자", "거래일", "date"],
            0,
        ),
    )

with b:
    description_col = st.selectbox(
        "사용 내역 열",
        raw.columns,
        index=guess_column(
            ["내역", "사용처", "가맹점명", "내용", "description"],
            1,
        ),
    )

with c:
    amount_col = st.selectbox(
        "금액 열",
        raw.columns,
        index=guess_column(
            ["금액", "이용금액", "지출금액", "amount"],
            2,
        ),
    )

if len({date_col, description_col, amount_col}) != 3:
    st.warning("날짜, 내역, 금액은 서로 다른 열을 선택해주세요.")
    st.stop()


date_formats = {
    "2026-11-02": "%Y-%m-%d",
    "2026/11/02": "%Y/%m/%d",
    "2026.11.02": "%Y.%m.%d",
    "20261102": "%Y%m%d",
}

date_label = st.selectbox(
    "CSV의 날짜 모양",
    list(date_formats),
)

sign_mode = st.radio(
    "지출 금액 표시 방식",
    [
        "양수가 지출",
        "음수가 지출",
    ],
    horizontal=True,
)

st.caption(
    "선택한 부호의 지출만 분석합니다. "
    "0원과 반대 부호의 거래는 제외하며, 환불을 차감하지 않습니다."
)

data, rejected = clean_data(
    raw,
    date_col,
    description_col,
    amount_col,
    date_formats[date_label],
)

if not rejected.empty:
    st.warning(
        f"날짜 또는 금액을 읽을 수 없는 {len(rejected):,}건을 제외했어요."
    )

    with st.expander("제외된 행 확인"):
        st.dataframe(rejected, hide_index=True)

if data.empty:
    st.error("분석할 데이터가 없어요. 날짜와 금액 형식을 확인해주세요.")
    st.stop()

if sign_mode == "양수가 지출":
    expenses = data.loc[data["금액"] > 0].copy()
else:
    expenses = data.loc[data["금액"] < 0].copy()
    expenses["금액"] = expenses["금액"].abs()

excluded_count = len(data) - len(expenses)

if excluded_count:
    st.info(
        f"0원 또는 반대 부호의 거래 {excluded_count:,}건을 제외했어요."
    )

if expenses.empty:
    st.warning("지출이 없어요. 지출 금액 표시 방식을 확인해주세요.")
    st.stop()

expenses["카테고리"] = expenses["내역"].apply(classify_category)
expenses["월"] = expenses["날짜"].dt.strftime("%Y-%m")


# ==================================================
# 필터
# ==================================================

with st.sidebar:
    st.header("🔎 분석 조건")

    first_date = expenses["날짜"].min().date()
    last_date = expenses["날짜"].max().date()

    selected_dates = st.date_input(
        "분석 기간",
        value=(first_date, last_date),
        min_value=first_date,
        max_value=last_date,
    )

    category_options = sorted(
        expenses["카테고리"].unique().tolist()
    )

    selected_categories = st.multiselect(
        "카테고리",
        options=category_options,
        default=category_options,
    )

    keyword = st.text_input(
        "내역 검색",
        placeholder="예: 커피, 택시",
    )

    st.divider()
    st.header("🎯 월 예산")

    budget_month = st.selectbox(
        "예산 확인 월",
        sorted(expenses["월"].unique(), reverse=True),
    )

    budget = st.number_input(
        "월 예산 (원)",
        min_value=0,
        value=500000,
        step=10000,
    )

    st.caption(
        "예산 비교는 업로드된 해당 월의 전체 지출 기준입니다. "
        "분석 필터는 적용하지 않아요."
    )

if len(selected_dates) != 2:
    st.info("분석 시작일과 종료일을 선택해주세요.")
    st.stop()

start_date, end_date = selected_dates

filtered = expenses.loc[
    (expenses["날짜"].dt.date >= start_date)
    & (expenses["날짜"].dt.date <= end_date)
    & expenses["카테고리"].isin(selected_categories)
].copy()

if keyword.strip():
    filtered = filtered.loc[
        filtered["내역"].str.contains(
            keyword.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]


# ==================================================
# 월 예산
# ==================================================

st.subheader("2. 월 예산")

month_total = expenses.loc[
    expenses["월"] == budget_month, "금액"
].sum()

st.write(f"**{budget_month} 지출: {month_total:,.0f}원**")

if budget > 0:
    usage = float(month_total / budget)

    st.progress(min(usage, 1.0))
    st.write(f"예산 {budget:,.0f}원 중 {usage:.1%} 사용")

    if month_total > budget:
        st.error(
            f"예산보다 {month_total - budget:,.0f}원 초과했어요."
        )
    else:
        st.success(
            f"남은 예산은 {budget - month_total:,.0f}원이에요."
        )
else:
    st.info("예산을 0원보다 크게 입력하면 사용률을 보여드려요.")

st.caption("업로드한 자료가 해당 월 전체 내역인지 확인해주세요.")


# ==================================================
# 지출 요약
# ==================================================

st.subheader("3. 선택한 조건의 지출 요약")

if filtered.empty:
    st.info("조건에 맞는 지출이 없어요. 필터를 변경해주세요.")
    st.stop()

total = float(filtered["금액"].sum())
count = len(filtered)
period_days = (end_date - start_date).days + 1
average = float(filtered["금액"].mean())

a, b, c, d = st.columns(4)

a.metric("총 지출", f"{total:,.0f}원")
b.metric("거래 건수", f"{count:,}건")
c.metric("건당 평균", f"{average:,.0f}원")
d.metric("하루 평균", f"{total / period_days:,.0f}원")

st.caption(
    "하루 평균은 지출이 없는 날도 포함한 선택 기간 기준입니다. "
    "카테고리는 내역의 키워드로 추정합니다."
)


# ==================================================
# 차트
# ==================================================

month_index = pd.period_range(
    start=start_date,
    end=end_date,
    freq="M",
).astype(str)

monthly = (
    filtered.groupby("월")["금액"]
    .sum()
    .reindex(month_index, fill_value=0)
    .rename_axis("월")
    .to_frame("지출액")
)

category_totals = (
    filtered.groupby("카테고리")["금액"]
    .sum()
    .sort_values(ascending=False)
    .to_frame("지출액")
)

left, right = st.columns(2)

with left:
    st.subheader("월별 지출")
    st.bar_chart(monthly)

with right:
    st.subheader("카테고리별 지출")
    st.bar_chart(category_totals)

st.caption(
    "선택 기간이 월 중간부터 시작하거나 끝나면 "
    "그 월은 일부 기간만 집계됩니다."
)


# ==================================================
# 상세 내역 다운로드
# ==================================================

st.subheader("4. 상세 내역")

display = filtered.sort_values("날짜", ascending=False).copy()
display["날짜"] = display["날짜"].dt.strftime("%Y-%m-%d")
display = display[["날짜", "내역", "카테고리", "금액"]]

st.dataframe(display, hide_index=True)

st.download_button(
    "📥 분석 내역 CSV 다운로드",
    data=to_csv_bytes(display),
    file_name="expense_analysis.csv",
    mime="text/csv",
)


# ==================================================
# 리포트용 집계
# ==================================================

summary_lines = [
    f"자료: {'샘플 데이터' if is_sample else '업로드한 CSV'}",
    f"분석 기간: {start_date} ~ {end_date}",
    f"선택 카테고리: {', '.join(selected_categories)}",
    f"내역 검색 조건: {keyword.strip() or '없음'}",
    f"총 지출: {total:,.0f}원",
    f"거래 건수: {count:,}건",
    f"건당 평균: {average:,.0f}원",
    f"하루 평균: {total / period_days:,.0f}원",
    "",
    "월별 지출:",
]

for month, row in monthly.iterrows():
    summary_lines.append(
        f"{month}: {row['지출액']:,.0f}원"
    )

summary_lines.extend(["", "카테고리별 지출:"])

for category, row in category_totals.iterrows():
    summary_lines.append(
        f"{category}: {row['지출액']:,.0f}원 "
        f"({row['지출액'] / total:.1%})"
    )

summary_text = "\n".join(summary_lines)

top_category = category_totals.index[0]

basic_analysis = (
    f"가장 지출이 큰 카테고리는 {top_category}입니다.\n\n"
    "지출이 큰 카테고리의 상세 내역부터 검토해보세요.\n"
    "다음 달에는 카테고리별 예산을 정하고 실제 지출과 비교해보세요.\n\n"
    "이 자료만으로 소득이나 가계 전체의 재정 상태를 판단할 수 없습니다. "
    "월별 자료가 전체 월을 포함하지 않을 수 있으며, "
    "환불이나 수입은 반영하지 않았습니다."
)

payload = {
    "자료": "샘플 데이터" if is_sample else "업로드 자료",
    "통화": "KRW",
    "분석기간": [str(start_date), str(end_date)],
    "카테고리필터": selected_categories,
    "내역검색필터_적용여부": bool(keyword.strip()),
    "총지출": total,
    "거래건수": count,
    "건당평균": average,
    "하루평균": total / period_days,
    "월별지출": {
        month: float(row["지출액"])
        for month, row in monthly.iterrows()
    },
    "카테고리별지출": {
        category: float(row["지출액"])
        for category, row in category_totals.iterrows()
    },
}

# 자료나 필터가 바뀌면 이전 AI 결과를 제거합니다.
signature_source = (
    filtered.to_csv(index=False)
    + summary_text
    + json.dumps(payload, ensure_ascii=False)
)

data_signature = hashlib.sha256(
    signature_source.encode("utf-8")
).hexdigest()

if st.session_state.get("report_signature") != data_signature:
    st.session_state["report_signature"] = data_signature
    st.session_state.pop("ai_report", None)


# ==================================================
# AI 리포트
# ==================================================

st.subheader("5. AI 리포트")

st.write(
    "AI에는 총액·월별·카테고리별 집계를 보냅니다. "
    "원본 CSV, 개별 사용처, 검색어는 보내지 않습니다."
)

with st.expander("OpenAI API 연결 설정", expanded=True):
    api_key_input = st.text_input(
        "OpenAI API 키",
        type="password",
        help="API 키는 코드나 채팅에 넣지 말고 여기에 입력하세요.",
    )

    # 환경변수가 설정되어 있다면 그 키도 사용할 수 있습니다.
    api_key = (
        api_key_input.strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )

    consent = st.checkbox(
        "지출 집계를 OpenAI에 전송해 AI 리포트를 생성합니다."
    )

    st.caption(
        "API 사용에는 비용이 발생할 수 있어요. "
        "리포트 생성 버튼을 누를 때만 요청합니다."
    )

if st.button("✨ AI 리포트 생성", type="primary"):
    if not api_key:
        st.warning("OpenAI API 키를 입력해주세요.")

    elif not consent:
        st.warning("지출 집계 전송 항목을 체크해주세요.")

    else:
        try:
            with st.spinner("AI가 리포트를 작성하고 있어요..."):
                with OpenAI(
                    api_key=api_key,
                    timeout=60.0,
                    max_retries=0,
                ) as client:
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        store=False,
                        instructions=(
                            "당신은 개인 지출 분석 도우미입니다. "
                            "제공된 집계만 근거로 한국어 리포트를 작성하세요. "
                            "구성은 1. 총지출 요약, 2. 소비 패턴, "
                            "3. 실천 가능한 절약 제안 3가지, "
                            "4. 분석의 한계입니다. "
                            "소득, 사용처, 반복 결제는 추측하지 마세요. "
                            "카테고리는 키워드로 추정한 결과입니다. "
                            "월별 금액은 일부 기간의 자료일 수 있으므로 "
                            "증감을 소비 습관의 변화라고 단정하지 마세요. "
                            "검색 필터가 적용되었다면 전체 지출이 아닌 "
                            "일부 내역의 분석임을 명시하세요. "
                            "환불과 수입은 반영하지 않았습니다. "
                            "절약 목표 금액은 실제 집계와 구분하고 "
                            "가정 또는 제안이라고 표시하세요. "
                            "마크다운 표, 별표, 이모지 없이 "
                            "번호와 문단으로 1000자 이내로 작성하세요."
                        ),
                        input=json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                        max_output_tokens=1800,
                    )

                if response.status != "completed":
                    raise ValueError(
                        "리포트 생성이 완료되지 않았어요. 다시 시도해주세요."
                    )

                ai_text = response.output_text.strip()

                if not ai_text:
                    raise ValueError(
                        "AI가 리포트 내용을 반환하지 않았어요."
                    )

                st.session_state["ai_report"] = ai_text

            st.success("AI 리포트가 완성됐어요!")

        except AuthenticationError:
            st.error("API 키가 올바른지 확인해주세요.")

        except RateLimitError:
            st.error(
                "API 사용 한도 또는 결제 잔액을 확인해주세요. "
                "요청이 많다면 잠시 후 다시 시도하세요."
            )

        except APIConnectionError:
            st.error(
                "OpenAI에 연결하지 못했어요. 인터넷 연결을 확인해주세요."
            )

        except APIStatusError as error:
            st.error(
                f"API 요청 실패: 상태 코드 {error.status_code}. "
                "모델 사용 권한과 계정 상태를 확인해주세요."
            )

        except ValueError as error:
            st.error(str(error))


ai_report = st.session_state.get("ai_report")

if ai_report:
    st.text(ai_report)
else:
    st.info(
        "아직 AI 리포트가 없어요. "
        "API 키 없이도 아래에서 기본 집계 PDF를 만들 수 있어요."
    )


# ==================================================
# PDF 다운로드
# ==================================================

st.subheader("6. PDF 다운로드")

with st.expander("PDF 한글 글꼴 설정"):
    st.write(
        "Windows에서는 맑은 고딕을 자동으로 사용합니다. "
        "다른 환경에서는 한글을 지원하는 TTF 파일을 올려주세요."
    )

    font_file = st.file_uploader(
        "한글 TTF 글꼴 업로드 — 선택 사항",
        type=["ttf"],
        key="report_font",
    )

font_bytes = (
    font_file.getvalue()
    if font_file is not None
    else None
)

analysis_text = ai_report or basic_analysis

pdf_signature = hashlib.sha256(
    (
        data_signature
        + analysis_text
        + (
            hashlib.sha256(font_bytes).hexdigest()
            if font_bytes is not None
            else "windows-default-font"
        )
    ).encode("utf-8")
).hexdigest()

# 자료, AI 리포트 또는 글꼴이 바뀌면 이전 PDF를 제거합니다.
if st.session_state.get("pdf_signature") != pdf_signature:
    st.session_state["pdf_signature"] = pdf_signature
    st.session_state.pop("report_pdf", None)

if ai_report:
    st.write("현재 AI 리포트를 포함한 PDF를 만듭니다.")
else:
    st.write("기본 집계와 간단한 안내를 포함한 PDF를 만듭니다.")

if st.button("📄 PDF 만들기"):
    try:
        with st.spinner("한글 PDF를 만들고 있어요..."):
            st.session_state["report_pdf"] = create_pdf(
                summary_text,
                analysis_text,
                font_bytes,
            )

        st.success("PDF가 준비됐어요. 아래 버튼으로 내려받으세요.")

    except Exception:
        st.error(
            "PDF를 만들지 못했어요. 한글 글꼴을 확인해주세요. "
            "업로드한 글꼴이 있다면 정적 TrueType 형식의 "
            "한글 TTF 파일로 바꿔보세요."
        )

if "report_pdf" in st.session_state:
    st.download_button(
        label="📥 총지출 리포트 PDF 다운로드",
        data=st.session_state["report_pdf"],
        file_name="expense_report.pdf",
        mime="application/pdf",
    )
    