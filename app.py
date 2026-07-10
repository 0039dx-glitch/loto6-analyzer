import io
import itertools
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="ロト6分析システム",
    page_icon="🎯",
    layout="wide",
)

DATA_URL = "https://loto6.thekyo.jp/data/loto6.csv"
NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


# =========================================================
# データ取得
# =========================================================

def extract_number(value):
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+", str(value).replace(",", ""))
    return int(match.group()) if match else np.nan


def find_column(columns, words):
    for word in words:
        for column in columns:
            cleaned = str(column).replace(" ", "").replace("　", "")
            if word in cleaned:
                return column
    return None


def normalize_data(raw):
    raw = raw.copy()
    raw = raw.loc[
        :,
        ~raw.columns.astype(str).str.startswith("Unnamed")
    ]

    columns = list(raw.columns)

    draw_col = find_column(
        columns,
        ["開催回", "抽せん回", "抽選回", "回号", "回別"],
    )
    date_col = find_column(
        columns,
        ["抽せん日", "抽選日", "日付"],
    )
    bonus_col = find_column(
        columns,
        ["ボーナス数字", "BONUS数字", "BONUS", "ボーナス"],
    )

    number_columns = []

    for position in range(1, 7):
        found = find_column(
            columns,
            [
                f"第{position}数字",
                f"本数字{position}",
                f"数字{position}",
            ],
        )
        if found is not None:
            number_columns.append(found)

    # 列名を識別できないCSVの場合
    if (
        draw_col is None
        or date_col is None
        or len(number_columns) != 6
    ):
        if len(raw.columns) < 9:
            raise ValueError(
                "CSVから回号、日付、本数字を識別できません。"
            )

        draw_col = raw.columns[0]
        date_col = raw.columns[1]
        number_columns = list(raw.columns[2:8])
        bonus_col = raw.columns[8]

    result = pd.DataFrame()
    result["draw_no"] = raw[draw_col].map(extract_number)
    result["draw_date"] = pd.to_datetime(
        raw[date_col],
        errors="coerce",
    )

    for i, column in enumerate(number_columns, start=1):
        result[f"n{i}"] = raw[column].map(extract_number)

    if bonus_col is not None:
        result["bonus"] = raw[bonus_col].map(extract_number)
    else:
        result["bonus"] = np.nan

    result = result.dropna(
        subset=["draw_no", "draw_date"] + NUM_COLS
    ).copy()

    result["draw_no"] = result["draw_no"].astype(int)

    for column in NUM_COLS:
        result[column] = result[column].astype(int)

    valid = []

    for _, row in result.iterrows():
        numbers = [int(row[column]) for column in NUM_COLS]
        valid.append(
            len(set(numbers)) == 6
            and all(1 <= number <= 43 for number in numbers)
        )

    result = result.loc[valid]
    result = result.drop_duplicates("draw_no", keep="last")
    result = result.sort_values("draw_no").reset_index(drop=True)

    return result


def read_csv_bytes(content):
    last_error = None

    for encoding in ["cp932", "shift_jis", "utf-8-sig", "utf-8"]:
        try:
            raw = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
            )
            return normalize_data(raw)
        except Exception as error:
            last_error = error

    raise ValueError(f"CSVを読み込めませんでした: {last_error}")


@st.cache_data(ttl=3600)
def download_data():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 Windows NT 10.0 "
            "AppleWebKit/537.36 Chrome Safari"
        )
    }

    response = requests.get(
        DATA_URL,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    return read_csv_bytes(response.content)


# =========================================================
# 分析関数
# =========================================================

def frequency_table(data):
    counts = {number: 0 for number in range(1, 44)}

    for _, row in data.iterrows():
        for column in NUM_COLS:
            counts[int(row[column])] += 1

    result = pd.DataFrame(
        {
            "数字": list(counts.keys()),
            "出現回数": list(counts.values()),
        }
    )

    expected = len(data) * 6 / 43
    result["期待回数"] = expected
    result["期待値との差"] = result["出現回数"] - expected
    result["出現率"] = (
        result["出現回数"] / len(data) * 100
        if len(data)
        else 0
    )

    return result


def cycle_table(data):
    rows = []
    latest_position = len(data) - 1

    for number in range(1, 44):
        positions = []

        for position, (_, row) in enumerate(data.iterrows()):
            numbers = [int(row[column]) for column in NUM_COLS]

            if number in numbers:
                positions.append(position)

        gaps = np.diff(positions)

        average = (
            float(np.mean(gaps))
            if len(gaps)
            else np.nan
        )
        median = (
            float(np.median(gaps))
            if len(gaps)
            else np.nan
        )
        maximum = (
            int(np.max(gaps))
            if len(gaps)
            else np.nan
        )
        current_gap = (
            latest_position - positions[-1]
            if positions
            else len(data)
        )
        ratio = (
            current_gap / average
            if pd.notna(average) and average > 0
            else np.nan
        )

        rows.append(
            {
                "数字": number,
                "出現回数": len(positions),
                "平均周期": average,
                "周期中央値": median,
                "最長周期": maximum,
                "現在の空白回数": current_gap,
                "周期比": ratio,
            }
        )

    return pd.DataFrame(rows)


def pattern_table(data):
    rows = []
    previous = None

    for _, row in data.iterrows():
        numbers = sorted(
            int(row[column]) for column in NUM_COLS
        )

        odd = sum(number % 2 == 1 for number in numbers)
        low = sum(number <= 21 for number in numbers)
        consecutive = sum(
            numbers[i + 1] - numbers[i] == 1
            for i in range(5)
        )
        repeated = (
            len(set(numbers) & set(previous))
            if previous is not None
            else 0
        )

        rows.append(
            {
                "回号": int(row["draw_no"]),
                "合計値": sum(numbers),
                "奇数：偶数": f"{odd}：{6 - odd}",
                "低数字：高数字": f"{low}：{6 - low}",
                "連続数字ペア数": consecutive,
                "前回引っ張り数": repeated,
            }
        )

        previous = numbers

    return pd.DataFrame(rows)


def pair_tables(data):
    pair_counts = np.zeros((43, 43), dtype=int)
    single_counts = np.zeros(43, dtype=int)

    for _, row in data.iterrows():
        numbers = sorted(
            int(row[column]) for column in NUM_COLS
        )

        for number in numbers:
            single_counts[number - 1] += 1

        for first, second in itertools.combinations(numbers, 2):
            pair_counts[first - 1, second - 1] += 1
            pair_counts[second - 1, first - 1] += 1

    lift = np.full((43, 43), np.nan)

    for first in range(43):
        for second in range(43):
            if first == second:
                continue

            if single_counts[first] and single_counts[second]:
                lift[first, second] = (
                    pair_counts[first, second] * len(data)
                    / (
                        single_counts[first]
                        * single_counts[second]
                    )
                )

    return pair_counts, lift


# =========================================================
# データ読み込み
# =========================================================

st.title("🎯 ロト6 数字データ分析システム")
st.caption(
    "第1回から最新回までの当せん番号を分析します。"
)

try:
    df = download_data()
except Exception as error:
    st.error("ネットからのデータ取得に失敗しました。")
    st.code(str(error))
    st.stop()


# CSV追加機能
st.sidebar.header("データ管理")

if st.sidebar.button("最新データを再取得"):
    st.cache_data.clear()
    st.rerun()

uploaded = st.sidebar.file_uploader(
    "追加・更新CSVを読み込む",
    type=["csv"],
)

if uploaded is not None:
    try:
        uploaded_df = read_csv_bytes(uploaded.getvalue())
        df = pd.concat([df, uploaded_df])
        df = df.drop_duplicates(
            "draw_no",
            keep="last",
        ).sort_values("draw_no")
        st.sidebar.success("CSVを追加しました。")
    except Exception as error:
        st.sidebar.error(str(error))


latest = df.iloc[-1]
latest_numbers = [
    int(latest[column]) for column in NUM_COLS
]

col1, col2, col3, col4 = st.columns(4)

col1.metric("収録回数", f"{len(df):,}回")
col2.metric("最新回", f"第{int(latest['draw_no'])}回")
col3.metric(
    "最新抽せん日",
    latest["draw_date"].strftime("%Y/%m/%d"),
)
col4.metric(
    "最新本数字",
    "・".join(f"{number:02d}" for number in latest_numbers),
)

if pd.notna(latest["bonus"]):
    st.info(
        f"最新ボーナス数字：{int(latest['bonus']):02d}"
    )

tabs = st.tabs(
    [
        "前回→次回",
        "出現回数",
        "周期分析",
        "パターン",
        "ペア相性",
        "追加分析・履歴",
    ]
)


# =========================================================
# 1 前回数字の次に出た数字
# =========================================================

with tabs[0]:
    st.header("前当せん数字の次に出た数字")

    selected = st.selectbox(
        "前回に出た基準数字",
        range(1, 44),
    )

    next_numbers = []
    details = []

    ordered = df.reset_index(drop=True)

    for index in range(len(ordered) - 1):
        current = [
            int(ordered.loc[index, column])
            for column in NUM_COLS
        ]

        if selected in current:
            following = [
                int(ordered.loc[index + 1, column])
                for column in NUM_COLS
            ]

            next_numbers.extend(following)

            details.append(
                {
                    "基準回": int(
                        ordered.loc[index, "draw_no"]
                    ),
                    "次回": int(
                        ordered.loc[index + 1, "draw_no"]
                    ),
                    "次回本数字": "・".join(
                        f"{number:02d}"
                        for number in following
                    ),
                }
            )

    transition = pd.Series(
        next_numbers
    ).value_counts().reindex(
        range(1, 44),
        fill_value=0,
    ).rename_axis("次回数字").reset_index(
        name="出現回数"
    )

    fig = px.bar(
        transition,
        x="次回数字",
        y="出現回数",
        color="出現回数",
        title=f"数字{selected}が出た次の回の数字",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(xaxis_dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        transition.sort_values(
            "出現回数",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("該当した全抽せん履歴")
    st.dataframe(
        pd.DataFrame(details),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 2 出現回数
# =========================================================

with tabs[1]:
    st.header("直近回数を指定した出現回数")

    recent_count = st.slider(
        "分析する直近回数",
        min_value=5,
        max_value=len(df),
        value=min(100, len(df)),
    )

    recent_df = df.tail(recent_count)
    frequency = frequency_table(recent_df)

    fig = px.bar(
        frequency,
        x="数字",
        y="出現回数",
        color="期待値との差",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"直近{recent_count}回の出現回数",
    )
    fig.add_hline(
        y=recent_count * 6 / 43,
        line_dash="dash",
    )
    fig.update_layout(xaxis_dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    hot, cold = st.columns(2)

    with hot:
        st.subheader("🔥 出現上位")
        st.dataframe(
            frequency.sort_values(
                "出現回数",
                ascending=False,
            ).head(15),
            hide_index=True,
            use_container_width=True,
        )

    with cold:
        st.subheader("❄️ 出現下位")
        st.dataframe(
            frequency.sort_values(
                "出現回数",
                ascending=True,
            ).head(15),
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# 3 周期
# =========================================================

with tabs[2]:
    st.header("数字の出現周期")

    cycles = cycle_table(df)

    fig = px.scatter(
        cycles,
        x="平均周期",
        y="現在の空白回数",
        size="出現回数",
        color="周期比",
        text="数字",
        color_continuous_scale="Turbo",
        title="平均周期と現在の空白回数",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cycles.sort_values(
            "周期比",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 4 パターン
# =========================================================

with tabs[3]:
    st.header("よく出現していたパターン")

    pattern_count = st.slider(
        "分析対象回数",
        min_value=20,
        max_value=len(df),
        value=min(500, len(df)),
        key="pattern_count",
    )

    patterns = pattern_table(df.tail(pattern_count))

    pattern_name = st.selectbox(
        "表示するパターン",
        [
            "合計値",
            "奇数：偶数",
            "低数字：高数字",
            "連続数字ペア数",
            "前回引っ張り数",
        ],
    )

    if pattern_name == "合計値":
        fig = px.histogram(
            patterns,
            x="合計値",
            nbins=35,
            title="本数字合計値の分布",
        )
    else:
        summary = (
            patterns[pattern_name]
            .value_counts()
            .reset_index()
        )
        summary.columns = [pattern_name, "出現回数"]

        fig = px.bar(
            summary,
            x=pattern_name,
            y="出現回数",
            color="出現回数",
            title=f"{pattern_name}の出現パターン",
        )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 5 ペア・相性
# =========================================================

with tabs[4]:
    st.header("ペア数字・相性分析")

    pair_count = st.slider(
        "ペア分析対象回数",
        min_value=20,
        max_value=len(df),
        value=min(500, len(df)),
        key="pair_count",
    )

    pair_counts, pair_lift = pair_tables(
        df.tail(pair_count)
    )

    pair_number = st.selectbox(
        "相性を確認する数字",
        range(1, 44),
        key="pair_number",
    )

    pair_result = pd.DataFrame(
        {
            "相手数字": range(1, 44),
            "共起回数": pair_counts[pair_number - 1],
            "相性リフト": pair_lift[pair_number - 1],
        }
    )

    pair_result = pair_result[
        pair_result["相手数字"] != pair_number
    ].sort_values(
        ["相性リフト", "共起回数"],
        ascending=False,
    )

    st.dataframe(
        pair_result,
        use_container_width=True,
        hide_index=True,
    )

    heat_frame = pd.DataFrame(
        pair_lift,
        index=range(1, 44),
        columns=range(1, 44),
    )

    fig = px.imshow(
        heat_frame,
        labels={
            "x": "相手数字",
            "y": "基準数字",
            "color": "相性リフト",
        },
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="全数字ペア相性ヒートマップ",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "相性リフト1以上は、個別出現率から計算される"
        "期待値より多く同時出現したことを示します。"
    )


# =========================================================
# 6 追加分析・履歴
# =========================================================

with tabs[5]:
    st.header("追加分析：空白回数・短期トレンド")

    cycles = cycle_table(df)
    recent_frequency = frequency_table(df.tail(min(50, len(df))))
    all_frequency = frequency_table(df)

    trend = recent_frequency[
        ["数字", "出現率"]
    ].rename(
        columns={"出現率": "直近50回出現率"}
    )

    trend = trend.merge(
        all_frequency[
            ["数字", "出現率"]
        ].rename(
            columns={"出現率": "全期間出現率"}
        ),
        on="数字",
    )

    trend["短期－全期間"] = (
        trend["直近50回出現率"]
        - trend["全期間出現率"]
    )

    trend = trend.merge(
        cycles[
            ["数字", "現在の空白回数", "周期比"]
        ],
        on="数字",
    )

    fig = px.bar(
        trend,
        x="数字",
        y="短期－全期間",
        color="短期－全期間",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="直近50回と全期間の出現率差",
    )
    fig.update_layout(xaxis_dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("現在の空白回数ランキング")

    st.dataframe(
        trend.sort_values(
            "現在の空白回数",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("直近60回の出現マップ")

    heat_data = df.tail(min(60, len(df)))
    matrix = np.zeros((len(heat_data), 43), dtype=int)

    for row_index, (_, row) in enumerate(
        heat_data.iterrows()
    ):
        for column in NUM_COLS:
            matrix[row_index, int(row[column]) - 1] = 1

    fig = px.imshow(
        matrix,
        x=list(range(1, 44)),
        y=heat_data["draw_no"].astype(str),
        labels={
            "x": "数字",
            "y": "回号",
            "color": "出現",
        },
        color_continuous_scale=[
            [0, "#f1f5f9"],
            [1, "#ef4444"],
        ],
        aspect="auto",
        title="数字出現ヒートマップ",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("当せん番号履歴")

    history = df.copy()
    history["本数字"] = history.apply(
        lambda row: "・".join(
            f"{int(row[column]):02d}"
            for column in NUM_COLS
        ),
        axis=1,
    )

    display_history = history[
        ["draw_no", "draw_date", "本数字", "bonus"]
    ].rename(
        columns={
            "draw_no": "回号",
            "draw_date": "抽せん日",
            "bonus": "ボーナス",
        }
    ).sort_values(
        "回号",
        ascending=False,
    )

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True,
    )

    csv_output = df.to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        "全データをCSVでダウンロード",
        data=csv_output,
        file_name="loto6_analysis_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.warning(
    "このシステムは過去データを集計・可視化するものです。"
    "将来の当せんを保証するものではありません。"
)
