# -*- coding: utf-8 -*-
"""
KOSDAQ 150 Supply-Demand Divergence Analysis
=============================================
Author : Senior Data Scientist (ARM / Tiny-Data Philosophy)
Env    : conda activate quantdev
Run    : python kosdaq150_divergence.py

철학적 배경
-----------
Andrew Gelman의 ARM 접근법은 "데이터를 이산적으로 자르지 말고(No Hard Cutoff),
연속형 변수 분포를 있는 그대로 관찰하라"고 강조한다.
주식 수급 분석에서도 마찬가지로, 특정 수익률 이하를 제거하는 필터링 대신
Price Return과 기관 순매수 비율을 연속형 지표로 유지하여
사용자가 2차원 공간에서 이상치·군집을 직접 판단하도록 설계했다.

'수급 다이버전스(Divergence)'란 가격 추세와 수급 방향이 괴리되는 현상을 가리킨다.
가격은 하락하는데 기관이 꾸준히 매수하는 종목은 '잠재적 매집(Stealth Accumulation)'
가설을 세울 수 있으며, 이는 인과 추론의 출발점이 된다.

NOTE: KRX 서버는 주말/공휴일에 일부 API를 차단한다.
      - OHLCV(NAVER 경유): 항상 작동
      - 투자자별 순매수(KRX 직접): 평일에만 작동
      - 종목 리스트(KRX 직접): 평일에만 작동
      주말에 실행 시 hardcoded 종목 리스트를 사용하며,
      투자자별 순매수 데이터가 없는 종목은 스킵된다.
"""

import warnings
from datetime import datetime

import pandas as pd
from pykrx import stock
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ============================================================================
# 0. KOSDAQ 150 HARDCODED FALLBACK (2025-12 기준 최신 구성)
# ============================================================================
# KRX 서버가 주말/공휴일에 종목 리스트 API를 차단하므로, fallback 용도.
# 평일에는 pykrx API를 통해 최신 구성 종목을 자동 갱신한다.
KOSDAQ_150_FALLBACK = [
    "028300", "247540", "086520", "196170", "403870", "328130", "277810",
    "041510", "035720", "263750", "095340", "036930", "140860", "298380",
    "058470", "036540", "357780", "950160", "078600", "067630", "039030",
    "240810", "094360", "950210", "222160", "226330", "272110", "214150",
    "089010", "950170", "060310", "225570", "114840", "335890", "322310",
    "018310", "033640", "131970", "352480", "000250", "025900", "251970",
    "068760", "036810", "031310", "067080", "950200", "091990", "348210",
    "053800", "112040", "293490", "041920", "078340", "069080", "122870",
    "035900", "253450", "067310", "352820", "399720", "389030",
    # 추가 코스닥 150 종목들 (시총 상위)
    "203690", "052690", "057050", "048410", "145020", "064760",
    "215600", "046890", "141080", "123040", "108320", "042000",
    "032640", "108860", "099190", "090460", "095610", "101490",
    "222080", "066970", "109960", "051360", "090710", "068270",
    "054620", "041190", "038460", "365340", "009520", "039200",
    "084370", "019540", "260930", "192250", "043150", "054050",
    "090350", "036620", "161890", "234300", "205470", "060250",
    "033100", "044490", "241560", "053030", "263860", "060280",
    "217190", "238090", "236200", "098120", "389260", "190510",
    "110020", "340570", "405640", "078150", "397030", "204270",
    "200470", "230360", "183300", "417500", "096530", "053610",
    "336260", "337930", "354200", "148780", "039440", "315640",
    "330860", "950140", "382480", "418420", "419270", "160980",
    "445180", "376300", "073010", "048260", "166090",
]


# ============================================================================
# 1. DATA PIPELINE
# ============================================================================

def get_kosdaq150_tickers() -> list[str]:
    """코스닥 150 구성 종목 코드를 반환한다.

    전략: pykrx API를 먼저 시도하고, 실패하면 hardcoded 리스트를 사용한다.
    """
    try:
        tickers = stock.get_index_portfolio_deposit_file("2056")
        if isinstance(tickers, list) and len(tickers) > 100:
            print(f"[INFO] KRX API로 코스닥 150 구성 종목 {len(tickers)}개 확보")
            return tickers
        # DataFrame returned or empty list → fallback
        if hasattr(tickers, '__len__') and len(tickers) > 100:
            return list(tickers)
    except Exception:
        pass

    # 날짜를 지정하여 재시도
    today = datetime.now().strftime("%Y%m%d")
    try:
        tickers = stock.get_index_portfolio_deposit_file("2056", today)
        if isinstance(tickers, list) and len(tickers) > 100:
            print(f"[INFO] KRX API(날짜 지정)로 코스닥 150 구성 종목 {len(tickers)}개 확보")
            return tickers
    except Exception:
        pass

    print("[INFO] KRX 서버 미응답 — Hardcoded 코스닥 150 리스트 사용 "
          f"({len(KOSDAQ_150_FALLBACK)}개)")
    return KOSDAQ_150_FALLBACK


def fetch_data(start: str = "20251101",
               end: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """코스닥 150 구성 종목의 OHLCV + 투자자별 순매수 데이터를 수집한다.

    Parameters
    ----------
    start : str  — 분석 시작일 (YYYYMMDD). 충분한 Base 기간 확보를 위해
                   기본값을 2025-11-01로 설정.
    end   : str  — 분석 종료일. None이면 오늘 날짜 사용.

    Returns
    -------
    data_dict  : {ticker: DataFrame} — OHLCV + 누적 순매수 컬럼 포함
    name_map   : {ticker: 종목명}

    설계 철학
    ---------
    `for`문은 pykrx API 호출(종목별로 불가피)에만 사용하고,
    이후의 누적 순매수 계산은 pandas 벡터 연산(`cumsum`)으로 처리한다.
    불필요한 `.copy()` 호출을 지양하고, 원본 DataFrame에 직접 컬럼을 추가한다.
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")

    tickers = get_kosdaq150_tickers()
    print(f"[INFO] 데이터 수집 시작: {len(tickers)}개 종목, 기간 {start}~{end}")

    data_dict: dict[str, pd.DataFrame] = {}
    name_map: dict[str, str] = {}
    inv_fail_count = 0

    for i, ticker in enumerate(tickers):
        try:
            name = stock.get_market_ticker_name(ticker)
            if not name:
                continue
            name_map[ticker] = name

            # --- OHLCV (NAVER 경유 — 항상 작동) ---
            ohlcv = stock.get_market_ohlcv_by_date(start, end, ticker)
            if ohlcv.empty or len(ohlcv) < 5:
                continue

            # --- 투자자별 순매수 수량 (KRX 직접 — 평일만 작동) ---
            # pykrx의 get_market_trading_volume_by_date는 KRX API에 의존한다.
            # 주말/공휴일에는 빈 DataFrame을 반환한다.
            inv = stock.get_market_trading_volume_by_date(start, end, ticker)

            if inv.empty or len(inv) == 0:
                inv_fail_count += 1
                if inv_fail_count <= 3:
                    # 몇 개가 연속 실패하면 KRX 서버 다운으로 판단
                    continue
                elif inv_fail_count == 4:
                    print("[WARN] KRX 투자자 데이터 서버 미응답 (주말/공휴일 가능). "
                          "OHLCV만으로 수집을 계속합니다.")
                # OHLCV만 저장하고, 투자자 컬럼은 0으로 채움
                ohlcv["기관합계"] = 0
                ohlcv["개인"] = 0
                ohlcv["외국인합계"] = 0
                merged = ohlcv
            else:
                # 투자자 컬럼명 정리
                for col_name in ["기관합계", "개인", "외국인합계"]:
                    if col_name not in inv.columns:
                        inv[col_name] = 0

                # OHLCV와 투자자 데이터를 날짜 인덱스 기준으로 조인
                merged = ohlcv.join(inv[["기관합계", "개인", "외국인합계"]], how="left")
                merged[["기관합계", "개인", "외국인합계"]] = (
                    merged[["기관합계", "개인", "외국인합계"]].fillna(0)
                )

            # --- 벡터화: 누적 순매수량 ---
            # cumsum()은 O(n) 벡터 연산으로, 시계열 전체에 대해 한 번만 수행.
            # 누적 순매수량의 방향과 기울기 변화가 Regime Shift 탐지의 핵심 입력이다.
            merged["기관_누적"] = merged["기관합계"].cumsum()
            merged["개인_누적"] = merged["개인"].cumsum()
            merged["외국인_누적"] = merged["외국인합계"].cumsum()

            data_dict[ticker] = merged

            if (i + 1) % 30 == 0:
                print(f"  ... {i + 1}/{len(tickers)} 종목 수집 완료")

        except Exception as e:
            # 개별 종목 실패는 무시하고 계속 진행
            continue

    print(f"[INFO] 총 {len(data_dict)}개 종목 데이터 수집 완료")
    if inv_fail_count > 3:
        print("[INFO] 투자자별 수급 데이터 없음 — Scatter Plot의 inst_ratio는 0입니다.")
        print("[INFO] 평일에 다시 실행하면 수급 데이터가 반영됩니다.")
    return data_dict, name_map


# ============================================================================
# 2. SCORING — 연속형 변수 기반 (No Hard Cutoff)
# ============================================================================

def compute_scores(data_dict: dict[str, pd.DataFrame],
                   name_map: dict[str, str]) -> pd.DataFrame:
    """종목별 Price Return과 기관 순매수 비율을 계산한다.

    ARM 관점에서의 스코어링 철학
    ----------------------------
    결정론적 필터링(예: "수익률 -10% 이하 제거")은 정보를 파괴한다.
    대신, 두 축을 연속형 변수로 유지하면:
      1. 사용자가 2D 공간에서 자연스런 군집(cluster)과 이상치(outlier)를 관찰할 수 있다.
      2. 추후 베이지안 회귀나 Multilevel Model로 확장할 때 데이터 손실이 없다.

    Price Return: 기간 내 단순 수익률 → 가격 추세의 방향과 크기
    Inst_Ratio  : (기관 누적 순매수량) / (전체 누적 거래량)
                  → 전체 거래 중 기관의 순방향 참여 정도.
                  이 비율이 양(+)이면서 가격이 하락한 종목은
                  '잠재적 매집(Stealth Accumulation)' 가설의 후보가 된다.
    """
    records = []

    for ticker, df in data_dict.items():
        if len(df) < 2:
            continue

        first_close = df["종가"].iloc[0]
        last_close = df["종가"].iloc[-1]
        price_return = (last_close / first_close) - 1 if first_close != 0 else 0.0

        total_volume = df["거래량"].sum()
        inst_cumulative = df["기관합계"].sum()
        inst_ratio = inst_cumulative / total_volume if total_volume != 0 else 0.0

        records.append({
            "종목코드": ticker,
            "종목명": name_map.get(ticker, ticker),
            "price_return": price_return,
            "inst_ratio": inst_ratio,
        })

    scores_df = pd.DataFrame(records)
    print(f"[INFO] 스코어링 완료: {len(scores_df)}개 종목")
    return scores_df


# ============================================================================
# 3. SCATTER PLOT — 전체 종목 2D 분포
# ============================================================================

def plot_scatter(scores_df: pd.DataFrame) -> go.Figure:
    """Price Return vs Inst_Ratio 산점도를 그린다.

    시각화 의도
    -----------
    Gelman은 "Plot the data before modeling"을 반복 강조한다.
    이 산점도는 모델링 이전에 전체 분포의 형태를 파악하는 EDA 도구이다.

    - 좌상단(가격↓, 기관 매수↑): 잠재적 매집 — Divergence 후보
    - 우하단(가격↑, 기관 매도↑): 기관 차익 실현 가능 영역
    - 이상치(Outlier)는 개별 조사 대상이 된다.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scores_df["price_return"],
        y=scores_df["inst_ratio"],
        mode="markers+text",
        text=scores_df["종목명"],
        textposition="top center",
        textfont=dict(size=8, color="gray"),
        customdata=scores_df["종목코드"],
        marker=dict(
            size=9,
            color=scores_df["price_return"],
            colorscale="RdYlGn",
            colorbar=dict(title="Price Return"),
            opacity=0.85,
            line=dict(width=0.5, color="white"),
        ),
        hovertemplate=(
            "<b>%{text}</b> (%{customdata})<br>"
            "Price Return: %{x:.2%}<br>"
            "Inst Ratio: %{y:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    # 원점 참조선 — Divergence 판별의 시각적 앵커
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=dict(
            text="KOSDAQ 150 — Supply-Demand Divergence Map",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Price Return (기간 수익률)",
            tickformat=".0%",
            zeroline=True,
        ),
        yaxis=dict(
            title="Inst_Ratio (기관 순매수 / 전체 거래량)",
            zeroline=True,
        ),
        template="plotly_white",
        height=700,
        width=1100,
        annotations=[
            dict(
                text="← 가격 하락 · 기관 매수 ↑ (잠재적 매집)",
                x=0.02, y=0.98, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="crimson"),
            ),
            dict(
                text="가격 상승 · 기관 매수 ↑ (순풍) →",
                x=0.98, y=0.98, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="seagreen"),
                xanchor="right",
            ),
        ],
    )

    return fig


# ============================================================================
# 4. INDIVIDUAL STOCK DASHBOARD
# ============================================================================

# 매물대 고점(High Volume Node)에서 캔들 차트에 수평 참조선을 그리기 위한
# 색상 팔레트. 거래량이 집중된 가격대는 잠재적 지지/저항선이며,
# 기관 매집의 '비용 기저(Cost Basis)'를 시각적으로 추정하는 데 유용하다.
_HVN_COLORS = [
    "rgba(239, 83, 80, 0.45)",   # 빨강 계열
    "rgba(255, 152, 0, 0.40)",   # 주황
    "rgba(171, 71, 188, 0.40)",  # 보라
    "rgba(66, 165, 245, 0.35)",  # 파랑
    "rgba(38, 166, 154, 0.35)",  # 틸
]


def plot_dashboard(ticker: str,
                   data_dict: dict[str, pd.DataFrame],
                   name_map: dict[str, str],
                   regime_date: str = "2026-01-15") -> go.Figure:
    """특정 종목의 가격·매물대·수급 대시보드를 그린다.

    Subplot 구조 (사용자 퀀트 툴 레이아웃 재현)
    -------------------------------------------
    Row 1, Col 1 (~75%) : 캔들스틱 + MA(5,10,20,60) + 매물대 수평 참조선
    Row 1, Col 2 (~25%) : Vol Profile (가격대별 누적 거래량, 가로 막대)
                          y축은 캔들 차트와 공유 → 가격대 1:1 매칭
    Row 2, Col 1-2(병합): 개인·기관·외국인 누적 순매수량 라인차트

    Regime Shift Marker
    -------------------
    2026-01-15 부근에 수직 점선을 상하단 모두에 표시.
    Interrupted Time Series의 'Intervention Point'에 해당한다.
    """
    df = data_dict[ticker]
    name = name_map.get(ticker, ticker)

    # --- 이동평균선 4종 (벡터 연산) ---
    # 복수의 MA를 겹쳐 그리면 단기·중기 추세의 정렬(alignment) 여부를
    # 한눈에 판단할 수 있다. 정배열/역배열은 추세 강도의 proxy이다.
    ma5  = df["종가"].rolling(window=5,  min_periods=1).mean()
    ma10 = df["종가"].rolling(window=10, min_periods=1).mean()
    ma20 = df["종가"].rolling(window=20, min_periods=1).mean()
    ma60 = df["종가"].rolling(window=60, min_periods=1).mean()

    # --- Volume Profile 계산 ---
    # 가격 범위를 균등 분할하여 각 대역의 누적 거래량을 집계한다.
    # 매물대는 지지/저항 수준을 시각적으로 드러내며,
    # 수급 분석과 결합하면 '어떤 가격대에서 기관이 매집했는가'를 추론할 수 있다.
    n_bins = 40
    price_bins = pd.cut(df["종가"], bins=n_bins, retbins=True)
    bin_edges = price_bins[1]
    bin_labels = [
        (bin_edges[i] + bin_edges[i + 1]) / 2
        for i in range(len(bin_edges) - 1)
    ]
    price_bin_series = pd.cut(
        df["종가"], bins=bin_edges, labels=bin_labels, include_lowest=True,
    )
    volume_profile = df.groupby(price_bin_series, observed=True)["거래량"].sum()

    price_min, price_max = df["저가"].min(), df["고가"].max()
    y_range = [price_min * 0.97, price_max * 1.03]

    # --- High Volume Nodes (HVN): 매물대 상위 5개 가격대 ---
    # 거래량 집중 가격대는 시장 참여자의 '합의 가격(consensus price)'이며,
    # 향후 되돌림 시 지지/저항으로 작용할 확률이 높다.
    top_hvn = volume_profile.nlargest(5)

    # === Subplot 생성 ===
    # column_widths: 캔들 75%, Volume Profile 25% (사용자 툴 비율)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.10,
        horizontal_spacing=0.02,
        row_heights=[0.58, 0.42],
        column_widths=[0.75, 0.25],
        subplot_titles=[
            f"{name} ({ticker})",
            "Vol Profile",
            "",  # 하단 차트 제목은 y축 라벨로 대체
        ],
    )

    dates = df.index

    # ══════════════════════════════════════════════════════════
    # [R1C1] 캔들스틱 + 이동평균선 + HVN 수평 참조선
    # ══════════════════════════════════════════════════════════
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["시가"], high=df["고가"],
        low=df["저가"], close=df["종가"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name="OHLC",
        showlegend=False,
    ), row=1, col=1)

    # MA Traces — 색상은 단기→장기 순으로 따뜻→차가운 색
    ma_config = [
        (ma5,  "MA5",  "#ff9800", 1.0),  # 주황 (단기)
        (ma10, "MA10", "#ffc107", 1.0),  # 노랑
        (ma20, "MA20", "#42a5f5", 1.2),  # 파랑 (중기)
        (ma60, "MA60", "#ab47bc", 1.2),  # 보라 (장기)
    ]
    for ma_series, ma_name, ma_color, ma_width in ma_config:
        fig.add_trace(go.Scatter(
            x=dates, y=ma_series,
            mode="lines", name=ma_name,
            line=dict(color=ma_color, width=ma_width),
            showlegend=False,
        ), row=1, col=1)

    # HVN 수평 참조선 — 매물대 상위 가격대를 캔들 차트에 점선으로 표시
    # 이 가격대는 대량 거래가 발생한 '합의 영역'이므로 지지/저항 역할을 한다.
    for i, (price_level, vol) in enumerate(top_hvn.items()):
        color = _HVN_COLORS[i % len(_HVN_COLORS)]
        fig.add_shape(
            type="line",
            x0=dates[0], x1=dates[-1],
            y0=float(price_level), y1=float(price_level),
            xref="x", yref="y",
            line=dict(dash="dash", color=color, width=1.0),
        )

    fig.update_yaxes(range=y_range, row=1, col=1)

    # ══════════════════════════════════════════════════════════
    # [R1C2] Volume Profile (가로 막대) — y축을 캔들 차트와 공유
    # ══════════════════════════════════════════════════════════
    vp_prices = [float(p) for p in volume_profile.index]
    vp_volumes = volume_profile.values

    # HVN 가격대 하이라이트를 위한 색상 매핑
    hvn_set = set(float(p) for p in top_hvn.index)
    bar_colors = [
        "rgba(239, 83, 80, 0.7)" if p in hvn_set
        else "rgba(100, 149, 237, 0.5)"
        for p in vp_prices
    ]

    fig.add_trace(go.Bar(
        x=vp_volumes,
        y=vp_prices,
        orientation="h",
        marker_color=bar_colors,
        name="Vol Profile",
        showlegend=False,
    ), row=1, col=2)

    # VP y축 범위를 캔들 차트와 완전 일치
    fig.update_yaxes(range=y_range, showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="Volume", title_font_size=10, row=1, col=2)

    # ══════════════════════════════════════════════════════════
    # [R2] 누적 순매수량 라인차트
    # ══════════════════════════════════════════════════════════
    # 누적 순매수 곡선의 기울기 변화 = 매수/매도 강도의 전환(Regime Shift).
    # 세 주체(개인, 기관, 외국인)의 괴리가 클수록 Divergence 시그널이 강해진다.
    # 색상: 사용자 툴과 동일하게 파랑=개인, 빨강/주황=기관, 초록=외국인
    fig.add_trace(go.Scatter(
        x=dates, y=df["개인_누적"],
        mode="lines", name="개인",
        line=dict(color="#1976d2", width=1.8),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df["기관_누적"],
        mode="lines", name="기관",
        line=dict(color="#e65100", width=2.0),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df["외국인_누적"],
        mode="lines", name="외국인",
        line=dict(color="#2e7d32", width=1.8),
    ), row=2, col=1)

    # ══════════════════════════════════════════════════════════
    # Regime Shift 수직선 (상하단 모두)
    # ══════════════════════════════════════════════════════════
    # Interrupted Time Series 관점에서, 특정 시점 전후의 기울기 변화를
    # 시각적으로 대조하기 위한 참조선이다.
    regime_ts = pd.Timestamp(regime_date)

    # xref 매핑: R1C1 → x, R2C1(colspan) → x3
    for xref in ["x", "x3"]:
        fig.add_shape(
            type="line",
            x0=regime_ts, x1=regime_ts,
            y0=0, y1=1,
            yref="paper",
            xref=xref,
            line=dict(dash="dot", color="rgba(0, 0, 0, 0.5)", width=1.2),
        )

    # ══════════════════════════════════════════════════════════
    # 레이아웃
    # ══════════════════════════════════════════════════════════
    fig.update_layout(
        title=dict(
            text=f"{name} ({ticker}) — Supply-Demand Divergence",
            font=dict(size=15),
            x=0.01, xanchor="left",
        ),
        template="plotly_white",
        height=800,
        width=1100,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.02,
            xanchor="left", x=0.0,
            font=dict(size=10),
        ),
        xaxis=dict(rangeslider=dict(visible=False)),
        margin=dict(t=50, b=40, l=60, r=20),
    )

    # R1C1과 R2 x축 연동 (날짜) — 줌/팬 동기화
    fig.update_xaxes(matches="x", tickformat="%b %Y", row=2, col=1)
    fig.update_xaxes(tickformat="%b %Y", row=1, col=1)

    # 하단 차트 y축 라벨
    fig.update_yaxes(title_text="누적 순매수량", title_font_size=10, row=2, col=1)

    return fig


# ============================================================================
# 5. MAIN — 실행 진입점
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KOSDAQ 150 Supply-Demand Divergence Analysis")
    print("=" * 60)

    # ── Step 1: 데이터 수집 ──
    data_dict, name_map = fetch_data(start="20251101")

    if not data_dict:
        print("[ERROR] 데이터를 수집하지 못했습니다. 네트워크를 확인하세요.")
        exit(1)

    # ── Step 2: 스코어링 ──
    scores = compute_scores(data_dict, name_map)

    # ── Step 3: Scatter Plot — 전체 종목 분포 ──
    fig_scatter = plot_scatter(scores)
    fig_scatter.show()

    # ── Step 4: 개별 종목 대시보드 ──
    # Divergence 후보 선정: inst_ratio가 가장 높으면서 price_return이 음(-)인 종목
    # → "가격은 빠졌지만 기관은 꾸준히 매수한" 잠재적 매집 종목
    divergence_candidates = scores[scores["price_return"] < 0]

    if not divergence_candidates.empty:
        top_candidate = divergence_candidates.sort_values(
            "inst_ratio", ascending=False
        ).iloc[0]
    else:
        top_candidate = scores.sort_values("inst_ratio", ascending=False).iloc[0]

    top_ticker = top_candidate["종목코드"]
    print(f"\n[PICK] 대시보드 대상: {top_candidate['종목명']} ({top_ticker})")
    print(f"       Price Return = {top_candidate['price_return']:.2%}")
    print(f"       Inst Ratio   = {top_candidate['inst_ratio']:.4f}")

    fig_dashboard = plot_dashboard(top_ticker, data_dict, name_map)
    fig_dashboard.show()

    # ── Interactive Mode ──
    print("\n" + "=" * 60)
    print("📌 Interactive Mode")
    print("=" * 60)
    print("데이터가 메모리에 로드되어 있습니다.")
    print("특정 종목 대시보드를 보려면 아래 코드를 실행하세요:")
    print()
    print("  fig = plot_dashboard('종목코드', data_dict, name_map)")
    print("  fig.show()")
    print()
    print("예시:")
    print("  fig = plot_dashboard('028300', data_dict, name_map)  # HLB")
    print("  fig = plot_dashboard('247540', data_dict, name_map)  # 에코프로비엠")
    print()
    print("스코어 상위 종목 확인:")
    print(scores.sort_values("inst_ratio", ascending=False).head(10).to_string(index=False))
    print("\n[DONE] 분석 완료. 브라우저에서 차트를 확인하세요.")
