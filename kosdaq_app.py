import streamlit as st
import pandas as pd
from datetime import datetime
from kosdaq150_divergence import fetch_data, compute_scores, plot_scatter, plot_dashboard

st.set_page_config(layout="wide", page_title="KOSDAQ 150 Divergence Dashboard")

st.title("📊 KOSDAQ 150 수급 다이버전스 분석")
st.markdown("""
이 대시보드는 KOSDAQ 150 종목을 대상으로 기관 순매수 비율(Inst Ratio)과 가격 변동률(Price Return)을 비교하여, 
**'가격은 하락하는데 기관은 매집하는'** (잠재적 매집) 다이버전스 종목을 탐색합니다.
""")

# --- 데이터 로드 (Caching) ---
# --- 데이터 로드 (Caching) ---
# 캐시 키를 바꾸기 위해 함수명 변경 (강제 캐시 무효화 효과)
@st.cache_data(ttl=3600*2, show_spinner=False)  
def get_divergence_data(start_date_str):
    from kosdaq150_divergence import get_kosdaq150_tickers
    from pykrx import stock
    from datetime import datetime
    
    end = datetime.now().strftime("%Y%m%d")
    start = start_date_str
    
    tickers = get_kosdaq150_tickers()
    
    data_dict = {}
    name_map = {}
    inv_fail_count = 0
    
    # Streamlit UI 요소 생성
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_log = st.empty()
    
    total = len(tickers)
    errors = []
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"[{i+1}/{total}] 수집 중: {ticker}...")
            
            name = stock.get_market_ticker_name(ticker)
            if not name:
                continue
            name_map[ticker] = name

            # 1. OHLCV
            ohlcv = stock.get_market_ohlcv_by_date(start, end, ticker)
            if ohlcv.empty or len(ohlcv) < 5:
                continue

            # 2. 투자자별 순매수 (주말엔 실패함)
            inv = stock.get_market_trading_volume_by_date(start, end, ticker)
            
            if inv.empty or len(inv) == 0:
                inv_fail_count += 1
                ohlcv["기관합계"] = 0
                ohlcv["개인"] = 0
                ohlcv["외국인합계"] = 0
                merged = ohlcv
            else:
                for col_name in ["기관합계", "개인", "외국인합계"]:
                    if col_name not in inv.columns:
                        inv[col_name] = 0
                merged = ohlcv.join(inv[["기관합계", "개인", "외국인합계"]], how="left")
                merged[["기관합계", "개인", "외국인합계"]] = (
                    merged[["기관합계", "개인", "외국인합계"]].fillna(0)
                )

            # 3. 누적 계산
            merged["기관_누적"] = merged["기관합계"].cumsum()
            merged["개인_누적"] = merged["개인"].cumsum()
            merged["외국인_누적"] = merged["외국인합계"].cumsum()

            data_dict[ticker] = merged

        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            continue
            
        # 진행상황 게이지 업데이트
        progress_bar.progress((i + 1) / total)
        
    status_text.text(f"수집 완료: {len(data_dict)}/{total} 종목 성공")
    
    if errors and len(data_dict) == 0:
        error_log.error(f"모든 종목 수집 실패. 대표 에러: {errors[0]}")
    elif len(data_dict) == 0:
        error_log.error("수집된 종목이 0개입니다. (서버/네트워크 혹은 시간대 문제)")
        
    if not data_dict:
        raise RuntimeError("KRX API에서 데이터를 수집하지 못했습니다. (0개 종목)")
        
    scores = compute_scores(data_dict, name_map)
    return data_dict, name_map, scores

with st.sidebar:
    st.header("⚙️ 설정 / 도구")
    
    # 분석 시작일 선택기 추가
    target_start_date = st.date_input(
        "📅 분석 시작일", 
        value=datetime(2025, 11, 1), 
        max_value=datetime.today()
    )
    start_date_str = target_start_date.strftime("%Y%m%d")
    
    st.markdown("---")
    if st.button("🔄 캐시 지우기 및 다시 수집"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("KRX 데이터를 수집 중입니다... (최대 1~3분 소요)"):
    try:
        data_dict, name_map, scores_df = get_divergence_data(start_date_str)
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        st.stop()

if scores_df.empty:
    st.warning("분석할 데이터가 없습니다.")
    st.stop()

# 점수가 높은 순(기관 순매수 비율 상위)으로 정렬
scores_sorted = scores_df.sort_values("inst_ratio", ascending=False).reset_index(drop=True)

# 화면 분할: 좌측은 Scatter/Table, 우측은 개별 Dashboard
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1. Divergence Map (전체 종목 분포)")
    
    fig_scatter = plot_scatter(scores_sorted)
    # 잘못 눌렀을 때의 확대를 막기 위해 기본 드래그 모드를 'pan(이동)'으로 제한
    fig_scatter.update_layout(dragmode="pan", height=450, width=None, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': True})

    st.subheader("2. 종목 순위 및 검색")
    
    # 검색 기능
    search_query = st.text_input("🔍 종목명 또는 종목코드로 검색하세요:", "").strip()
    
    st.markdown("👇 표에서 종목명 옆의 체크박스를 클릭하면 우측에 대시보드가 나타납니다.")
    
    # 데이터 포맷팅
    display_df = scores_sorted.copy()
    display_df["price_return"] = display_df["price_return"].apply(lambda x: f"{x:.2%}")
    display_df["inst_ratio"] = display_df["inst_ratio"].apply(lambda x: f"{x:.4f}")
    
    # 검색어 필터링 적용
    if search_query:
        mask = display_df["종목명"].str.contains(search_query, case=False, na=False) | \
               display_df["종목코드"].str.contains(search_query, case=False, na=False)
        filtered_df = display_df[mask].reset_index(drop=True)
    else:
        filtered_df = display_df

    # DataFrame Selection 기능 사용
    event = st.dataframe(
        filtered_df[["종목코드", "종목명", "price_return", "inst_ratio"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=350,
    )

with col_right:
    st.subheader("3. 선택 종목 상세 대시보드")
    
    col_dash1, col_dash2 = st.columns([1, 1])
    with col_dash1:
        st.markdown("💡 차트 더블 클릭시 화면 줌이 **초기화**됩니다.")
    with col_dash2:
        # 이 버튼을 누르면 Streamlit이 화면을 아예 새로고침하여 줌을 리셋합니다
        if st.button("차트 크기 원래대로 (Reset)", use_container_width=True):
            pass # rerun automatically happens on button click
    
    # 이벤트에서 선택된 행 가져오기
    selected_rows = event.selection.rows
    
    if len(selected_rows) > 0:
        # 필터링된 데이터프레임 기준으로 인덱스 접근
        selected_idx = selected_rows[0]
        selected_ticker = filtered_df.iloc[selected_idx]["종목코드"]
    else:
        # 기본값: inst_ratio 1위 종목 중 가격이 마이너스인 핵심 후보
        divergence_candidates = scores_sorted[scores_sorted["price_return"] < 0]
        if not divergence_candidates.empty:
            selected_ticker = divergence_candidates.iloc[0]["종목코드"]
        else:
            selected_ticker = scores_sorted.iloc[0]["종목코드"]

    target_name = name_map.get(selected_ticker, selected_ticker)
    
    with st.spinner(f"{target_name} 렌더링 중..."):
        fig_dashboard = plot_dashboard(selected_ticker, data_dict, name_map)
        
        # 잘못된 확대 스크롤/드래그 방지를 위해 기본 dragmode를 pan으로 변경
        fig_dashboard.update_layout(
            dragmode="pan",
            height=850, 
            width=None,
            margin=dict(t=50, b=30, l=0, r=0)
        )
        st.plotly_chart(fig_dashboard, use_container_width=True, config={'displayModeBar': True})
