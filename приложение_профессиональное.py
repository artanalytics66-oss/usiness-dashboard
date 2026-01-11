import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
from io import BytesIO
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ВАЖНО: ДОЛЖНО БЫТЬ ПЕРВОЙ streamlit-командой и ВЫЗЫВАТЬСЯ 1 РАЗ
st.set_page_config(
    page_title="Панель управления",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== КОНФИГУРАЦИЯ И БЕЗОПАСНОСТЬ ====================

MASTER_PASSWORD = "панель123"  # ИЗМЕНИТЕ НА СВОЙ ПАРОЛЬ!


def _хеш_пароля(пароль: str) -> str:
    return hashlib.sha256(пароль.encode("utf-8")).hexdigest()


def проверка_пароля() -> None:
    if "авторизован" not in st.session_state:
        st.session_state.авторизован = False

    if st.session_state.авторизован:
        return

    st.markdown(
        """
        <style>
            .блок-входа{
                max-width: 420px;
                margin: 10vh auto 0 auto;
                padding: 24px;
                border-radius: 14px;
                background: #151b24;
                border: 1px solid #2a3038;
                box-shadow: 0 10px 30px rgba(0,0,0,.35);
            }
            .заголовок{
                font-size: 26px;
                font-weight: 800;
                color: #fff;
                margin-bottom: 8px;
                text-align:center;
            }
            .подзаголовок{
                color:#8a92a0;
                text-align:center;
                margin-bottom: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="блок-входа">', unsafe_allow_html=True)
    st.markdown('<div class="заголовок">Панель управления бизнесом</div>', unsafe_allow_html=True)
    st.markdown('<div class="подзаголовок">Вход по паролю</div>', unsafe_allow_html=True)

    пароль = st.text_input("Пароль", type="password")
    if st.button("Войти", use_container_width=True):
        if _хеш_пароля(пароль) == _хеш_пароля(MASTER_PASSWORD):
            st.session_state.авторизован = True
            st.success("Вход выполнен")
            st.rerun()
        else:
            st.error("Неверный пароль")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ==================== СТИЛИ ====================

def применить_стили() -> None:
    st.markdown(
        """
        <style>
            body { background-color: #0f1419; color: #e0e0e0; }
            .metric-card{
                background: linear-gradient(135deg, #1a1f29 0%, #252d3a 100%);
                border: 1px solid #2a3038;
                border-radius: 12px;
                padding: 18px 18px 14px 18px;
                margin-bottom: 14px;
                box-shadow: 0 6px 16px rgba(0,0,0,.25);
            }
            .metric-title{
                color: #8a92a0;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .6px;
                margin-bottom: 10px;
            }
            .metric-value{
                color: #ffffff;
                font-size: 38px;
                font-weight: 800;
                line-height: 1.0;
                margin-bottom: 6px;
            }
            .metric-change{
                font-size: 13px;
                font-weight: 600;
            }
            .positive{ color: #10b981; }
            .negative{ color: #ef4444; }
            .neutral{  color: #8a92a0; }
            .warning{  color: #f59e0b; }

            .box{
                background: #151b24;
                border: 1px solid #2a3038;
                border-radius: 12px;
                padding: 14px 16px;
            }
            .alert{
                padding: 12px 14px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid;
                background: #151b24;
                border-top: 1px solid #2a3038;
                border-right: 1px solid #2a3038;
                border-bottom: 1px solid #2a3038;
            }
            .alert-danger{ border-left-color:#ef4444; }
            .alert-warn{   border-left-color:#f59e0b; }
            .alert-ok{     border-left-color:#10b981; }

            .hint{ color:#8a92a0; font-size:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================== ДАННЫЕ ====================

def _нормализовать_маржу(значение) -> float:
    try:
        v = float(значение)
    except Exception:
        return np.nan
    # Если маржа пришла как 25 (то есть 25%), приводим к 0.25
    if v > 1.0:
        v = v / 100.0
    return v


@st.cache_data
def демо_данные() -> pd.DataFrame:
    np.random.seed(42)
    месяцы = pd.date_range(start="2024-01-01", periods=24, freq="MS")

    строки = []
    план_база = 1_000_000

    регионы = ["Москва", "Санкт‑Петербург", "Регионы", "Интернет"]
    категории = ["Продукты", "Напитки", "Молочное", "Прочее"]

    for i, месяц in enumerate(месяцы):
        сезонность = 1.0 + 0.28 * np.sin(2 * np.pi * i / 12)
        план = план_база * сезонность * np.random.uniform(0.95, 1.06)
        факт = план * np.random.uniform(0.84, 1.10)

        база_маржа = 0.35 if i < 12 else 0.32
        маржа = база_маржа + np.random.uniform(-0.02, 0.02)
        маржа = float(np.clip(маржа, 0.20, 0.45))

        заказы = int(max(1, факт / 55_000 + np.random.randint(-8, 18)))
        средний_чек = факт / max(1, заказы)

        строки.append(
            {
                "Дата": месяц,
                "План": float(план),
                "Факт": float(факт),
                "Маржа": float(маржа),
                "Заказы": int(заказы),
                "Регион": str(np.random.choice(регионы)),
                "Категория": str(np.random.choice(категории)),
                "Клиент": f"Клиент_{int(np.random.randint(1, 60))}",
                "Средний_чек": float(средний_чек),
            }
        )

    return pd.DataFrame(строки)


def загрузить_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)

    # Приведение колонок (если пользователь назвал похоже)
    # Минимально нужны: Дата, План, Факт, Маржа, Заказы
    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    else:
        raise ValueError("В CSV нет колонки 'Дата'.")

    for col in ["План", "Факт", "Маржа", "Заказы"]:
        if col not in df.columns:
            raise ValueError(f"В CSV нет колонки '{col}'.")

    df["План"] = pd.to_numeric(df["План"], errors="coerce")
    df["Факт"] = pd.to_numeric(df["Факт"], errors="coerce")
    df["Заказы"] = pd.to_numeric(df["Заказы"], errors="coerce")

    df["Маржа"] = df["Маржа"].apply(_нормализовать_маржу)

    # Опциональные
    if "Регион" not in df.columns:
        df["Регион"] = "Не указан"
    if "Категория" not in df.columns:
        df["Категория"] = "Не указана"
    if "Клиент" not in df.columns:
        df["Клиент"] = "Не указан"

    df = df.dropna(subset=["Дата", "План", "Факт", "Маржа", "Заказы"]).sort_values("Дата")
    return df


# ==================== АНАЛИТИКА ====================

def рассчитать_показатели(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "доход_текущий": 0.0,
            "доход_изменение": 0.0,
            "маржа": 0.0,
            "маржа_снижается": False,
            "выполнение_плана": 0.0,
            "заказы": 0,
            "заказы_изменение": 0,
            "индекс_риска": 0.0,
        }

    df = df.sort_values("Дата")
    текущий = df.iloc[-1]
    предыдущий = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    доход_текущий = float(текущий["Факт"])
    доход_предыдущий = float(предыдущий["Факт"]) if float(предыдущий["Факт"]) != 0 else np.nan

    изменение = 0.0
    if not np.isnan(доход_предыдущий) and доход_предыдущий != 0:
        изменение = (доход_текущий - доход_предыдущий) / доход_предыдущий * 100

    маржа_тек = float(текущий["Маржа"]) * 100
    маржа_пред = float(предыдущий["Маржа"]) * 100
    маржа_снижается = маржа_тек < маржа_пред

    выполнение = 0.0
    if float(текущий["План"]) != 0:
        выполнение = float(текущий["Факт"]) / float(текущий["План"]) * 100

    заказы = int(текущий["Заказы"])
    заказы_пред = int(предыдущий["Заказы"])
    заказы_изм = заказы - заказы_пред

    # Индекс риска: недовыполнение + падение маржи (просто и понятно)
    фактор_маржа = max(0.0, (маржа_пред - маржа_тек))
    фактор_план = max(0.0, 100.0 - выполнение)
    индекс_риска = float(np.clip(фактор_маржа * 1.2 + фактор_план * 0.8, 0, 100))

    return {
        "доход_текущий": доход_текущий,
        "доход_изменение": float(изменение),
        "маржа": float(маржа_тек),
        "маржа_снижается": bool(маржа_снижается),
        "выполнение_плана": float(выполнение),
        "заказы": int(заказы),
        "заказы_изменение": int(заказы_изм),
        "индекс_риска": индекс_риска,
    }


def прогноз_на_3_месяца(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Дата")
    if len(df) < 3:
        return pd.DataFrame({"Дата": [], "Прогноз": []})

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Факт"].values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.array([[len(df)], [len(df) + 1], [len(df) + 2]])
    forecast = model.predict(future_X)

    dates = pd.date_range(start=df.iloc[-1]["Дата"] + pd.offsets.MonthBegin(1), periods=3, freq="MS")

    return pd.DataFrame({"Дата": dates, "Прогноз": forecast})


def abc_анализ(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Клиент", "Доход", "Доля_%", "Категория"])

    клиент_доход = df.groupby("Клиент")["Факт"].sum().sort_values(ascending=False)
    total = float(клиент_доход.sum()) if float(клиент_доход.sum()) != 0 else 1.0

    rows = []
    cumsum = 0.0
    for клиент, доход in клиент_доход.items():
        cumsum += float(доход)
        pct_cum = cumsum / total * 100

        if pct_cum <= 80:
            cat = "A"
        elif pct_cum <= 95:
            cat = "B"
        else:
            cat = "C"

        rows.append(
            {
                "Клиент": клиент,
                "Доход": float(доход),
                "Доля_%": float(доход) / total * 100,
                "Категория": cat,
            }
        )

    return pd.DataFrame(rows)


def сравнение_год_к_году(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"текущий_доход": 0.0, "прошлый_доход": 0.0, "pct": 0.0}

    max_date = df["Дата"].max()
    год_назад = max_date - timedelta(days=365)

    текущий = df[df["Дата"] > год_назад]
    прошлый = df[df["Дата"] <= год_назад]

    тек = float(текущий["Факт"].sum())
    прош = float(прошлый["Факт"].sum()) if len(прошлый) else 0.0

    pct = 0.0
    if прош != 0:
        pct = (тек - прош) / прош * 100

    return {"текущий_доход": тек, "прошлый_доход": прош, "pct": pct}


# ==================== ЭКСПОРТ ====================

def экспорт_в_excel(df: pd.DataFrame, показатели: dict, прогноз: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(
            {
                "Показатель": [
                    "Доход (последний месяц)",
                    "Изменение к предыдущему",
                    "Маржа (последний месяц)",
                    "Выполнение плана",
                    "Индекс риска",
                ],
                "Значение": [
                    f"{показатели['доход_текущий']:,.0f}",
                    f"{показатели['доход_изменение']:+.1f}%",
                    f"{показатели['маржа']:.1f}%",
                    f"{показатели['выполнение_плана']:.0f}%",
                    f"{показатели['индекс_риска']:.0f}/100",
                ],
            }
        )
        summary_df.to_excel(writer, sheet_name="Показатели", index=False)
        df.to_excel(writer, sheet_name="Данные", index=False)
        прогноз.to_excel(writer, sheet_name="Прогноз", index=False)

    output.seek(0)
    return output


# ==================== ПРИЛОЖЕНИЕ ====================

def main() -> None:
    проверка_пароля()
    применить_стили()

    if "df" not in st.session_state:
        st.session_state.df = демо_данные()

    df = st.session_state.df.copy()

    with st.sidebar:
        st.markdown("### Управление")
        st.markdown("**Данные**")
        uploaded = st.file_uploader("Загрузить CSV", type="csv")
        if uploaded is not None:
            try:
                st.session_state.df = загрузить_csv(uploaded)
                st.success("Данные загружены")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка CSV: {e}")

        st.markdown("---")
        st.markdown("**Фильтры**")

        период = st.selectbox("Период", ["Все", "Последний год", "Последний квартал", "Последний месяц"])

        регионы = sorted(df["Регион"].astype(str).unique().tolist())
        категории = sorted(df["Категория"].astype(str).unique().tolist())

        выбранные_регионы = st.multiselect("Регион", регионы, default=регионы)
        выбранные_категории = st.multiselect("Категория", категории, default=категории)

        st.markdown("---")
        st.markdown("**Экспорт** (по отфильтрованным данным)")
        st.caption("Excel формируется из текущего набора данных.")

        if st.button("Выйти", use_container_width=True):
            st.session_state.авторизован = False
            st.rerun()

    # Применяем фильтры
    df_f = df.copy().sort_values("Дата")

    if период == "Последний год":
        df_f = df_f[df_f["Дата"] >= df_f["Дата"].max() - timedelta(days=365)]
    elif период == "Последний квартал":
        df_f = df_f[df_f["Дата"] >= df_f["Дата"].max() - timedelta(days=90)]
    elif период == "Последний месяц":
        df_f = df_f[df_f["Дата"] >= df_f["Дата"].max() - timedelta(days=31)]

    df_f = df_f[df_f["Регион"].isin(выбранные_регионы) & df_f["Категория"].isin(выбранные_категории)]

    # Аналитика по отфильтрованным данным
    показатели = рассчитать_показатели(df_f)
    прогноз = прогноз_на_3_месяца(df_f)
    abc = abc_анализ(df_f)
    yoy = сравнение_год_к_году(df_f)

    # Экспорт кнопка (после расчётов)
    with st.sidebar:
        excel_bytes = экспорт_в_excel(df_f, показатели, прогноз)
        st.download_button(
            "Скачать Excel",
            data=excel_bytes.getvalue(),
            file_name=f"отчет_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Заголовок
    left, right = st.columns([3, 1])
    with left:
        st.markdown("# Панель управления")
        st.markdown('<span class="hint">Профессиональная аналитика бизнеса</span>', unsafe_allow_html=True)
    with right:
        st.markdown(f"**Обновлено:** {datetime.now().strftime('%d.%m.%Y')}")

    st.markdown("---")

    # 6 KPI
    st.markdown("### Ключевые показатели")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    def _карточка(title, value, change_text, cls):
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-change {cls}">{change_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c1:
        cls = "positive" if показатели["доход_изменение"] >= 0 else "negative"
        стрелка = "↑" if показатели["доход_изменение"] >= 0 else "↓"
        _карточка(
            "Доход",
            f"{показатели['доход_текущий']/1_000_000:.2f}М",
            f"{стрелка} {abs(показатели['доход_изменение']):.1f}%",
            cls,
        )

    with c2:
        cls = "warning" if показатели["маржа_снижается"] else "positive"
        подпись = "Маржа снижается" if показатели["маржа_снижается"] else "Маржа стабильна"
        _карточка("Маржа", f"{показатели['маржа']:.1f}%", подпись, cls)

    with c3:
        perf = показатели["выполнение_плана"]
        cls = "positive" if perf >= 95 else "warning" if perf >= 85 else "negative"
        _карточка("Выполнение плана", f"{perf:.0f}%", "Факт / План", cls)

    with c4:
        risk = показатели["индекс_риска"]
        cls = "positive" if risk < 30 else "warning" if risk < 60 else "negative"
        _карточка("Индекс риска", f"{risk:.0f}", "Шкала 0–100", cls)

    with c5:
        cls = "positive" if показатели["заказы_изменение"] >= 0 else "negative"
        стрелка = "↑" if показатели["заказы_изменение"] >= 0 else "↓"
        _карточка("Объём заказов", f"{показатели['заказы']}", f"{стрелка} {abs(показатели['заказы_изменение'])}", cls)

    with c6:
        cls = "positive" if yoy["pct"] >= 0 else "negative"
        _карточка("Год к году", f"{yoy['pct']:+.1f}%", "Доход за 12 месяцев", cls)

    # План vs Факт
    st.markdown("### План vs Факт")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_f["Дата"],
            y=df_f["Факт"],
            mode="lines",
            name="Факт",
            line=dict(color="#10b981", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_f["Дата"],
            y=df_f["План"],
            mode="lines",
            name="План",
            line=dict(color="#8a92a0", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=380,
        paper_bgcolor="#0f1419",
        plot_bgcolor="#151b24",
        font=dict(color="#e0e0e0", size=12),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#2a3038"),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Прогноз
    st.markdown("### Прогноз на 3 месяца")
    if прогноз.empty:
        st.info("Недостаточно данных для прогноза (нужно хотя бы 3 точки).")
    else:
        st.markdown(
            f"""
            <div class="box">
                <div><b>Оценка тренда (по отфильтрованным данным):</b></div>
                <div style="margin-top:8px;">
                    {прогноз.iloc[0]['Прогноз']/1_000_000:.2f}М →
                    {прогноз.iloc[1]['Прогноз']/1_000_000:.2f}М →
                    {прогноз.iloc[2]['Прогноз']/1_000_000:.2f}М
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ABC
    st.markdown("### ABC-анализ клиентов")
    if abc.empty:
        st.info("Нет данных для ABC-анализа.")
    else:
        a = abc[abc["Категория"] == "A"]
        b = abc[abc["Категория"] == "B"]
        c = abc[abc["Категория"] == "C"]

        k1, k2, k3 = st.columns(3)
        k1.metric("Категория A", f"{len(a)}", f"{a['Доля_%'].sum():.1f}% дохода")
        k2.metric("Категория B", f"{len(b)}", f"{b['Доля_%'].sum():.1f}% дохода")
        k3.metric("Категория C", f"{len(c)}", f"{c['Доля_%'].sum():.1f}% дохода")

        st.dataframe(abc.head(20), use_container_width=True, hide_index=True)

    # Детали
    st.markdown("### Подробные данные")
    cols_to_show = ["Дата", "Регион", "Категория", "Клиент", "Факт", "План", "Заказы", "Маржа"]
    show = df_f[cols_to_show].copy()
    show["Маржа"] = (show["Маржа"] * 100).round(1)
    st.dataframe(show.sort_values("Дата", ascending=False), use_container_width=True, hide_index=True)

    # Алерты
    st.markdown("### Алерты")
    risk = показатели["индекс_риска"]
    perf = показатели["выполнение_плана"]
    margin_drop = показатели["маржа_снижается"]

    if risk >= 60:
        st.markdown(
            f'<div class="alert alert-danger"><b>Критично:</b> индекс риска {risk:.0f}/100. Нужны действия.</div>',
            unsafe_allow_html=True,
        )
    if perf < 85:
        st.markdown(
            f'<div class="alert alert-warn"><b>Внимание:</b> выполнение плана {perf:.0f}%. Проверьте причины.</div>',
            unsafe_allow_html=True,
        )
    if margin_drop:
        st.markdown(
            f'<div class="alert alert-warn"><b>Внимание:</b> маржа снижается (сейчас {показатели["маржа"]:.1f}%).</div>',
            unsafe_allow_html=True,
        )
    if (risk < 30) and (perf >= 90) and (not margin_drop):
        st.markdown(
            f'<div class="alert alert-ok"><b>Норма:</b> ключевые показатели выглядят стабильно.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Версия 1.0 • Панель управления бизнесом")


if __name__ == "__main__":
    main()
