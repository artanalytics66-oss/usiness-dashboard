import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import hashlib
from io import BytesIO
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ВАЖНО: ЭТА КОМАНДА ДОЛЖНА БЫТЬ САМОЙ ПЕРВОЙ
st.set_page_config(
    page_title="Панель управления",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== КОНФИГУРАЦИЯ ====================

MASTER_PASSWORD = "панель123"  # Пароль для входа


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
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="блок-входа"><div class="заголовок">🔒 Вход в систему</div>', unsafe_allow_html=True)
        пароль = st.text_input("Введите пароль доступа", type="password")
        
        if st.button("Войти в систему", use_container_width=True, type="primary"):
            if пароль == MASTER_PASSWORD:
                st.session_state.авторизован = True
                st.rerun()
            else:
                st.error("Неверный пароль")
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ==================== ГЕНЕРАЦИЯ И ЗАГРУЗКА ДАННЫХ ====================

@st.cache_data
def создать_шаблон() -> bytes:
    """Создает пример файла Excel для скачивания клиентом"""
    df_template = pd.DataFrame({
        "Дата": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Клиент": ["ООО Пример", "ИП Иванов", "ЗАО Стройка"],
        "Категория": ["Продукты", "Услуги", "Материалы"],
        "Сумма": [50000, 30000, 150000],
        "План": [45000, 35000, 140000],
        "Маржа": [25, 40, 15]
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Данные')
    return output.getvalue()

@st.cache_data
def загрузить_данные(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            return pd.DataFrame()
    else:
        # Генерация демо-данных
        даты = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
        категории = ["Продукты", "Электроника", "Одежда", "Услуги", "Логистика"]
        клиенты = [f"Клиент {i}" for i in range(1, 21)]
        
        data = []
        for дата in даты:
            n_orders = np.random.randint(1, 5)
            for _ in range(n_orders):
                сумма = np.random.normal(50000, 15000)
                план = сумма * np.random.normal(1.05, 0.1)
                data.append({
                    "Дата": дата,
                    "Клиент": np.random.choice(клиенты),
                    "Категория": np.random.choice(категории),
                    "Сумма": abs(сумма),
                    "План": abs(план),
                    "Маржа": np.random.uniform(10, 45)
                })
        df = pd.DataFrame(data)

    # === ИСПРАВЛЕНИЕ ДАТЫ (УБИРАЕМ ВРЕМЯ) ===
    # Преобразуем в datetime, а затем берем только .date (ГГГГ-ММ-ДД)
    df["Дата"] = pd.to_datetime(df["Дата"]).dt.date
    return df


def рассчитать_показатели(df):
    if df.empty:
        return None
    
    всего_доход = df["Сумма"].sum()
    всего_план = df["План"].sum()
    ср_маржа = df["Маржа"].mean()
    кол_заказов = len(df)
    
    выполнение_плана = (всего_доход / всего_план * 100) if всего_план > 0 else 0
    
    # Расчет риска (простая логика для демо)
    фактор_маржа = max(0, (30 - ср_маржа)) # Если маржа ниже 30%, риск растет
    фактор_план = max(0, (100 - выполнение_плана)) # Если план не выполнен
    # Исправлена опечатка (русская Ф)
    индекс_риска = float(np.clip(фактор_маржа * 1.2 + фактор_план * 0.8, 0, 100))
    
    return {
        "Доход": всего_доход,
        "План": всего_план,
        "Маржа": ср_маржа,
        "Риск": индекс_риска,
        "Заказов": кол_заказов,
        "Выполнение": выполнение_плана
    }

def прогноз_на_3_месяца(df):
    if df.empty:
        return None
        
    df_m = df.copy()
    # Для группировки конвертируем дату обратно в datetime
    df_m["Дата"] = pd.to_datetime(df_m["Дата"])
    daily = df_m.groupby("Дата")["Сумма"].sum().reset_index()
    
    daily["DayNum"] = (daily["Дата"] - daily["Дата"].min()).dt.days
    
    X = daily[["DayNum"]]
    y = daily["Сумма"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_day = daily["DayNum"].max()
    future_days = np.array([last_day + i for i in range(1, 91)]).reshape(-1, 1)
    future_pred = model.predict(future_days)
    
    future_dates = [daily["Дата"].max() + timedelta(days=i) for i in range(1, 91)]
    
    # Возвращаем даты как date objects для консистентности
    return pd.DataFrame({
        "Дата": [d.date() for d in future_dates],
        "Прогноз": future_pred
    })

def abc_анализ(df):
    if df.empty:
        return None
    agg = df.groupby("Клиент")["Сумма"].sum().sort_values(ascending=False).reset_index()
    agg["CumSum"] = agg["Сумма"].cumsum()
    agg["Share"] = agg["CumSum"] / agg["Сумма"].sum()
    
    def get_group(x):
        if x <= 0.8: return "A"
        elif x <= 0.95: return "B"
        return "C"
        
    agg["Group"] = agg["Share"].apply(get_group)
    return agg

def сравнение_год_к_году(df):
    if df.empty:
        return None
    
    df["Year"] = pd.to_datetime(df["Дата"]).dt.year
    pivot = df.pivot_table(index="Year", values="Сумма", aggfunc="sum")
    return pivot

def экспорт_в_excel(df, metrics, forecast):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Исходные данные', index=False)
        if forecast is not None:
            forecast.to_excel(writer, sheet_name='Прогноз', index=False)
            
        # Лист с показателями
        if metrics:
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='KPI', index=False)
            
    return output

# ==================== ОСНОВНОЙ ИНТЕРФЕЙС ====================

def main():
    проверка_пароля()
    
    # CSS стили
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #0f1116;
        }
        .metric-label {
            font-size: 14px;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🎛 Управление")
        
        # 1. Загрузка данных
        uploaded_file = st.file_uploader("Загрузить CSV/Excel", type=["csv", "xlsx"])
        
        # 2. Кнопка скачивания шаблона
        st.markdown("---")
        st.markdown("**Нет файла? Скачайте образец:**")
        template_bytes = создать_шаблон()
        st.download_button(
            label="📄 Скачать шаблон Excel",
            data=template_bytes,
            file_name="шаблон_данных.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Заполните этот файл своими данными и загрузите его выше"
        )
        st.markdown("---")

        # 3. Фильтры
        st.subheader("Фильтры")
        
        df_raw = загрузить_данные(uploaded_file)
        if df_raw.empty:
            st.warning("Нет данных для отображения")
            return

        min_date = df_raw["Дата"].min()
        max_date = df_raw["Дата"].max()
        
        # Конвертируем в date для слайдера, если вдруг там datetime
        if isinstance(min_date, datetime): min_date = min_date.date()
        if isinstance(max_date, datetime): max_date = max_date.date()

        date_range = st.date_input(
            "Период",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Обработка выбора одной даты или диапазона
        if isinstance(date_range, tuple):
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = date_range[0]
                end_date = date_range[0]
        else:
            start_date = date_range
            end_date = date_range

        # Список регионов/категорий
        if "Категория" in df_raw.columns:
            cats = df_raw["Категория"].unique().tolist()
            selected_cats = st.multiselect("Категория", cats, default=cats)
        else:
            selected_cats = []

    # --- ФИЛЬТРАЦИЯ ---
    mask = (
        (df_raw["Дата"] >= start_date) & 
        (df_raw["Дата"] <= end_date) &
        (df_raw["Категория"].isin(selected_cats) if selected_cats else True)
    )
    df_f = df_raw[mask]

    # --- ГЛАВНЫЙ ЭКРАН ---
    st.title("📊 Панель управления бизнесом")
    
    if df_f.empty:
        st.info("Выберите другие параметры фильтрации")
        return

    # Расчеты
    metrics = рассчитать_показатели(df_f)
    forecast = прогноз_на_3_месяца(df_f)
    abc = abc_анализ(df_f)
    
    # 1. KPI РЯД
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("Выручка", f"{metrics['Доход']:,.0f} ₽", f"{metrics['Выполнение']-100:.1f}% план")
    c2.metric("Маржа", f"{metrics['Маржа']:.1f}%", f"{metrics['Маржа']-20:.1f}%")
    c3.metric("Заказов", metrics['Заказов'])
    c4.metric("Вып. плана", f"{metrics['Выполнение']:.1f}%")
    
    delta_risk = 100 - metrics['Риск']
    c5.metric("Индекс риска", f"{metrics['Риск']:.0f}/100", f"Safe: {delta_risk:.0f}", delta_color="off")

    st.markdown("---")

    # 2. ГРАФИКИ
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("Динамика доходов")
        # Агрегация по дням для графика
        daily_chart = df_f.groupby("Дата")[["Сумма", "План"]].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_chart["Дата"], y=daily_chart["Сумма"], name="Факт", fill='tozeroy'))
        fig.add_trace(go.Scatter(x=daily_chart["Дата"], y=daily_chart["План"], name="План", line=dict(dash='dot')))
        
        if forecast is not None:
             fig.add_trace(go.Scatter(x=forecast["Дата"], y=forecast["Прогноз"], name="Прогноз", line=dict(color='green')))
             
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_g2:
        st.subheader("Структура (ABC)")
        if abc is not None:
            abc_count = abc["Group"].value_counts()
            fig_abc = go.Figure(data=[go.Pie(labels=abc_count.index, values=abc_count.values, hole=.4)])
            fig_abc.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_abc, use_container_width=True)

    # 3. ТАБЛИЦА ДЕТАЛЕЙ
    st.subheader("Детальные данные")
    
    # Стилизация таблицы (убираем лишние нули и форматируем дату)
    st.dataframe(
        df_f.sort_values("Дата", ascending=False),
        column_config={
            "Дата": st.column_config.DateColumn("Дата", format="DD.MM.YYYY"),
            "Сумма": st.column_config.NumberColumn("Сумма", format="%d ₽"),
            "План": st.column_config.NumberColumn("План", format="%d ₽"),
            "Маржа": st.column_config.NumberColumn("Маржа", format="%.1f %%"),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Кнопка скачивания отчета
    excel_data = экспорт_в_excel(df_f, metrics, forecast)
    st.download_button(
        "📥 Скачать отчет (Excel)",
        data=excel_data,
        file_name="business_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
