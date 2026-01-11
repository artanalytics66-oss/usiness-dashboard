import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import json
from io import BytesIO
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ И БЕЗОПАСНОСТЬ ====================

MASTER_PASSWORD = "панель123"  # ИЗМЕНИТЕ НА СВОЙ ПАРОЛЬ!

def hash_password(password):
    """Хеширует пароль"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password():
    """Проверка пароля при входе"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.set_page_config(page_title="Панель управления - Вход", layout="centered")
        st.markdown("""
            <style>
                .login-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("# 🔐 Панель управления бизнесом")
        st.markdown("### Профессиональный инструмент аналитики")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input("Пароль:", type="password", key="password_input")
            
            if st.button("Войти", use_container_width=True):
                if hash_password(password) == hash_password(MASTER_PASSWORD):
                    st.session_state.authenticated = True
                    st.success("✓ Вход выполнен!")
                    st.rerun()
                else:
                    st.error("✗ Неверный пароль")
        
        st.markdown("---")
        st.markdown("""
        ### 📊 О приложении
        
        **Профессиональная панель управления** для анализа бизнеса:
        - Прогнозирование доходов
        - ABC-анализ клиентов
        - Интерактивные фильтры
        - Экспорт отчётов
        - Автоматические алерты
        
        **Цена**: 50 000 руб
        """)
        st.stop()

# ==================== СТИЛИ ====================

def apply_styles():
    """Применяет стили приложения"""
    st.markdown("""
        <style>
        * {
            margin: 0;
            padding: 0;
        }
        
        body {
            background-color: #0f1419;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1a1f29 0%, #252d3a 100%);
            border: 1px solid #2a3038;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .metric-title {
            color: #8a92a0;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        
        .metric-value {
            color: #ffffff;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .metric-change {
            font-size: 14px;
            font-weight: 500;
        }
        
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .neutral { color: #8a92a0; }
        .warning { color: #f59e0b; }
        
        .section-title {
            color: #ffffff;
            font-size: 24px;
            font-weight: 700;
            margin: 32px 0 16px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid #2a3038;
        }
        
        .forecast-box {
            background: #1a2332;
            border-left: 4px solid #10b981;
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
        }
        
        .alert-box {
            background: #2d1f1f;
            border-left: 4px solid #ef4444;
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        }
        
        .alert-box.warning {
            background: #2d2410;
            border-left-color: #f59e0b;
        }
        
        .alert-box.success {
            background: #1f2d23;
            border-left-color: #10b981;
        }
        </style>
    """, unsafe_allow_html=True)

# ==================== ГЕНЕРИРОВАНИЕ И ЗАГРУЗКА ДАННЫХ ====================

@st.cache_data
def load_sample_data():
    """Загружает или генерирует данные"""
    np.random.seed(42)
    месяцы = pd.date_range(start='2023-01-01', periods=24, freq='MS')
    
    данные = []
    план_база = 1_000_000
    
    for i, месяц in enumerate(месяцы):
        сезонность = 1.0 + 0.3 * np.sin(2 * np.pi * i / 12)
        
        план = план_база * сезонность * np.random.uniform(0.95, 1.05)
        факт = план * np.random.uniform(0.85, 1.10)
        
        база_маржа = 0.35 if i < 12 else 0.32
        маржа = база_маржа + np.random.uniform(-0.02, 0.02)
        
        данные.append({
            'Дата': месяц,
            'План': план,
            'Факт': факт,
            'Маржа': max(0.20, min(0.40, маржа)),
            'Заказы': int(факт / 50_000 + np.random.randint(-10, 20)),
            'Регион': np.random.choice(['Москва', 'СПб', 'Регионы', 'Интернет']),
            'Категория': np.random.choice(['Продукты', 'Напитки', 'Молочное', 'Прочее']),
            'Клиент': f"Клиент_{np.random.randint(1, 50)}",
            'Средний_чек': fakkt / max(1, int(факт / 50_000)) if факт > 0 else 0
        })
    
    return pd.DataFrame(данные)

# ==================== РАСЧЁТЫ И АНАЛИТИКА ====================

def рассчитать_показатели(df):
    """Вычисляет все показатели"""
    текущий = df.iloc[-1]
    предыдущий = df.iloc[-2] if len(df) > 1 else df.iloc[0]
    
    результаты = {
        'доход_текущий': текущий['Факт'],
        'доход_изменение': ((текущий['Факт'] - предыдущий['Факт']) / предыдущий['Факт'] * 100) if предыдущий['Факт'] > 0 else 0,
        'маржа': текущий['Маржа'] * 100,
        'маржа_снижается': текущий['Маржа'] < предыдущий['Маржа'],
        'выполнение_плана': (текущий['Факт'] / текущий['План'] * 100) if текущий['План'] > 0 else 0,
        'заказы': текущий['Заказы'],
        'заказы_изменение': текущий['Заказы'] - предыдущий['Заказы'],
    }
    
    # Индекс риска
    фактор_маржа = max(0, (предыдущий['Маржа'] - текущий['Маржа']) / предыдущий['Маржа'] * 100) if предыдущий['Маржа'] > 0 else 0
    фактор_выполнение = max(0, (100 - результаты['выполнение_плана']) / 100 * 100)
    результаты['индекс_риска'] = (фактор_маржа * 0.4 + фактор_выполнение * 0.6)
    
    return результаты

def прогноз_на_3_месяца(df):
    """Прогнозирует доход на 3 месяца"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Факт'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    forecast = model.predict(future_X)
    
    dates = pd.date_range(start=df.iloc[-1]['Дата'] + timedelta(days=32), periods=3, freq='MS')
    
    return pd.DataFrame({
        'Дата': dates,
        'Прогноз': forecast
    })

def abc_анализ(df):
    """ABC анализ клиентов"""
    клиент_доход = df.groupby('Клиент')['Факт'].sum().sort_values(ascending=False)
    total = клиент_доход.sum()
    
    abc = []
    cumsum = 0
    for клиент, доход in клиент_доход.items():
        cumsum += доход
        процент = cumsum / total * 100
        
        if процент <= 80:
            категория = 'A'
        elif процент <= 95:
            категория = 'B'
        else:
            категория = 'C'
        
        abc.append({
            'Клиент': клиент,
            'Доход': доход,
            'Доля_%': доход/total*100,
            'Категория': категория
        })
    
    return pd.DataFrame(abc)

def сравнение_периодов(df):
    """Сравнивает два периода"""
    df_текущий = df[df['Дата'] >= df['Дата'].max() - timedelta(days=365)]
    df_прошлый = df[df['Дата'] < df['Дата'].max() - timedelta(days=365)]
    
    return {
        'текущий_доход': df_текущий['Факт'].sum(),
        'прошлый_доход': df_прошлый['Факт'].sum() if len(df_прошлый) > 0 else 0,
        'текущее_выполнение': (df_текущий['Факт'].sum() / df_текущий['План'].sum() * 100) if df_текущий['План'].sum() > 0 else 0,
        'прошлое_выполнение': (df_прошлый['Факт'].sum() / df_прошлый['План'].sum() * 100) if df_прошлый['План'].sum() > 0 else 0,
    }

# ==================== ЭКСПОРТ ====================

def экспорт_в_excel(df, показатели, прогноз):
    """Экспортирует отчёт в Excel"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Основные показатели
        summary_df = pd.DataFrame({
            'Показатель': ['Доход (текущий месяц)', 'Изменение', 'Маржа', 'Выполнение плана', 'Индекс риска'],
            'Значение': [
                f"{показатели['доход_текущий']:,.0f} руб",
                f"{показатели['доход_изменение']:.1f}%",
                f"{показатели['маржа']:.1f}%",
                f"{показатели['выполнение_плана']:.0f}%",
                f"{показатели['индекс_риска']:.0f}/100"
            ]
        })
        summary_df.to_excel(writer, sheet_name='Показатели', index=False)
        
        # Данные
        df.to_excel(writer, sheet_name='Данные', index=False)
        
        # Прогноз
        прогноз.to_excel(writer, sheet_name='Прогноз', index=False)
    
    output.seek(0)
    return output

def экспорт_в_pdf(показатели):
    """Подготавливает данные для PDF экспорта"""
    pdf_content = f"""
    ПАНЕЛЬ УПРАВЛЕНИЯ БИЗНЕСОМ
    ====================================
    
    Дата отчёта: {datetime.now().strftime('%d.%m.%Y %H:%M')}
    
    КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:
    
    Доход: {показатели['доход_текущий']:,.0f} руб
    Изменение: {показатели['доход_изменение']:+.1f}%
    
    Маржа: {показатели['маржа']:.1f}%
    Выполнение плана: {показатели['выполнение_плана']:.0f}%
    
    Индекс риска: {показатели['индекс_риска']:.0f}/100
    """
    return pdf_content

# ==================== ГЛАВНОЕ ПРИЛОЖЕНИЕ ====================

def main():
    check_password()
    apply_styles()
    
    st.set_page_config(page_title="Панель управления", layout="wide", initial_sidebar_state="expanded")
    
    # Загрузка данных
    if "df" not in st.session_state:
        st.session_state.df = load_sample_data()
    
    df = st.session_state.df
    показатели = рассчитать_показатели(df)
    прогноз = прогноз_на_3_месяца(df)
    abc = abc_анализ(df)
    сравнение = сравнение_периодов(df)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### ⚙️ Управление")
        
        # Загрузка данных
        st.markdown("**Загрузка данных**")
        uploaded_file = st.file_uploader("Загрузите CSV с данными", type="csv")
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✓ Данные загружены")
            st.rerun()
        
        # Фильтры
        st.markdown("**Фильтры**")
        periode_filter = st.selectbox("Период", ["Все", "Последний год", "Последний квартал", "Последний месяц"])
        
        region_filter = st.multiselect("Регион", df['Регион'].unique(), default=df['Регион'].unique())
        category_filter = st.multiselect("Категория", df['Категория'].unique(), default=df['Категория'].unique())
        
        # Применение фильтров
        df_filtered = df.copy()
        if periode_filter == "Последний год":
            df_filtered = df_filtered[df_filtered['Дата'] >= df_filtered['Дата'].max() - timedelta(days=365)]
        elif periode_filter == "Последний квартал":
            df_filtered = df_filtered[df_filtered['Дата'] >= df_filtered['Дата'].max() - timedelta(days=90)]
        elif periode_filter == "Последний месяц":
            df_filtered = df_filtered[df_filtered['Дата'] >= df_filtered['Дата'].max() - timedelta(days=30)]
        
        df_filtered = df_filtered[df_filtered['Регион'].isin(region_filter) & df_filtered['Категория'].isin(category_filter)]
        
        # Экспорт
        st.markdown("**Экспорт отчёта**")
        excel_file = экспорт_в_excel(df_filtered, показатели, прогноз)
        st.download_button(
            label="📥 Скачать Excel",
            data=excel_file.getvalue(),
            file_name=f"отчёт_{datetime.now().strftime('%d.%m.%Y')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Выход
        st.markdown("---")
        if st.button("🚪 Выход"):
            st.session_state.authenticated = False
            st.rerun()
    
    # ==================== ГЛАВНАЯ СТРАНИЦА ====================
    
    # Заголовок
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 📊 Панель управления")
        st.markdown("*Профессиональная аналитика вашего бизнеса*")
    with col2:
        st.markdown(f"**Обновлено:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    st.markdown("---")
    
    # ==================== 6 ОСНОВНЫХ ПОКАЗАТЕЛЕЙ ====================
    st.markdown("### 📈 Ключевые показатели")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Доход</div>
            <div class="metric-value">{показатели['доход_текущий']/1_000_000:.2f}М</div>
            <div class="metric-change {'positive' if показатели['доход_изменение'] >= 0 else 'negative'}">
                {'↑' if показатели['доход_изменение'] >= 0 else '↓'} {abs(показатели['доход_изменение']):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        цвет = 'warning' if показатели['маржа_снижается'] else 'positive'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Маржа</div>
            <div class="metric-value {цвет}">{показатели['маржа']:.1f}%</div>
            <div class="metric-change neutral">
                {'⚠️ На спаде' if показатели['маржа_снижается'] else '✓ Стабильна'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        цвет = 'positive' if показатели['выполнение_плана'] >= 95 else 'warning' if показатели['выполнение_плана'] >= 85 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Выполнение плана</div>
            <div class="metric-value {цвет}">{показатели['выполнение_плана']:.0f}%</div>
            <div class="metric-change neutral">факт / план</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        цвет = 'positive' if показатели['индекс_риска'] < 30 else 'warning' if показатели['индекс_риска'] < 60 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Индекс риска</div>
            <div class="metric-value {цвет}">{показатели['индекс_риска']:.0f}</div>
            <div class="metric-change neutral">шкала 0-100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        цвет = 'positive' if показатели['заказы_изменение'] >= 0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Объём заказов</div>
            <div class="metric-value">{показатели['заказы']}</div>
            <div class="metric-change {цвет}">
                {'↑' if показатели['заказы_изменение'] >= 0 else '↓'} {abs(показатели['заказы_изменение'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        изм = сравнение['текущий_доход'] - сравнение['прошлый_доход']
        цвет = 'positive' if изм >= 0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Год к году</div>
            <div class="metric-value {цвет}">{изм/1_000_000:+.2f}М</div>
            <div class="metric-change neutral">за год</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== ГРАФИК ПЛАН VS ФАКТ ====================
    st.markdown("### 📉 План vs Факт")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['Дата'], y=df_filtered['Факт'], mode='lines', name='Факт', line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=df_filtered['Дата'], y=df_filtered['План'], mode='lines', name='План', line=dict(color='#8a92a0', width=2, dash='dash')))
    
    fig.update_layout(
        template='plotly_dark', hovermode='x unified', height=400,
        paper_bgcolor='#0f1419', plot_bgcolor='#1a1f29',
        font=dict(color='#e0e0e0', size=12),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#2a3038'),
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # ==================== ПРОГНОЗ ====================
    st.markdown("### 🔮 Прогноз на 3 месяца")
    
    st.markdown(f"""
    <div class="forecast-box">
    <strong>Прогнозируемый доход на основе тренда:</strong><br>
    {f"{прогноз.iloc[0]['Прогноз']/1_000_000:.2f}М → {прогноз.iloc[1]['Прогноз']/1_000_000:.2f}М → {прогноз.iloc[2]['Прогноз']/1_000_000:.2f}М"}
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== ABC АНАЛИЗ ====================
    st.markdown("### 💡 ABC-анализ клиентов")
    st.markdown("*Какие клиенты приносят 80% дохода*")
    
    abc_a = abc[abc['Категория'] == 'A']
    abc_b = abc[abc['Категория'] == 'B']
    abc_c = abc[abc['Категория'] == 'C']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Категория A", f"{len(abc_a)} клиентов", f"{abc_a['Доля_%'].sum():.1f}% дохода")
    with col2:
        st.metric("🥈 Категория B", f"{len(abc_b)} клиентов", f"{abc_b['Доля_%'].sum():.1f}% дохода")
    with col3:
        st.metric("🥉 Категория C", f"{len(abc_c)} клиентов", f"{abc_c['Доля_%'].sum():.1f}% дохода")
    
    st.dataframe(abc.head(15), use_container_width=True, hide_index=True)
    
    # ==================== СРАВНЕНИЕ ПЕРИОДОВ ====================
    st.markdown("### 📊 Сравнение с прошлым годом")
    
    col1, col2 = st.columns(2)
    with col1:
        изм = сравнение['текущий_доход'] - сравнение['прошлый_доход']
        pct = (изм / сравнение['прошлый_доход'] * 100) if сравнение['прошлый_доход'] > 0 else 0
        st.metric("Доход (текущий год)", f"{сравнение['текущий_доход']/1_000_000:.2f}М", f"{pct:+.1f}%")
    with col2:
        изм = сравнение['текущее_выполнение'] - сравнение['прошлое_выполнение']
        st.metric("Выполнение плана", f"{сравнение['текущее_выполнение']:.0f}%", f"{изм:+.1f}%")
    
    # ==================== ТАБЛИЦА ДАННЫХ ====================
    st.markdown("### 📋 Подробные данные")
    
    cols_to_show = ['Дата', 'Регион', 'Категория', 'Факт', 'План', 'Заказы', 'Маржа']
    st.dataframe(
        df_filtered[cols_to_show].sort_values('Дата', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # ==================== АЛЕРТЫ ====================
    st.markdown("### ⚠️ Алерты и уведомления")
    
    if показатели['индекс_риска'] > 60:
        st.markdown(f"""
        <div class="alert-box" style="border-left-color: #ef4444;">
        <strong>🚨 КРИТИЧЕСКОЕ ВНИМАНИЕ</strong><br>
        Индекс риска достиг {показатели['индекс_риска']:.0f}%. Требуется срочное действие.
        </div>
        """, unsafe_allow_html=True)
    
    if показатели['выполнение_плана'] < 85:
        st.markdown(f"""
        <div class="alert-box warning">
        <strong>⚠️ Выполнение плана низкое</strong><br>
        Текущее выполнение {показатели['выполнение_плана']:.0f}%, требуется активизация.
        </div>
        """, unsafe_allow_html=True)
    
    if показатели['маржа_снижается']:
        st.markdown(f"""
        <div class="alert-box warning">
        <strong>⚠️ Маржа снижается</strong><br>
        Текущая маржа {показатели['маржа']:.1f}%. Проверьте себестоимость.
        </div>
        """, unsafe_allow_html=True)
    
    if показатели['индекс_риска'] < 30:
        st.markdown(f"""
        <div class="alert-box success">
        <strong>✅ Бизнес в порядке</strong><br>
        Все показатели в норме. Индекс риска {показатели['индекс_риска']:.0f}%.
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8a92a0; font-size: 12px;">
    <strong>Панель управления бизнесом</strong> | Профессиональный инструмент аналитики | v1.0<br>
    Цена: 50 000 руб | Вопросы: support@example.com
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()