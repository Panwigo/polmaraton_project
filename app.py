# ========================================
# IMPORTY
# ========================================

import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel
from typing import Optional
from langfuse.decorators import observe
from langfuse.openai import OpenAI

# ========================================
# KONFIGURACJA
# ========================================

# Wczytaj zmienne środowiskowe z .env (klucz OpenAI)
load_dotenv()

# Inicjalizacja Langfuse
from langfuse import Langfuse
langfuse = Langfuse()  # Automatycznie wczytuje klucze z .env

# Stałe
MODEL_NAME = 'polmaraton_model'

# ========================================
# FUNKCJE POMOCNICZE - FORMATOWANIE
# ========================================

def seconds_to_time(seconds: float) -> str:
    """Konwertuje sekundy na format HH:MM:SS"""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ========================================
# ŁADOWANIE MODELU
# ========================================

@st.cache_resource
def load_marathon_model():
    """Ładuje model PyCaret (cache = ładuje tylko raz)"""
    return load_model(MODEL_NAME)

# ========================================
# PYDANTIC SCHEMA - STRUKTURA DANYCH DLA GPT
# ========================================

class Features(BaseModel):
    """Definiuje jakie dane GPT ma wyłuskać z tekstu"""
    sex: Optional[str] = None
    age: Optional[int] = None
    time_5km_seconds: Optional[int] = None

# ========================================
# PARSOWANIE TEKSTU PRZEZ GPT
# ========================================

def get_openai_client():
    """Zwraca klienta OpenAI z kluczem API"""
    return OpenAI(api_key=st.session_state["openai_api_key"])

@observe()
def retrieve_structure(text: str) -> dict:
    """
    Parsuje tekst naturalny i wyłuskuje dane używając GPT
    
    INPUT: "Mam 30 lat, jestem mężczyzną, 5km biegnę w 25 minut"
    OUTPUT: {"sex": "mężczyzna", "age": 30, "time_5km_seconds": 1500}
    """
    instructor_client = instructor.from_openai(get_openai_client())
    
    result = instructor_client.chat.completions.create(
        model="gpt-4o",
        response_model=Features,
        messages=[
            {
                "role": "system",
                "content": "Wyciągasz dane z tekstu. Zwróć JSON: {sex, age, time_5km_seconds}. Jeśli czegoś brakuje, użyj null. Czas przelicz na sekundy."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    
    return result.model_dump()

# ========================================
# WALIDACJA DANYCH
# ========================================

def validate_data(data: dict) -> list:
    """
    Sprawdza które wymagane dane brakują
    
    INPUT: {"sex": None, "age": 30, "time_5km_seconds": 1500}
    OUTPUT: ["sex"] (lista brakujących kluczy)
    """
    required = ["sex", "age", "time_5km_seconds"]
    missing = []
    
    for require in required:
        if require not in data or data[require] == None:
            missing.append(require)
    
    return missing

# ========================================
# PRZYGOTOWANIE DATAFRAME DLA MODELU
# ========================================

def create_input_df(sex: str, age: int, time_5km_seconds: int) -> pd.DataFrame:
    """
    Tworzy DataFrame w formacie oczekiwanym przez model
    
    INPUT: sex="mężczyzna", age=30, time_5km_seconds=1500
    OUTPUT: DataFrame z kolumnami: Płeć, 5 km Czas, Wiek
    """
    
    # Normalizuj płeć do M/K
    sex_lower = sex.lower()
    if sex_lower in ["mężczyzna", "man", "male", "m"]:
        sex_code = "M"
    elif sex_lower in ["kobieta", "woman", "female", "k"]:
        sex_code = "K"
    else:
        sex_code = sex  # Na wszelki wypadek użyj co jest
    
    # Stwórz DataFrame zgodny z modelem
    df = pd.DataFrame([{
        'Płeć': sex_code,
        '5 km Czas': float(time_5km_seconds),
        'Wiek': age
    }])
    
    return df

# ========================================
# INTERFEJS STREAMLIT
# ========================================

# KROK 0: Inicjalizacja session_state dla klucza OpenAI
if "openai_api_key" not in st.session_state:
    # Sprawdź czy klucz jest w zmiennych środowiskowych (.env)
    import os
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        st.session_state["openai_api_key"] = None

# Jeśli nie ma klucza, poproś użytkownika
if not st.session_state["openai_api_key"]:
    st.info("🔑 Dodaj swój klucz OpenAI API, aby korzystać z aplikacji")
    st.page_link("https://platform.openai.com/account/api-keys", 
                 label="Zdobądź swój klucz tutaj", 
                 icon="🔑")
    
    api_key_input = st.text_input("Klucz API", type="password")
    
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input
        st.rerun()  # Odśwież aplikację z nowym kluczem
    
    st.stop()  # Zatrzymaj aplikację jeśli nie ma klucza

# ========================================
# GŁÓWNY INTERFEJS
# ========================================

st.title("🏃 A jaki czas Ty możesz mieć na półmaratonie?")

st.write("""
Dzięki modelowi wytrenowanemu na danych z półmaratonu Wrocławskiego jesteśmy 
w stanie obliczyć szacowany czas na bazie Twoich informacji.
""")

st.info("""
💡 **Jak to działa?**  
Opisz siebie tekstem - podaj wiek, płeć i swój najlepszy czas na 5km.  
Sztuczna inteligencja przeanalizuje Twój opis i przewidzi czas na półmaraton!
""")

# ========================================
# FORMULARZ
# ========================================

with st.form("user_form"):
    st.subheader("Opowiedz nam o sobie")
    
    user_text = st.text_area(
        "Ile masz lat? Jakiej jesteś płci? Jaki jest Twój najlepszy czas na 5km?",
        placeholder="Przykład: Mam 30 lat, jestem mężczyzną, 5km biegnę w 25 minut",
        height=100
    )
    
    submitted = st.form_submit_button("🔮 Oblicz mój czas!", type="primary")

# ========================================
# PRZETWARZANIE PO KLIKNIĘCIU
# ========================================

if submitted:
    if not user_text.strip():
        st.error("❌ Proszę wpisać opis!")
        st.stop()
    
    try:
        # KROK 2: Parsowanie przez GPT
        with st.spinner("🤖 Analizuję Twój opis..."):
            gpt_data = retrieve_structure(user_text)
        
        st.success("✅ Dane wyłuskane!")
        
        # Pokaż co GPT znalazł (dla debugowania)
        with st.expander("🔍 Co zrozumiałem"):
            st.json(gpt_data)
        
        # KROK 3: Walidacja
        missing = validate_data(gpt_data)
        
        if missing:
            # Mapowanie kluczy na polskie nazwy
            missing_names = {
                "sex": "Płeć",
                "age": "Wiek",
                "time_5km_seconds": "Czas na 5km"
            }
            missing_polish = [missing_names.get(k, k) for k in missing]
            
            st.error(f"❌ Brakuje danych: {', '.join(missing_polish)}")
            st.info("💡 Spróbuj ponownie i podaj wszystkie informacje!")
            st.stop()
        
        # KROK 4: Tworzenie DataFrame
        input_df = create_input_df(
            sex=gpt_data["sex"],
            age=gpt_data["age"],
            time_5km_seconds=gpt_data["time_5km_seconds"]
        )
        
        # Pokaż DataFrame (dla debugowania)
        with st.expander("📊 Dane dla modelu"):
            st.dataframe(input_df)
        
        # KROK 5: Predykcja
        with st.spinner("🔮 Obliczam Twój czas..."):
            model = load_marathon_model()
            prediction = predict_model(model, data=input_df)
            predicted_seconds = prediction["prediction_label"].iloc[0]
        
        # KROK 6: Wyświetlenie wyniku
        st.success("🎉 Obliczenia zakończone!")
        
        st.metric(
            label="Przewidywany czas na półmaraton (21 km)",
            value=seconds_to_time(predicted_seconds),
            help="To szacunkowy czas na podstawie Twojego profilu"
        )
        
        # Dodatkowe informacje
        with st.expander("📊 Szczegóły"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Płeć", input_df["Płeć"].iloc[0])
            with col2:
                st.metric("Wiek", input_df["Wiek"].iloc[0])
            with col3:
                st.metric("Czas 5km", f"{gpt_data['time_5km_seconds']//60} min")
        
        st.balloons()  # Animacja sukcesu! 🎈
        
    except Exception as e:
        st.error(f"❌ Wystąpił błąd: {str(e)}")
        st.info("💡 Spróbuj ponownie lub skontaktuj się z administratorem")
        # Pokaż pełny błąd (tylko dla developmentu)
        with st.expander("🐛 Szczegóły błędu (dla deweloperów)"):
            st.exception(e)