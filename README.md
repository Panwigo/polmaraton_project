# 🏃 Predykcja Czasu Półmaratonu

Aplikacja Streamlit przewidująca czas na półmaraton na podstawie danych użytkownika: płeć, wiek i czas na 5km.

## 🎯 Funkcjonalności

- 🤖 **Parsowanie języka naturalnego** - GPT-4o wyłuskuje dane z opisu tekstowego
- ✅ **Walidacja danych** - sprawdzanie kompletności informacji
- 🔮 **Predykcja ML** - model PyCaret trenowany na danych z Półmaratonu Wrocławskiego
- 📊 **Monitoring** - Langfuse śledzi wywołania LLM

## 📐 Schemat przepływu danych
```
┌─────────────────────────────────────────────────────┐
│ KROK 1: Użytkownik pisze tekst                      │
│ "Mam 30 lat, jestem mężczyzną, 5km biegnę w 25 min" │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ KROK 2: retrieve_structure(text)                    │
│ GPT parsuje → {"sex": "mężczyzna", "age": 30, ...}  │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ KROK 3: validate_data(gpt_data)                     │
│ Sprawdza czy czegoś nie brakuje → [] lub ["sex"]   │
└─────────────────────┬───────────────────────────────┘
                      ↓
         ┌────────────┴─────────────┐
         │ Brakuje danych?          │
         └──────┬──────────┬────────┘
             NIE│        TAK│
                ↓           ↓
    ┌───────────────┐  ┌────────────────────┐
    │ KROK 4:       │  │ Komunikat błędu:   │
    │ create_input_ │  │ "Podaj swoją płeć!"│
    │ df()          │  └────────────────────┘
    │ → DataFrame   │
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │ KROK 5:       │
    │ load_model()  │
    │ predict_model │
    │ → Wynik!      │
    └───────────────┘
```

## 🛠️ Technologie

- **Frontend:** Streamlit
- **ML:** PyCaret (regression)
- **LLM:** OpenAI GPT-4o + Instructor + Pydantic
- **Monitoring:** Langfuse
- **Data:** Pandas

## 🚀 Instalacja lokalna

1. Sklonuj repo:
```bash
git clone https://github.com/twoj-username/polmaraton-app.git
cd polmaraton-app
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Stwórz plik `.env`:
```
OPENAI_API_KEY=twoj-klucz
LANGFUSE_PUBLIC_KEY=twoj-klucz
LANGFUSE_SECRET_KEY=twoj-klucz
LANGFUSE_HOST=https://cloud.langfuse.com
```

4. Uruchom aplikację:
```bash
streamlit run app.py
```

## 📝 Jak używać

1. Wpisz swoje dane w języku naturalnym:
```
   Mam 30 lat, jestem mężczyzną, 5km biegnę w 25 minut
```

2. Kliknij "Oblicz mój czas!"

3. Zobacz przewidywany czas na półmaraton! 🎉

## 📊 Model

Model wytrenowany na danych z Półmaratonu Wrocławskiego.

**Wejście:**
- Płeć (M/K)
- Wiek
- Czas na 5km (sekundy)

**Wyjście:**
- Przewidywany czas na półmaraton (21 km)

## 🔐 Zmienne środowiskowe

Aplikacja wymaga kluczy API w pliku `.env`:
- `OPENAI_API_KEY` - klucz OpenAI
- `LANGFUSE_PUBLIC_KEY` - monitoring Langfuse
- `LANGFUSE_SECRET_KEY` - monitoring Langfuse
- `LANGFUSE_HOST` - endpoint Langfuse

## 📄 Licencja

MIT