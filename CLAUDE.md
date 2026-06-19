# Seminář: AI v medicíně — Streamlit aplikace

Interaktivní webová aplikace pro seminář o AI pro studenty 1. ročníku LF UK (autor: Jaroslav Valdauf, jaroslav.valdauf@lf3.cuni.cz). Veřejně dostupná na **https://biofyzika.valdauf.eu** přes reverse proxy.

## Účel a pedagogický cíl

Studenti na vlastní kůži zažijí, že **AI model je jen tak dobrý jako data, na kterých byl naučen** (*garbage in, garbage out*).

**Flow semináře (~20 minut na skupinu):**
1. Přihlášení heslem skupiny
2. Stažení dvou trénovacích datasetů (A a B) — oba vypadají stejně, jeden má náhodně přeházené štítky
3. Natrénování modelu na jednom datasetu — živá konzole + graf ztráty (~1–2 min)
4. Krok 3 fáze 1: prohlídka testovacích pacientů → kliknutí "Předhodit" → živá konzole + graf
5. Krok 3 fáze 2: data a tlačítko zmizí, zůstane konzole + graf + tabulka predikcí
6. Totéž s druhým datasetem
7. Krok 4: porovnání → odhalení správné odpovědi

## Spuštění

```powershell
pip install -r requirements.txt
python prepare_data.py   # jednou — stáhne UCI datasety, augmentuje, vytvoří data/
streamlit run app.py
```

Po změně `correct_dataset` v `settings.yaml` je nutné znovu spustit `prepare_data.py`.

## Struktura projektu

```
app.py              — hlavní Streamlit aplikace (veškerá UI logika)
settings.yaml       — konfigurace skupin, hesel, MLP parametrů, překladů sloupců
prepare_data.py     — stáhne/připraví datasety, augmentuje na 3000 řádků
ml/trainer.py       — preprocess() + build_mlp()
.streamlit/config.toml  — server na 0.0.0.0, enableCORS=false, headless=true
data/group{1-6}/
    dataset_A.csv   — trénovací data (správná nebo přeházená, dle correct_dataset)
    dataset_B.csv   — trénovací data (opačná)
    test.csv        — testovací data (se štítky, server-side only)
```

## 6 skupin a jejich datasety

| Skupina | Téma | Heslo | Správný dataset | Zdroj |
|---------|------|-------|-----------------|-------|
| 1 | Diagnostika srdeční choroby | `srdce` | **B** | UCI Heart Disease (id=45) |
| 2 | Diagnostika diabetu | `cukr` | **A** | Pima Indians Diabetes (id=34) |
| 3 | Diagnostika rakoviny prsu | `prsa` | **A** | sklearn breast_cancer |
| 4 | Prognóza přežití po srdečním selhání | `infarkt` | **A** | UCI Heart Failure (id=519) |
| 5 | Prognóza readmise do nemocnice | `nemocnice` | **A** | syntetická data |
| 6 | Prognóza těžkého průběhu sepse | `sepse` | **B** | syntetická data |

Vzor B,A,A,A,A,B — záměrně nepravidelný, aby si skupiny nemohly poradit.

Admin heslo: `YouShallNotPass` (zobrazí přehled skupin s tlačítkem přímého přihlášení)

## Mechanismus "špatného" datasetu

`prepare_data.py` náhodně přeháže (`shuffle`) sloupec s cílovými štítky v trénovacích datech. Model trénovaný na takovém datasetu se nenaučí žádné smysluplné vzory — přesnost ~50 % oproti 75–90 % u správného datasetu.

## Datové soubory

Skupiny 1–4 používají reálná UCI data augmentovaná Gaussovým šumem (5 % std na spojitých sloupcích) na 3 000 řádků. Skupiny 5–6 mají syntetická klinicky věrohodná data (3 000 řádků). Všechny skupiny: 2 250 trénovacích + 750 testovacích vzorků po splitu 75/25.

## MLP konfigurace (settings.yaml)

```yaml
mlp:
  hidden_layer_sizes: [128, 64, 32]
  max_iter: 400
  n_iter_no_change: 60   # early stopping — čeká 60 průchodů bez zlepšení
  epoch_delay: 0.3       # sleep 0.3s/průchod → trénink trvá ~1–2 minuty
  early_stopping: true
  validation_fraction: 0.15
```

`epoch_delay` je klíčový pro pedagogický efekt — bez něj by trénink trval <10 s.

## Klíčové implementační detaily

### app.py — architektura

#### Background trénink (threading + st.fragment)

- **`_get_jobs()` / `_get_lock()`** — `@st.cache_resource` funkce vracející sdílený dict a threading.Lock. **Kritické:** modul-level `dict = {}` se při každém `st.rerun()` resetuje (Streamlit re-spouští celý skript). `@st.cache_resource` vrací vždy stejnou instanci, takže background thread zapisuje a main thread čte přes reruns. Klíč jobu: `{session_id}_{ds}`.
- **`get_session_id()`** — vrátí Streamlit session ID ze `ScriptRunContext`.
- **`_training_worker(...)`** — běží v daemonic threadu. Každý průchod aktualizuje `_TRAINING_JOBS[job_key]` (epoch, loss, history, log_lines). Po dokončení přidá mlp, výsledky a nastaví `done=True`.
- **`run_training(ds, ...)`** — preprocessuje data (s spinner), vytvoří job v `_TRAINING_JOBS`, uloží `training_job_{ds}` do session_state, spustí daemon thread.
- **`_collect_done_jobs()`** — volaná na začátku každého `render_dashboard()`. Pokud je job `done`, přesune model do `session_state` a vyčistí job i klíč.
- **`@st.fragment(run_every="0.5s") show_training_fragment(ds)`** — fragment se sám aktualizuje každých 0.5s, čte z `_TRAINING_JOBS`, zobrazuje konzoli + graf + progress bar. Když zachytí `done=True`, zavolá `st.rerun()` (full rerun) → `_collect_done_jobs` přesune model → Krok 3 se zobrazí.
- Při přepnutí datasetu fragment zaniká, trénink pokračuje v threadu. Indikátor `⏳ Dataset X se trénuje na pozadí` informuje o probíhajícím background tréninku.

#### Ostatní funkce

- **`EXPLANATION_TEXT`** — konstanta s vysvětlením pojmů; sdílená mezi fragmentem a statickým zobrazením po rerun.
- **`fixed_log(lines)`** — padduje log na `LOG_HEIGHT=16` řádků, blok nemění výšku.
- **`predict_with_progress(model_data, group, result_key)`** — batch predikce (25 dávek), živá konzole + graf % pozitivních predikcí. Ukládá `log_lines` a `pos_history` do `st.session_state[result_key]` pro zobrazení po rerun.
- **`render_krok3(trained_ds, group, neg_label, pos_label)`** — dvě fáze:
  - **Fáze 1** (`result_key` není v session_state): tabulka testovacích dat + tlačítko "Předhodit"
  - **Fáze 2** (`result_key` v session_state): data a tlačítko zmizí, zůstane konzole + graf + tabulka predikcí
  - Zobrazuje se pouze pro aktuálně vybraný dataset (přepnutím se skryje, přepnutím zpět obnoví).
- **`translate_cols(df, group)`** — přejmenuje sloupce dle `column_labels` v settings.yaml.
- **`build_table(records, group, show_true)`** — sestaví tabulku predikcí; `show_true` odhalí sloupec "Správná diagnóza".

### Terminologie v UI

- "Epocha" → **"Průchod"** (srozumitelnější pro mediky)
- Konzole zobrazuje: `✓ Konvergováno v průchodu N.`
- Vysvětlivky (průchod, ztráta, přesnost, konvergováno) v konstantě `EXPLANATION_TEXT`

### Plotly grafy — unikátní klíče

Všechny `st.plotly_chart()` mají explicitní `key=`. Grafy v smyčkách (trénink, predikce) dostávají `key=str(uuid.uuid4())` — bez toho hází `StreamlitDuplicateElementKey`. Statické grafy (po rerun) mají fixní klíče: `f"train_saved_{ds}"`, `f"pred_saved_{trained_ds}"`, `"comparison"`, `f"cm_{trained_ds}"`.

### trainer.py — funkce

```python
preprocess(train_path, test_path, target_column, display_columns)
    → X_train_sc, y_train_enc, X_test_sc, y_test_enc, display_df

build_mlp(mlp_cfg)
    → MLPClassifier(warm_start=True, max_iter=1, ...)  # pro epochový loop v threadu
```

Predikce i tréninkový loop probíhají v `app.py` kvůli live UI updatům (trénink v background threadu, predikce synchronně s placeholdery).

### Požadovaná verze Streamlit

`streamlit>=1.37.0` — nutné pro `@st.fragment(run_every=...)`. Uvedeno v `requirements.txt`.

### Překlady sloupců

Každá skupina má v `settings.yaml` klíč `column_labels` — slovník `{anglický_název: český_název_s_jednotkou}`. Překlady se aplikují v: náhledu datasetů, tabulce testovacích pacientů, tabulce predikcí. CSV soubory ke stažení ponechávají původní anglické názvy.

### Opravené bugy

- **Přerušení tréninku při přepnutí datasetu**: trénink nyní běží v background threadu, `st.rerun()` ho nepřeruší. Fragment `show_training_fragment` polluje výsledky každých 0.5s.
- **Tlačítko zmizí a nic se neděje**: modul-level `_TRAINING_JOBS = {}` se resetoval při každém `st.rerun()`. Oprava: `@st.cache_resource` na `_get_jobs()` a `_get_lock()`.
- **Mizející Krok 3**: `render_krok3()` volaná pro aktuálně vybraný `ds`; výsledky v session_state se obnoví při přepnutí zpět.
- **StreamlitDuplicateElementKey**: grafy v smyčkách dostávají `key=str(uuid.uuid4())`.
- **WebSocket rejected**: `enableCORS = false` v `.streamlit/config.toml`.

## .streamlit/config.toml

```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true      # přeskočí výzvu k zadání emailu
enableCORS = false   # nutné pro přístup přes reverse proxy / vlastní doménu

[browser]
serverAddress = "biofyzika.valdauf.eu"
serverPort = 443
```

## Přihlašovací pole

`st.text_input` bez `type="password"` — znaky jsou viditelné, prohlížeč nerozpozná pole jako heslo a nenabídne uložení.

## Patička

`render_footer()` volaná z `main()` — zobrazí se na všech stránkách (login, dashboard, admin). Implementována jako **sticky CSS** (`position: fixed; bottom: 0`), takže je vždy viditelná bez ohledu na délku stránky nebo fázi tréninku. Používá `unsafe_allow_html=True`.

```
Jaroslav Valdauf · 3. LF UK · 2026
```

## Fixní výška konzole (LOG_HEIGHT)

Konstanta `LOG_HEIGHT = 16` (řádků). Pomocná funkce `fixed_log(lines)` padduje log na vždy stejný počet řádků — blok `st.code()` nemění výšku při tréninku, stránka neposkakuje. Používá se ve všech `log_ph.code(...)` voláních v `run_training()` i `predict_with_progress()`.

## Známé limitace

- **Scrollování při tréninku**: `progress_ph` je umístěn **pod** sloupci s konzolí a grafem, aby jeho updaty neposouvaly obsah výše. Fixní výška logu (`LOG_HEIGHT`) zabraňuje růstu bloku. Úplné odstranění problému by vyžadovalo přechod na jiný framework.

## Možná vylepšení (neimplementováno)

- Zobrazení ROC křivky pro každý model
- Export výsledků do PDF/CSV pro vyučujícího
- Časový limit na seminář (countdown timer)
- Možnost učitele resetovat session všech skupin najednou
- Více vizualizací dat v Kroku 1 (distribuce proměnných, korelační matice)
