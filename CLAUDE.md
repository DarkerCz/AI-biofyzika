# Seminář: AI v medicíně — Streamlit aplikace

Interaktivní webová aplikace pro seminář o AI pro studenty 1. ročníku LF UK (autor: Jaroslav Valdauf, jaroslav.valdauf@lf3.cuni.cz). Veřejně dostupná na **https://biofyzika.valdauf.eu** přes reverse proxy.

## Účel a pedagogický cíl

Studenti na vlastní kůži zažijí, že **AI model je jen tak dobrý jako data, na kterých byl naučen** (*garbage in, garbage out*).

**Flow semináře (~20 minut na skupinu):**
1. Přihlášení heslem skupiny (Enter nebo kliknutí, znaky viditelné)
2. Krok 1: prohlídka čtyř datasetů (A–D) přímo v záložkách — všechny vypadají stejně
3. Krok 2: trénink modelu na zvoleném datasetu — živá konzole + graf ztráty (~2–3 min, běží na pozadí)
4. Krok 3 fáze 1: prohlídka testovacích pacientů → kliknutí "Předhodit" → živá konzole + graf (~30 s)
5. Krok 3 fáze 2: data a tlačítko zmizí, zůstane konzole + graf + tabulka predikcí
6. Opakovat pro všechny 4 datasety
7. Krok 4: průběžné porovnání od 2 modelů, plná odpověď po otestování všech 4

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
.streamlit/config.toml  — server na 0.0.0.0, enableCORS=true, headless=true
data/group{1-6}/
    dataset_A.csv   — trénovací data
    dataset_B.csv   — trénovací data
    dataset_C.csv   — trénovací data
    dataset_D.csv   — trénovací data
    test.csv        — testovací data (se štítky, server-side only)
```

## 6 skupin a jejich datasety

| Skupina | Téma | Heslo | Správný dataset | Zdroj |
|---------|------|-------|-----------------|-------|
| 1 | Diagnostika srdeční choroby | `srdce` | **C** | Fedesoriano (2021) — 918 řádků, 11 features, target: `HeartDisease` |
| 2 | Diagnostika diabetu | `cukr` | **D** | Pima Indians Diabetes (id=34) |
| 3 | Diagnostika rakoviny prsu | `prsa` | **B** | sklearn breast_cancer |
| 4 | Prognóza přežití po srdečním selhání | `infarkt` | **A** | UCI Heart Failure (id=519) |
| 5 | Prognóza readmise do nemocnice | `nemocnice` | **D** | syntetická data |
| 6 | Prognóza těžkého průběhu sepse | `sepse` | **C** | syntetická data |

Vzor C,D,B,A,D,C — záměrně nepravidelný, žádné dvě sousední skupiny nemají stejný správný dataset.

Každá skupina má 4 trénovací datasety (A–D). Jeden je správný, ostatní tři jsou pokazeny různými metodami — pořadí korupcí se liší skupinu od skupiny (definováno v `_CORRUPTION_ORDER` v `prepare_data.py`):

| Typ korupce | Popis |
|------------|-------|
| Přeházené štítky | `shuffle` — náhodné pořadí štítků → model se nic nenaučí (~50 % přesnost) |
| Invertované štítky | `1 - label` — každý štítek překlopený → model se naučí opak (~0–30 % přesnost) |
| Částečná korupce | 30 % štítků náhodně překlopeno → model se naučí jen část (~60–70 % přesnost) |

Admin heslo: `YouShallNotPass` — admin tabulka zobrazuje hesla, správné datasety a tlačítko **Vstoupit** pro přímé přihlášení do libovolné skupiny bez zadání hesla.

## Mechanismus "špatných" datasetů

`prepare_data.py` generuje 4 trénovací datasety — jeden správný a tři pokazené různými metodami (`shuffle_labels`, `invert_labels`, `partial_shuffle_labels`). Pořadí korupcí pro každou skupinu je pevně dáno v `_CORRUPTION_ORDER` (různé pro každou skupinu, aby si skupiny nemohly poradit).

## Datové soubory

Všechny skupiny: 2 250 trénovacích + 750 testovacích vzorků po splitu 75/25, augmentováno Gaussovým šumem (5 % std) na 3 000 řádků.

- **Skupina 1**: Fedesoriano (2021) — 918 původních řádků. Kategoriální sloupce (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) zakódovány číselně v `prepare_data.py`. Fallback: Cleveland UCI (id=45) s přejmenovanými sloupci na Fedesoriano schéma.
- **Skupiny 2–4**: reálná UCI data.
- **Skupiny 5–6**: syntetická klinicky věrohodná data.

## MLP konfigurace (settings.yaml)

```yaml
mlp:
  hidden_layer_sizes: [128, 64, 32]
  max_iter: 400
  n_iter_no_change: 60   # early stopping — čeká 60 průchodů bez zlepšení
  epoch_delay: 0.5       # sleep 0.5s/průchod → trénink trvá ~2–3 minuty
  predict_delay: 1.2     # sleep 1.2s/dávku → zpracování testovacích dat ~30 s
  predict_batches: 25    # počet dávek; velikost dávky = n_test // predict_batches
  early_stopping: true
  validation_fraction: 0.15
```

`epoch_delay` je klíčový pro pedagogický efekt — bez něj by trénink trval <10 s. Implementováno jako `time.sleep(epoch_delay)` na konci každé epochy v `_training_worker()` v `app.py`. Podobně `predict_delay` zpomaluje zpracování testovacích dat v `predict_with_progress()`.

## Klíčové implementační detaily

### app.py — architektura

#### Background trénink (threading + st.fragment)

- **`_get_jobs()` / `_get_lock()`** — `@st.cache_resource` funkce vracející sdílený dict a threading.Lock. **Kritické:** modul-level `dict = {}` se při každém `st.rerun()` resetuje (Streamlit re-spouští celý skript). `@st.cache_resource` vrací vždy stejnou instanci přes reruns. Klíč jobu: `{session_id}_{ds}`.
- **`get_session_id()`** — vrátí Streamlit session ID ze `ScriptRunContext`.
- **`_training_worker(job_key, run_id, ...)`** — běží v daemonic threadu. Před každým zápisem volá `still_mine()` — ověří, že `_get_jobs()[job_key]["run_id"] == run_id` a `cancelled == False`. Pokud ne (byl spuštěn nový trénink), vlákno se tiše ukončí. Zabraňuje skoků epoch při souběžných vlákna se stejným klíčem.
- **`run_training(ds, ...)`** — před startem zruší starý job se stejným klíčem (`cancelled=True`), vygeneruje `run_id = uuid4()`, vytvoří nový job dict s `run_id`, uloží `training_job_{ds}` do session_state, spustí daemon thread.
- **`_collect_done_jobs()`** — volaná na začátku každého `render_dashboard()` pro všechna A–D. Pokud je job `done`, přesune model do `session_state` a vyčistí job i klíč.
- **`@st.fragment(run_every="0.5s") show_training_fragment(ds)`** — polluje `_get_jobs()` každých 0.5s, zobrazuje konzoli + graf + progress bar. Když zachytí `done=True`, zavolá `st.rerun()` → `_collect_done_jobs` přesune model → Krok 3 se zobrazí.
- Při přepnutí datasetu fragment zaniká, trénink pokračuje v threadu. Indikátor `⏳ Trénují se na pozadí: Dataset X (průchod Y/400)` informuje o všech aktivních background trénincích.

#### Ostatní funkce

- **`EXPLANATION_TEXT`** — konstanta s vysvětlením pojmů; sdílená mezi fragmentem a statickým zobrazením po rerun.
- **`fixed_log(lines)`** — padduje log na `LOG_HEIGHT=16` řádků, blok nemění výšku při tréninku.
- **`predict_with_progress(model_data, group, result_key, mlp_cfg)`** — batch predikce (25 dávek × `predict_delay`), živá konzole + graf % pozitivních predikcí. Ukládá `log_lines` a `pos_history` do `st.session_state[result_key]` pro zobrazení po rerun.
- **`render_krok3(trained_ds, group, neg_label, pos_label, mlp_cfg)`** — dvě fáze:
  - **Fáze 1** (`result_key` není v session_state): tabulka testovacích dat + tlačítko "Předhodit"
  - **Fáze 2** (`result_key` v session_state): data a tlačítko zmizí, zůstane konzole + graf + tabulka predikcí
  - Zobrazuje se pouze pro aktuálně vybraný dataset.
- **`plot_comparison(results: dict, baseline: float)`** — přijímá `{letter: accuracy}` a přidá červenou přerušovanou čáru "Náhodný model" na úrovni `baseline`. Barevně zvýrazní nejlepší model.
- **`majority_baseline(cm)`** — z matice záměn spočítá přesnost naivního modelu vždy predikujícího většinovou třídu: `max(actual_neg, actual_pos) / total`.
- **`translate_cols(df, group)`** — přejmenuje sloupce dle `column_labels` v settings.yaml. `column_labels` bylo odstraněno → funkce vrací df beze změny (anglické názvy).
- **`build_table(records, group, show_true)`** — sestaví tabulku predikcí; `show_true` odhalí sloupec "Správná diagnóza".
- **`render_admin(settings)`** — tabulka skupin s tlačítkem **Vstoupit** pro přímé přihlášení do skupiny bez zadání hesla. Kliknutí nastaví `group_id`/`group` do session_state a smaže `is_admin`.

### UI detaily

- **Přihlášení**: `st.form("login_form")` + `st.form_submit_button` — Enter i kliknutí fungují. `st.text_input` bez `type="password"` — znaky viditelné, prohlížeč nenabídne uložení. Heslo se normalizuje přes `normalize_pwd()` (malá písmena + odstranění diakritiky pomocí `unicodedata`), takže `SRDCE`, `Srdce`, `srdcě` i `srdce` jsou ekvivalentní.
- **Terminologie**: "Epocha" → **"Průchod"**, konzole zobrazuje `✓ Konvergováno v průchodu N.`
- **Krok 1**: čtyři záložky (A–D) s prvními 15 řádky, bez tlačítek ke stažení. Expander **ℹ️ Popis proměnných** zobrazí klinické vysvětlení ze `column_descriptions` v `settings.yaml`.
- **Krok 4**: průběžné porovnání od 2 modelů. `plot_comparison(results, baseline)` přidá červenou čáru "Náhodný model" (`majority_baseline(cm)`) s popisem pod grafem. Plná odpověď až po 4 modelech — zobrazí přehled všech datasetů A–D s typem korupce ze `dataset_corruptions` v `settings.yaml`.

### Plotly grafy — unikátní klíče

Grafy v smyčkách dostávají `key=str(uuid.uuid4())` — bez toho hází `StreamlitDuplicateElementKey`. Statické grafy mají fixní klíče: `f"train_saved_{ds}"`, `f"pred_saved_{trained_ds}"`, `"comparison"`, `f"cm_{trained_ds}"`.

### trainer.py — funkce

```python
preprocess(train_path, test_path, target_column, display_columns)
    → X_train_sc, y_train_enc, X_test_sc, y_test_enc, display_df

build_mlp(mlp_cfg)
    → MLPClassifier(warm_start=True, max_iter=1, ...)  # pro epochový loop v threadu
```

Predikce i tréninkový loop probíhají v `app.py`. Trénink v background threadu, predikce synchronně s placeholdery.

### Požadovaná verze Streamlit

`streamlit>=1.37.0` — nutné pro `@st.fragment(run_every=...)`. Uvedeno v `requirements.txt`.

### Streamlit API konvence

`width='stretch'` místo `use_container_width=True`, `width='content'` místo `use_container_width=False`.

### Barevná tlačítka

CSS injektováno jako konstanta `_BTN_CSS` na začátku `render_dashboard()`. Mechanismus: Streamlit 1.37+ přidává CSS třídu `st-key-{key}` na container každého widgetu s explicitním `key=`. Trénovací tlačítko má `key=f"train_btn_{ds}"` → třída `st-key-train_btn_A` atd. → zelená (#1b5e20). Predikční tlačítko má `key=f"predict_btn_{trained_ds}"` → teal (#006064).

### Typy korupce datasetů

Každá skupina má v `settings.yaml` klíč `dataset_corruptions` — mapování `{letter: typ}` kde typ je `correct`, `shuffled`, `inverted`, nebo `partial`. Zobrazuje se v expanderu správné odpovědi v Kroku 4. Hodnoty odpovídají `_CORRUPTION_ORDER` v `prepare_data.py` — při změně datasetů je nutné oba synchronizovat.

### Popis proměnných a zdroj dat

`column_labels` bylo odstraněno — sloupce jsou zobrazovány s původními anglickými názvy.

`column_descriptions` v `settings.yaml` pokrývá **všechny sloupce** každé skupiny. Formát: `anglický_název: "český popis — English description"`. Zobrazuje se v expanderu **ℹ️ Popis proměnných** v Kroku 1. Krok 1 zobrazuje prvních 100 řádků s poznámkou "pouze náhled".

`data_source` — citace s URL + inline HTML download link (base64 data URL) správného datasetu ve formátu `Data ke stažení: dataset.csv ⬇️`.

Zdroje dat:
| Skupina | Citace |
|---------|--------|
| 1 | Fedesoriano (2021). Heart Failure Prediction. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction — kombinace 5 UCI databází, 918 řádků, sloupce Fedesoriano schématu (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease). Kategoriální sloupce zakódovány číselně v prepare_data.py. Fallback: Cleveland UCI (id=45) s přejmenovanými sloupci. |
| 2 | Smith et al. (1988). Pima Indians Diabetes. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| 3 | Street et al. (1993). Breast Cancer Wisconsin. UCI / sklearn. https://archive.ics.uci.edu/dataset/17 |
| 4 | Chicco & Jurman (2020). UCI Heart Failure. https://archive.ics.uci.edu/dataset/519 |
| 5 | Syntetická data — Strack et al. (2014). https://doi.org/10.1155/2014/781670 |
| 6 | Syntetická data — Reyna et al. (2020). PhysioNet Challenge 2019. https://physionet.org/content/challenge-2019/ |

### Opravené bugy

- **Přerušení tréninku při přepnutí datasetu**: trénink běží v background threadu, `st.rerun()` ho nepřeruší.
- **Tlačítko "Natrénovat" zmizí a nic se neděje**: modul-level `dict = {}` se resetoval při každém `st.rerun()`. Oprava: `@st.cache_resource` na `_get_jobs()` a `_get_lock()`.
- **Epochy skáčou zpět (~80 průchodů)**: souběžná vlákna se stejným `job_key` závodila o zápis. Oprava: `run_id` (UUID) přiřazený každému spuštění; worker se ukončí, pokud `run_id` nesedí (`still_mine()`).
- **Mizející Krok 3**: výsledky v session_state přežijí přepnutí datasetu a zobrazí se zpět.
- **StreamlitDuplicateElementKey**: grafy v smyčkách dostávají `key=str(uuid.uuid4())`.
- **WebSocket rejected**: řešeno přes `enableCORS` v `.streamlit/config.toml` (aktuálně `true`).

## .streamlit/config.toml

```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true      # přeskočí výzvu k zadání emailu
enableCORS = true

[browser]
serverAddress = "biofyzika.valdauf.eu"
serverPort = 443
```

## Reset a UX detaily

- **Tlačítko "Začít znovu"** v sidebaru — smaže pouze tréninkové výsledky (`model_*`, `results_*`, `training_job_*`, `show_true_*`, `dataset_choice`), zachová přihlášení. Užitečné pokud student klikne špatně.
- **Varování v sidebaru**: `⚠️ Obnovení stránky (F5) smaže výsledky.`
- **Legenda matice záměn**: pod každou maticí je `st.caption` vysvětlující SP/SN/FP/FN v kontextu konkrétní diagnózy skupiny. Každá buňka zobrazuje počet i procento z celku (`120\n(48.0 %)`).
- **Popis náhodného modelu**: pod grafem Kroku 4 je `st.caption` vysvětlující, že červená čára = model vždy predikující početnější třídu.
- **TTL cleanup jobů**: `_collect_done_jobs()` maže joby starší 30 minut (`created_at` uložen při startu jobu). Brání hromadění dat z disconnectnutých sessionů.

## Logování přístupů

`access.log` v kořenovém adresáři projektu. `RotatingFileHandler` — max 1 MB, 3 zálohy. Logger `"seminar"` inicializován jednou přes `_setup_logger()` (guard `if log.handlers`).

Výstup jde současně do souboru (`access.log`) i do konzole kde běží Streamlit. Pro potlačení konzole stačí odebrat `log.addHandler(ch)` v `_setup_logger()`.

Logované události:
```
LOGIN  skupina=1  Diagnostika srdeční choroby   # úspěšné přihlášení skupiny
LOGIN  admin                                    # přihlášení admina
LOGIN  FAIL                                     # nesprávné heslo
TRÉNINK  skupina=1  dataset=C                   # spuštění tréninku
TRÉNINK HOTOV  job=…_C  průchodů=187  přesnost=84.2%  # dokončení (z workeru)
PREDIKCE  key=results_c  přesnost=81.3%         # předhození testovacích dat
```

## Patička

`render_footer()` volaná z `main()` — sticky CSS (`position: fixed; bottom: 0`), viditelná na všech stránkách.

## Fixní výška konzole

`LOG_HEIGHT = 16` řádků. `fixed_log(lines)` padduje log na konstantní výšku — blok `st.code()` nemění výšku při tréninku, stránka neposkakuje. `progress_ph` umístěn **pod** sloupci (ne nad nimi) ze stejného důvodu.

## Známé limitace

- **Scrollování při tréninku**: Streamlit re-renderuje celou stránku při každém updatu fragmentu. Zmírněno fixní výškou logu a umístěním progress baru pod sloupce. Úplné odstranění by vyžadovalo jiný framework.

## Možná vylepšení (neimplementováno)

- Zobrazení ROC křivky pro každý model
- Export výsledků do PDF/CSV pro vyučujícího
- Časový limit na seminář (countdown timer)
- Možnost učitele resetovat session všech skupin najednou (aktuálně jen per-student "Začít znovu")
- Distribuce proměnných / korelační matice v Kroku 1
- Diskusní otázky po odhalení odpovědi (záměrně vynecháno — seminář si vede vyučující)
