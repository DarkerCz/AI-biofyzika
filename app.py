import streamlit as st
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import uuid
import threading
import base64
import warnings
warnings.filterwarnings("ignore")

import logging
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from ml.trainer import preprocess, build_mlp

# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logger():
    log = logging.getLogger("seminar")
    if log.handlers:
        return log  # already configured

    fmt_app = logging.Formatter("%(asctime)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
    fmt_st  = logging.Formatter("%(asctime)s  [%(levelname)s] %(name)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

    # Aplikační log (přihlášení, trénink, predikce) — soubor + konzole
    fh_app = RotatingFileHandler("access.log", maxBytes=1_000_000, backupCount=3,
                                 encoding="utf-8")
    fh_app.setFormatter(fmt_app)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt_app)
    log.setLevel(logging.INFO)
    log.addHandler(fh_app)
    log.addHandler(ch)

    # Streamlit systémový log (WARNING+) — připojení, chyby, odpojení
    fh_st = RotatingFileHandler("streamlit.log", maxBytes=1_000_000, backupCount=3,
                                encoding="utf-8")
    fh_st.setFormatter(fmt_st)
    fh_st.setLevel(logging.WARNING)
    for name in ("streamlit", "streamlit.runtime", "tornado.access"):
        l = logging.getLogger(name)
        l.addHandler(fh_st)

    return log

log = _setup_logger()


def normalize_pwd(s: str) -> str:
    """Malá písmena + odstranění diakritiky."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

st.set_page_config(
    page_title="AI v medicíně – Seminář",
    page_icon="⚕️",
    layout="wide",
)

# ── Background training jobs ──────────────────────────────────────────────────
# @st.cache_resource zajistí, že dict a lock přežijí reruns (modul-level
# přiřazení by se při každém rerunu resetovalo na {}).
@st.cache_resource
def _get_jobs() -> dict:
    return {}

@st.cache_resource
def _get_lock() -> threading.Lock:
    return threading.Lock()

LOG_HEIGHT = 16

EXPLANATION_TEXT = (
    "**Průchod** — jedno úplné projití modelu přes všechna trénovací data.\n\n"
    "**Ztráta** *(loss)* — číselná míra toho, jak moc se model mýlí. "
    "Čím nižší, tím lépe.\n\n"
    "**Přesnost** *(accuracy)* — podíl správně klasifikovaných vzorků.\n\n"
    "**Konvergováno** — model přestal zlepšovat svou ztrátu; "
    "trénink byl automaticky zastaven."
)


@st.cache_data
def load_settings():
    with open("settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_session_id():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "default"
    except Exception:
        return "default"


# ── Helpers ───────────────────────────────────────────────────────────────────

def fixed_log(lines):
    tail = lines[-LOG_HEIGHT:]
    padded = tail + [""] * (LOG_HEIGHT - len(tail))
    return "\n".join(padded)


def plot_confusion_matrix(cm, neg_label, pos_label):
    labels = [neg_label, pos_label]
    total = sum(sum(row) for row in cm) or 1
    text = [[f"{cm[i][j]}<br>({cm[i][j]/total*100:.1f} %)" for j in range(2)]
            for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Predikce: {l}" for l in labels],
        y=[f"Skutečně: {l}" for l in labels],
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def plot_loss(history):
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["loss"],
        mode="lines", line=dict(color="#1565C0", width=2)
    ))
    fig.update_layout(
        xaxis_title="Průchod", yaxis_title="Ztráta (loss)",
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def plot_comparison(results: dict, baseline: float = None):
    """results = {"A": accuracy, ...}; baseline = přesnost naivního modelu."""
    letters = sorted(results.keys())
    accs = [results[l] for l in letters]
    best = max(accs)
    colors = ["#1565C0" if a == best else "#90CAF9" for a in accs]
    fig = go.Figure(go.Bar(
        x=[f"Dataset {l}" for l in letters],
        y=[round(a * 100, 1) for a in accs],
        text=[f"{a:.1%}" for a in accs],
        textposition="outside",
        marker_color=colors,
    ))
    if baseline is not None:
        fig.add_hline(
            y=baseline * 100,
            line_dash="dash", line_color="#e53935",
            annotation_text=f"Náhodný model: {baseline:.1%}",
            annotation_position="top right",
        )
    fig.update_layout(
        yaxis_title="Přesnost (%)", yaxis_range=[0, 105],
        height=300, margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


_BTN_CSS = """
<style>
/* Zelené tlačítko tréninku — cíleno přes st-key-train_btn_{ds} */
.st-key-train_btn_A button, .st-key-train_btn_B button,
.st-key-train_btn_C button, .st-key-train_btn_D button {
    background-color: #1b5e20 !important;
    color: white !important;
    border-color: #1b5e20 !important;
}
.st-key-train_btn_A button:hover, .st-key-train_btn_B button:hover,
.st-key-train_btn_C button:hover, .st-key-train_btn_D button:hover {
    background-color: #2e7d32 !important;
    border-color: #2e7d32 !important;
}
/* Teal tlačítko testování — cíleno přes st-key-predict_btn_{ds} */
.st-key-predict_btn_A button, .st-key-predict_btn_B button,
.st-key-predict_btn_C button, .st-key-predict_btn_D button {
    background-color: #006064 !important;
    color: white !important;
    border-color: #006064 !important;
}
.st-key-predict_btn_A button:hover, .st-key-predict_btn_B button:hover,
.st-key-predict_btn_C button:hover, .st-key-predict_btn_D button:hover {
    background-color: #00838f !important;
    border-color: #00838f !important;
}
</style>
"""


def majority_baseline(cm):
    """Přesnost naivního modelu 'vždy hádej většinovou třídu'."""
    total = sum(sum(row) for row in cm)
    return max(cm[0][0] + cm[0][1], cm[1][0] + cm[1][1]) / total if total else 0


def translate_cols(df, group):
    labels = group.get("column_labels", {})
    return df.rename(columns=labels)


def build_table(records, group, show_true):
    df = pd.DataFrame(records)
    pos = group["positive_label"]
    neg = group["negative_label"]
    df["Predikce"] = df["__pred__"].map({0: neg, 1: pos})
    if show_true:
        df["Správná diagnóza"] = df["__true__"].map({0: neg, 1: pos})
    df = df.drop(columns=["__pred__", "__true__"])
    df = df.rename(columns={"__prob__": "Jistota (%)"})
    return df


# ── Background training ───────────────────────────────────────────────────────

def _training_worker(job_key, run_id, X_train_sc, y_train_enc, X_test_sc,
                     y_test_enc, display_df, mlp_cfg):
    max_epochs = mlp_cfg.get("max_iter", 400)
    n_iter_no_change = mlp_cfg.get("n_iter_no_change", 60)
    epoch_delay = mlp_cfg.get("epoch_delay", 0.3)

    def still_mine():
        """Vrátí False pokud byl job nahrazen novějším spuštěním nebo zrušen."""
        j = _get_jobs().get(job_key, {})
        return j.get("run_id") == run_id and not j.get("cancelled")

    mlp = build_mlp(mlp_cfg)
    history = []
    log_lines = ["Spouštím trénink...", ""]
    best_loss = float("inf")
    no_change = 0
    final_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        with _get_lock():
            if not still_mine():
                return

        mlp.fit(X_train_sc, y_train_enc)
        loss = mlp.loss_
        acc = mlp.score(X_train_sc, y_train_enc)
        history.append({"epoch": epoch, "loss": loss})

        time.sleep(epoch_delay)

        log_lines.append(
            f"[{epoch:4d}/{max_epochs}]  ztráta: {loss:.4f}  |  přesnost: {acc:.1%}"
        )

        with _get_lock():
            if not still_mine():
                return
            _get_jobs()[job_key].update({
                "epoch": epoch,
                "loss": loss,
                "history": list(history),
                "log_lines": list(log_lines),
            })

        if loss < best_loss - 1e-4:
            best_loss = loss
            no_change = 0
        else:
            no_change += 1
        if no_change >= n_iter_no_change:
            final_epoch = epoch
            break

    final_acc = mlp.score(X_train_sc, y_train_enc)
    log_lines += ["", f"✓ Konvergováno v průchodu {final_epoch}.",
                  f"  Přesnost na trénovacích datech: {final_acc:.1%}"]

    with _get_lock():
        if not still_mine():
            return
        log.info(f"TRÉNINK HOTOV  job={job_key}  průchodů={final_epoch}  přesnost={final_acc:.1%}")
        _get_jobs()[job_key].update({
            "running": False,
            "done": True,
            "mlp": mlp,
            "X_test_sc": X_test_sc,
            "y_test_enc": y_test_enc,
            "display_df": display_df,
            "n_train": len(y_train_enc),
            "n_iter": final_epoch,
            "train_acc": final_acc,
            "history": list(history),
            "log_lines": list(log_lines),
        })


def run_training(ds, data_path, group, mlp_cfg):
    session_id = get_session_id()
    job_key = f"{session_id}_{ds}"
    run_id = str(uuid.uuid4())
    display_cols = group.get("display_columns")
    target_col = group["target_column"]

    # Zruš případné starší vlákno se stejným klíčem
    with _get_lock():
        if job_key in _get_jobs():
            _get_jobs()[job_key]["cancelled"] = True

    with st.spinner("Načítám a připravuji data..."):
        X_train_sc, y_train_enc, X_test_sc, y_test_enc, display_df = preprocess(
            data_path / f"dataset_{ds}.csv",
            data_path / "test.csv",
            target_col,
            display_cols,
        )

    with _get_lock():
        _get_jobs()[job_key] = {
            "run_id": run_id,
            "running": True, "done": False, "cancelled": False,
            "epoch": 0, "loss": None,
            "max_epochs": mlp_cfg.get("max_iter", 400),
            "n_train": len(y_train_enc), "n_test": len(y_test_enc),
            "history": [], "log_lines": ["Spouštím trénink...", ""],
            "created_at": time.time(),
        }

    log.info(f"TRÉNINK  skupina={st.session_state.get('group_id', '?')}  dataset={ds}")
    st.session_state[f"training_job_{ds.lower()}"] = job_key

    threading.Thread(
        target=_training_worker,
        args=(job_key, run_id, X_train_sc, y_train_enc, X_test_sc, y_test_enc,
              display_df, mlp_cfg),
        daemon=True,
    ).start()


def _collect_done_jobs():
    """Přesune dokončené joby do session_state; smaže joby starší 30 minut."""
    now = time.time()
    with _get_lock():
        stale = [k for k, j in _get_jobs().items()
                 if now - j.get("created_at", now) > 1800]
        for k in stale:
            _get_jobs().pop(k, None)

    for ds_check in ["A", "B", "C", "D"]:
        job_key = st.session_state.get(f"training_job_{ds_check.lower()}")
        if not job_key:
            continue
        with _get_lock():
            job = _get_jobs().get(job_key, {})
            if not job.get("done"):
                continue
            model_key = f"model_{ds_check.lower()}"
            if model_key not in st.session_state:
                st.session_state[model_key] = {
                    "mlp": job["mlp"],
                    "X_test_sc": job["X_test_sc"],
                    "y_test_enc": job["y_test_enc"],
                    "display_df": job["display_df"],
                    "n_train": job["n_train"],
                    "n_iter": job["n_iter"],
                    "train_acc": job["train_acc"],
                    "history": job["history"],
                    "log_lines": job["log_lines"],
                }
            _get_jobs().pop(job_key, None)
        del st.session_state[f"training_job_{ds_check.lower()}"]


@st.fragment(run_every="0.5s")
def show_training_fragment(ds):
    job_key = st.session_state.get(f"training_job_{ds.lower()}")
    if not job_key:
        return

    with _get_lock():
        job = dict(_get_jobs().get(job_key, {}))

    if not job:
        return

    epoch = job.get("epoch", 0)
    max_epochs = job.get("max_epochs", 400)
    history = job.get("history", [])
    log_lines = job.get("log_lines", [])

    col_log, col_chart = st.columns([1, 1])
    with col_log:
        st.markdown(EXPLANATION_TEXT)
        st.code(fixed_log(log_lines), language=None)
    with col_chart:
        if history:
            st.plotly_chart(plot_loss(history), use_container_width=True,
                            key=str(uuid.uuid4()))

    if job.get("running"):
        loss_str = f" · ztráta: {job['loss']:.4f}" if job.get("loss") else ""
        st.progress(
            min(epoch / max_epochs, 1.0),
            f"Průchod {epoch}/{max_epochs}{loss_str}"
        )
    elif job.get("done"):
        st.progress(1.0, f"✓ Trénink dokončen – {job.get('n_iter', epoch)} průchodů")
        st.rerun()  # full rerun → _collect_done_jobs přesune model do session_state


# ── Predikce s live konzolí ───────────────────────────────────────────────────

def predict_with_progress(model_data, group, result_key, mlp_cfg):
    mlp = model_data["mlp"]
    X_test_sc = model_data["X_test_sc"]
    y_test_enc = model_data["y_test_enc"]
    display_df = model_data["display_df"]
    n_test = len(y_test_enc)
    pos_label = group["positive_label"]
    neg_label = group["negative_label"]
    predict_delay = mlp_cfg.get("predict_delay", 1.2)
    predict_batches = mlp_cfg.get("predict_batches", 25)

    batch_size = max(10, n_test // predict_batches)

    col_log, col_chart = st.columns([1, 1])
    log_ph = col_log.empty()
    chart_ph = col_chart.empty()
    progress_ph = st.progress(0, "Zahajuji analýzu...")

    log_lines = ["Načítám testovací pacienty...", ""]
    all_preds, all_probs = [], []
    pos_history = []

    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        preds = mlp.predict(X_test_sc[i:end])
        probs = mlp.predict_proba(X_test_sc[i:end])[:, 1]
        all_preds.extend(preds)
        all_probs.extend(probs)

        pos = int(np.sum(np.array(all_preds) == 1))
        neg = len(all_preds) - pos
        pos_history.append({"pacient": end,
                             "% pozitivních": round(pos / len(all_preds) * 100, 1)})

        log_lines.append(
            f"Pacienti {i+1:4d}–{end:4d}/{n_test}  |  "
            f"{pos_label}: {pos}   {neg_label}: {neg}"
        )
        time.sleep(predict_delay)
        log_ph.code(fixed_log(log_lines), language=None)

        fig = go.Figure(go.Scatter(
            x=[h["pacient"] for h in pos_history],
            y=[h["% pozitivních"] for h in pos_history],
            mode="lines", line=dict(color="#1565C0", width=2)
        ))
        fig.update_layout(
            xaxis_title="Pacient", yaxis_title="% pozitivních predikcí",
            yaxis_range=[0, 100], height=280,
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        chart_ph.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
        progress_ph.progress(end / n_test, f"Predikuji... {end}/{n_test} pacientů")

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    pos_total = int(np.sum(y_pred == 1))
    neg_total = n_test - pos_total

    log_lines += ["", f"✓ Hotovo. Celkem {n_test} pacientů.",
                  f"  {pos_label}: {pos_total}   {neg_label}: {neg_total}"]
    log_ph.code(fixed_log(log_lines), language=None)
    progress_ph.progress(1.0, f"✓ Predikce dokončena – {n_test} pacientů")

    cm = confusion_matrix(y_test_enc, y_pred, labels=[0, 1]).tolist()
    table_df = display_df.copy()
    table_df["__pred__"] = y_pred
    table_df["__prob__"] = (y_prob * 100).round(1)
    table_df["__true__"] = y_test_enc

    acc = float(accuracy_score(y_test_enc, y_pred))
    log.info(f"PREDIKCE  key={result_key}  přesnost={acc:.1%}")
    st.session_state[result_key] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "n_test": n_test,
        "table": table_df.to_dict("records"),
        "log_lines": list(log_lines),
        "pos_history": list(pos_history),
    }


# ── Krok 3 ────────────────────────────────────────────────────────────────────

def render_krok3(trained_ds, group, neg_label, pos_label, mlp_cfg):
    model_key = f"model_{trained_ds.lower()}"
    result_key = f"results_{trained_ds.lower()}"
    m = st.session_state[model_key]
    has_results = result_key in st.session_state

    st.markdown("---")

    if not has_results:
        st.header(f"Krok 3 – Předložte modelu testovací pacienty (Dataset {trained_ds})")
        st.markdown(
            f"Níže vidíte **{len(m['display_df'])} testovacích pacientů** — "
            f"model trénovaný na Dataset {trained_ds} je nikdy neviděl. "
            "Kliknutím mu je předložte a sledujte, jak je klasifikuje."
        )
        st.dataframe(translate_cols(m["display_df"], group),
                     width='stretch', height=300)

        if st.button(f"🔬 Předhodit modelu testovací data (Dataset {trained_ds})",
                     width='stretch', key=f"predict_btn_{trained_ds}"):
            predict_with_progress(m, group, result_key, mlp_cfg)
            st.session_state[f"show_true_{trained_ds.lower()}"] = False
            st.rerun()
    else:
        res = st.session_state[result_key]
        st.header(f"Krok 3 – Výsledky predikce (Dataset {trained_ds})")

        col1, col2 = st.columns(2)
        col1.metric("Přesnost na testovacích datech", f"{res['accuracy']:.1%}")
        col2.metric("Testovací pacienti", res["n_test"])

        col_log, col_chart = st.columns([1, 1])
        with col_log:
            st.code(fixed_log(res["log_lines"]), language=None)
        with col_chart:
            ph = res["pos_history"]
            fig = go.Figure(go.Scatter(
                x=[h["pacient"] for h in ph],
                y=[h["% pozitivních"] for h in ph],
                mode="lines", line=dict(color="#1565C0", width=2)
            ))
            fig.update_layout(
                xaxis_title="Pacient", yaxis_title="% pozitivních predikcí",
                yaxis_range=[0, 100], height=280,
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"pred_saved_{trained_ds}")

        with st.expander("Matice záměn"):
            st.plotly_chart(
                plot_confusion_matrix(res["confusion_matrix"], neg_label, pos_label),
                use_container_width=True, key=f"cm_{trained_ds}",
            )
            st.caption(
                f"**SP** (správně pozitivní / TP): model správně predikoval *{pos_label}* · "
                f"**SN** (správně negativní / TN): model správně predikoval *{neg_label}* · "
                f"**FP** (falešný poplach): predikoval *{pos_label}*, ale skutečnost byla *{neg_label}* · "
                f"**FN** (přehlédnutý případ): predikoval *{neg_label}*, ale skutečnost byla *{pos_label}*"
            )

        show_true = st.session_state.get(f"show_true_{trained_ds.lower()}", False)
        table_df = translate_cols(build_table(res["table"], group, show_true), group)

        st.dataframe(
            table_df, width='stretch', height=440,
            column_config={
                "Jistota (%)": st.column_config.ProgressColumn(
                    "Jistota (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        if not show_true:
            if st.button("🔎 Zobrazit správné diagnózy", key=f"reveal_{trained_ds}"):
                st.session_state[f"show_true_{trained_ds.lower()}"] = True
                st.rerun()
        else:
            st.success("Sloupec **Správná diagnóza** přidán — porovnejte s předpověďmi.")
            if st.button("Skrýt správné diagnózy", key=f"hide_{trained_ds}"):
                st.session_state[f"show_true_{trained_ds.lower()}"] = False
                st.rerun()


# ── Login ─────────────────────────────────────────────────────────────────────

def render_login(settings):
    st.title("⚕️ Seminář: AI v medicíně")
    st.markdown("---")
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.subheader("Přihlášení skupiny")
        with st.form("login_form"):
            pwd = st.text_input("Zadejte heslo vaší skupiny:")
            submitted = st.form_submit_button("Přihlásit se", width='stretch')
        if submitted:
            pwd_norm = normalize_pwd(pwd)
            for gid, grp in settings["groups"].items():
                if pwd_norm == normalize_pwd(grp["password"]):
                    log.info(f"LOGIN  skupina={gid}  {grp['topic']}")
                    st.session_state.group_id = gid
                    st.session_state.group = grp
                    st.rerun()
            if pwd_norm == normalize_pwd(settings.get("admin_password", "")):
                log.info("LOGIN  admin")
                st.session_state.is_admin = True
                st.rerun()
            log.info("LOGIN  FAIL")
            st.error("Nesprávné heslo.")


# ── Admin ─────────────────────────────────────────────────────────────────────

def render_admin(settings):
    with st.sidebar:
        st.title("⚕️ Admin")
        st.markdown("---")
        if st.button("Odhlásit se"):
            st.session_state.clear(); st.rerun()

    st.title("🔑 Administrace semináře")

    header = st.columns([1, 4, 2, 2, 2])
    for col, label in zip(header, ["Skupina", "Téma", "Heslo", "Správný dataset", ""]):
        col.markdown(f"**{label}**")
    st.markdown("---")

    for gid, grp in settings["groups"].items():
        col1, col2, col3, col4, col5 = st.columns([1, 4, 2, 2, 2])
        col1.write(f"Skupina {gid}")
        col2.write(grp["topic"])
        col3.write(f"`{grp['password']}`")
        col4.write(f"Dataset **{grp['correct_dataset']}**")
        if col5.button("Vstoupit", key=f"admin_login_{gid}"):
            st.session_state.pop("is_admin", None)
            st.session_state.group_id = gid
            st.session_state.group = grp
            st.rerun()

    st.markdown("---")
    st.info("Nastavení upravujte v souboru `settings.yaml`.")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def render_dashboard(settings):
    group = st.session_state.group
    gid = st.session_state.group_id
    data_path = Path(f"data/group{gid}")
    mlp_cfg = settings["mlp"]
    pos_label = group["positive_label"]
    neg_label = group["negative_label"]

    # Přesun dokončených background jobů do session_state
    _collect_done_jobs()

    with st.sidebar:
        st.title("⚕️ AI v medicíně")
        st.markdown(f"**Skupina {gid}** · {group['name']}")
        st.markdown(f"*{group['topic']}*")
        st.markdown("---")
        if st.button("🔄 Začít znovu", width='stretch'):
            reset_keys = [k for k in st.session_state
                          if any(k.startswith(p) for p in
                                 ["model_", "results_", "training_job_",
                                  "show_true_", "dataset_choice"])]
            for k in reset_keys:
                del st.session_state[k]
            st.rerun()
        if st.button("Odhlásit se", width='stretch'):
            st.session_state.clear(); st.rerun()
        st.caption("⚠️ Obnovení stránky (F5) smaže výsledky.")

    st.markdown(_BTN_CSS, unsafe_allow_html=True)
    st.title(group["topic"])
    st.info(group["description"])

    if not data_path.exists():
        st.error("Data nebyla připravena. Spusťte: `python prepare_data.py`")
        return

    # ── KROK 1 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Krok 1 – Prozkoumejte datasety")
    st.markdown(
        "Všechny čtyři datasety mají **stejné proměnné** — nevíte, který má správné štítky."
    )

    all_ds = ["A", "B", "C", "D"]
    ds_files = {d: pd.read_csv(data_path / f"dataset_{d}.csv") for d in all_ds}

    n_total = len(ds_files["A"])
    tabs = st.tabs([f"Dataset {d}  ({n_total} záznamů)" for d in all_ds])
    for tab, d in zip(tabs, all_ds):
        with tab:
            st.dataframe(translate_cols(ds_files[d].head(100), group),
                         width='stretch')
            st.caption(f"Zobrazeno prvních 100 z {n_total} záznamů — pouze náhled.")

    data_source = group.get("data_source")
    if data_source:
        correct_ds = group["correct_dataset"]
        csv_b64 = base64.b64encode(
            ds_files[correct_ds].to_csv(index=False).encode("utf-8")
        ).decode()
        st.markdown(
            f"Zdroj dat: {data_source}  \n"
            f'Data ke stažení: <a href="data:text/csv;base64,{csv_b64}" '
            f'download="dataset.csv">dataset.csv ⬇️</a>',
            unsafe_allow_html=True,
        )

    descs = group.get("column_descriptions")
    if descs:
        with st.expander("ℹ️ Popis proměnných"):
            rows = [{"Proměnná": col, "Popis": desc}
                    for col, desc in descs.items()]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── KROK 2 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Krok 2 – Natrénujte model")
    st.markdown("Vyberte dataset a spusťte trénink neuronové sítě.")

    if "dataset_choice" not in st.session_state:
        st.session_state.dataset_choice = "A"
    ds = st.session_state.dataset_choice
    model_key = f"model_{ds.lower()}"
    job_key_active = st.session_state.get(f"training_job_{ds.lower()}")

    sel_cols = st.columns(4)
    for col, d in zip(sel_cols, ["A", "B", "C", "D"]):
        with col:
            if st.button(f"Dataset {d}", width='stretch',
                         type="primary" if ds == d else "secondary"):
                st.session_state.dataset_choice = d
                st.rerun()

    # Indikátor tréninků na pozadí pro ostatní datasety
    bg_msgs = []
    for other in ["A", "B", "C", "D"]:
        if other == ds:
            continue
        other_job = st.session_state.get(f"training_job_{other.lower()}")
        if other_job and _get_jobs().get(other_job, {}).get("running"):
            ep = _get_jobs().get(other_job, {}).get("epoch", 0)
            mx = _get_jobs().get(other_job, {}).get("max_epochs", 400)
            bg_msgs.append(f"Dataset {other} (průchod {ep}/{mx})")
    if bg_msgs:
        st.caption("⏳ Trénují se na pozadí: " + ", ".join(bg_msgs))

    # Tlačítko spustit trénink — skryj pokud právě běží
    if not job_key_active:
        if st.button(f"🧠 Natrénovat model na Dataset {ds}",
                     width='stretch', key=f"train_btn_{ds}"):
            run_training(ds, data_path, group, mlp_cfg)
            st.rerun()

    # Výsledky tréninku
    if model_key in st.session_state:
        m = st.session_state[model_key]
        col1, col2 = st.columns(2)
        col1.metric("Průchodů", m["n_iter"])
        col2.metric("Přesnost na trénovacích datech", f"{m['train_acc']:.1%}")
        col_log, col_chart = st.columns([1, 1])
        with col_log:
            st.markdown(EXPLANATION_TEXT)
            st.code(fixed_log(m["log_lines"]), language=None)
        with col_chart:
            st.plotly_chart(plot_loss(m["history"]), use_container_width=True,
                            key=f"train_saved_{ds}")

    elif job_key_active:
        # Trénink běží na pozadí — živý fragment
        show_training_fragment(ds)

    # ── KROK 3 ────────────────────────────────────────────────────────────────
    if model_key in st.session_state:
        render_krok3(ds, group, neg_label, pos_label, mlp_cfg)

    # ── KROK 4 ────────────────────────────────────────────────────────────────
    done_results = {d: st.session_state[f"results_{d.lower()}"]["accuracy"]
                    for d in ["A", "B", "C", "D"]
                    if f"results_{d.lower()}" in st.session_state}

    if len(done_results) >= 2:
        st.markdown("---")
        tested = sorted(done_results.keys())
        all_done = len(done_results) == 4
        st.header("Krok 4 – Porovnání modelů" + (" a závěr" if all_done else ""))

        first_cm = st.session_state[
            f"results_{tested[0].lower()}"
        ]["confusion_matrix"]
        baseline = majority_baseline(first_cm)
        st.plotly_chart(plot_comparison(done_results, baseline),
                        use_container_width=True, key="comparison")
        st.caption(
            f"Červená čára — **náhodný model** ({baseline:.1%}): model, který vždy "
            f"předpovídá početnější diagnózu bez ohledu na data pacientů. "
            f"Pokud váš model tuto hranici nepřekonává, je horší než prosté hádání."
        )

        best_ds = max(done_results, key=done_results.get)
        best_acc = done_results[best_ds]
        if not all_done:
            remaining = [d for d in ["A", "B", "C", "D"] if d not in done_results]
            st.info(f"Zatím otestováno {len(done_results)}/4 modelů. "
                    f"Zbývá: Dataset {', '.join(remaining)}.")
        else:
            others = {d: a for d, a in done_results.items() if d != best_ds}
            worst_ds = min(others, key=others.get)
            st.success(
                f"**Model {best_ds}** dosáhl nejvyšší přesnosti **{best_acc:.1%}**. "
                "Který dataset byl tedy správně označen?"
            )

            with st.expander("🔍 Zobrazit správnou odpověď (až po skupinové diskuzi!)"):
                correct = group["correct_dataset"]
                corruptions = group.get("dataset_corruptions", {})
                labels = {
                    "correct":  "✅ správná data",
                    "shuffled": "🔀 přeházené štítky — náhodný shuffle, model se nic nenaučí",
                    "inverted": "🔁 invertované štítky — každý 0↔1, model se naučí opak",
                    "partial":  "⚠️ 30 % štítků překlopeno — model se naučil jen část vzorů",
                }
                st.markdown(f"### ✅ Správný dataset byl: **Dataset {correct}**")
                st.markdown("**Přehled datasetů:**")
                for ds_letter in ["A", "B", "C", "D"]:
                    ctype = corruptions.get(ds_letter, "?")
                    desc = labels.get(ctype, ctype)
                    st.markdown(f"- **Dataset {ds_letter}**: {desc}")
                st.markdown(
                    "\n> **AI model je jen tak dobrý, jako data, na kterých byl naučen.**"
                )


# ── Footer ────────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown("""
<style>
.footer-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: white; border-top: 1px solid #e6e6e6;
    padding: 5px 0; text-align: center;
    color: #aaa; font-size: 0.8rem; z-index: 999;
}
.main .block-container { padding-bottom: 2.5rem; }
</style>
<div class="footer-bar">Jaroslav Valdauf &nbsp;·&nbsp; 3. LF UK &nbsp;·&nbsp; 2026</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()
    if "is_admin" in st.session_state:
        render_admin(settings)
    elif "group_id" in st.session_state:
        render_dashboard(settings)
    else:
        render_login(settings)
    render_footer()


if __name__ == "__main__":
    main()
