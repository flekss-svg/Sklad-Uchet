"""
Решение задачи поиска аномальных респондентов в активности поисковых запросов SoS.

Алгоритм:
---------
Для каждой тройки (BrandID, CategoryDelivery, researchdate) вычисляем daily_ots
каждого респондента. Затем применяем два дополняющих друг друга критерия:

1. Robust Z-score (Modified Z-score на основе MAD):
   score = 0.6745 * (ots - median(ots)) / MAD(ots)
   Порог: score > 3.5 (стандартный порог для modified z-score).
   Метод устойчив к выбросам и не чувствителен к единичным аномалиям.

2. Доля OTS в бренд-дне:
   share = daily_ots_i / sum(daily_ots) в этот день для бренда
   Порог: доля > 0.5 при n_respondents >= MIN_RESPONDENTS_FOR_SHARE.
   Ловит случаи, когда один человек "забирает" больше половины всего OTS бренда.

Малый OTS сам по себе НЕ является причиной удаления.
Аномалия = только чрезмерно высокий OTS.

Единица удаления: (SubjectID, researchdate) — удаляем весь день респондента.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ─── Параметры алгоритма ─────────────────────────────────────────────────────
DATA_DIR = "data_train"          # папка с parquet-файлами
OUTPUT_DIR = "output"            # куда писать результаты

# Robust Z-score
ZSCORE_THRESHOLD = 3.5           # стандартный порог modified z-score
MIN_RESPONDENTS_ZSCORE = 5       # минимум респондентов в группе для применения z-score

# Доля OTS
SHARE_THRESHOLD = 0.50           # если один респондент занимает >50% OTS бренда за день
MIN_RESPONDENTS_SHARE = 5        # минимум респондентов для применения критерия доли


def load_data(data_dir: str) -> pd.DataFrame:
    """Загружает все parquet-файлы из директории."""
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"Parquet-файлы не найдены в {data_dir}")
    print(f"  Найдено {len(files)} parquet-файлов...")
    dfs = [pd.read_parquet(f) for f in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Загружено {len(df):,} строк, {df['SubjectID'].nunique():,} уникальных респондентов")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует и приводит типы данных."""
    # Оставляем только строки из поставки SoS с непустой CategoryDelivery
    df = df[(df["BrandinDelivery"] == 1) &
            (df["CategoryNameDelivery"].notna()) &
            (df["CategoryNameDelivery"] != "")].copy()

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["researchdate"] = pd.to_datetime(df["researchdate"])
    df = df.dropna(subset=["Weight", "researchdate", "SubjectID", "BrandID"])
    print(f"  После фильтрации: {len(df):,} строк")
    return df


def compute_daily_ots(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет daily_ots = Weight * count_rows для каждой тройки (SubjectID, BrandID, researchdate)."""
    agg = (df.groupby(
                ["SubjectID", "researchdate", "BrandID", "CategoryNameDelivery", "Brand"],
                sort=False)
             .agg(count_rows=("QueryText", "count"),
                  Weight=("Weight", "first"))
             .reset_index())
    agg["daily_ots"] = agg["Weight"] * agg["count_rows"]
    return agg


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Modified Z-score (Iglewicz & Hoaglin, 1993). Устойчив к выбросам."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        # При нулевом MAD используем среднее+std как запасной вариант
        std = values.std()
        if std == 0:
            return np.zeros(len(values))
        return 0.6745 * np.abs(values - med) / (std * 1.4826)
    return 0.6745 * (values - med) / mad


def detect_anomalies(daily_ots: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет оба критерия и возвращает DataFrame с аномалиями.
    Возвращает: SubjectID, researchdate, BrandID, Brand, CategoryDelivery,
                daily_ots, score, threshold, reason
    """
    results = []

    # Группируем по (BrandID, CategoryNameDelivery, researchdate) — "бренд-день"
    groups = daily_ots.groupby(
        ["BrandID", "CategoryNameDelivery", "researchdate"], sort=False)

    for (brand_id, cat, date), grp in groups:
        n = len(grp)
        ots_vals = grp["daily_ots"].values
        brand_name = grp["Brand"].iloc[0]

        # ─── Критерий 1: Robust Z-score ───────────────────────────────────
        if n >= MIN_RESPONDENTS_ZSCORE:
            z_scores = robust_zscore(ots_vals)
            anomaly_mask = z_scores > ZSCORE_THRESHOLD
            # Дополнительное условие: OTS должен быть выше медианы (только высокие)
            above_median = ots_vals > np.median(ots_vals)
            anomaly_mask = anomaly_mask & above_median

            for idx, (is_anom, z_sc) in enumerate(zip(anomaly_mask, z_scores)):
                if is_anom:
                    row = grp.iloc[idx]
                    results.append({
                        "SubjectID": row["SubjectID"],
                        "researchdate": date,
                        "BrandID": brand_id,
                        "Brand": brand_name,
                        "CategoryDelivery": cat,
                        "daily_ots": row["daily_ots"],
                        "score": round(float(z_sc), 4),
                        "threshold": ZSCORE_THRESHOLD,
                        "reason": f"robust_zscore={z_sc:.2f} > {ZSCORE_THRESHOLD} (бренд-день n={n})"
                    })

        # ─── Критерий 2: Доля в бренд-дне ────────────────────────────────
        if n >= MIN_RESPONDENTS_SHARE:
            total_ots = ots_vals.sum()
            if total_ots > 0:
                shares = ots_vals / total_ots
                for idx, share in enumerate(shares):
                    if share > SHARE_THRESHOLD:
                        row = grp.iloc[idx]
                        subj = row["SubjectID"]
                        # Добавляем только если ещё не добавлен по z-score (или добавляем с другим reason)
                        results.append({
                            "SubjectID": subj,
                            "researchdate": date,
                            "BrandID": brand_id,
                            "Brand": brand_name,
                            "CategoryDelivery": cat,
                            "daily_ots": row["daily_ots"],
                            "score": round(float(share), 4),
                            "threshold": SHARE_THRESHOLD,
                            "reason": (f"доля_в_ОТС={share:.2%} > {SHARE_THRESHOLD:.0%} "
                                       f"от суммарного OTS бренда за день (n={n})")
                        })

    reasons_df = pd.DataFrame(results)
    return reasons_df


def build_anomalies_csv(reasons_df: pd.DataFrame) -> pd.DataFrame:
    """Схлопывает reasons до уникальных пар (SubjectID, researchdate)."""
    if reasons_df.empty:
        return pd.DataFrame(columns=["SubjectID", "researchdate"])
    anomalies = (reasons_df[["SubjectID", "researchdate"]]
                 .drop_duplicates()
                 .sort_values(["researchdate", "SubjectID"])
                 .reset_index(drop=True))
    return anomalies


def compute_ots_totals(daily_ots: pd.DataFrame,
                       anomalies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Считает суммарный OTS по дням до и после удаления."""
    # До удаления
    before = (daily_ots.groupby("researchdate")["daily_ots"]
              .sum().reset_index().rename(columns={"daily_ots": "ots_before"}))

    # После удаления
    anom_keys = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    mask_clean = ~daily_ots.apply(
        lambda r: (r["SubjectID"], r["researchdate"]) in anom_keys, axis=1)
    after = (daily_ots[mask_clean].groupby("researchdate")["daily_ots"]
             .sum().reset_index().rename(columns={"daily_ots": "ots_after"}))

    merged = before.merge(after, on="researchdate", how="left").fillna(0)
    merged = merged.sort_values("researchdate")
    return merged, daily_ots[mask_clean]


def compute_category_ots_change(daily_ots: pd.DataFrame,
                                anomalies: pd.DataFrame) -> pd.DataFrame:
    """Изменение суммарного OTS по CategoryDelivery в процентах."""
    anom_keys = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    mask_clean = ~daily_ots.apply(
        lambda r: (r["SubjectID"], r["researchdate"]) in anom_keys, axis=1)

    before_cat = (daily_ots.groupby("CategoryNameDelivery")["daily_ots"]
                  .sum().reset_index().rename(columns={"daily_ots": "ots_before"}))
    after_cat = (daily_ots[mask_clean].groupby("CategoryNameDelivery")["daily_ots"]
                 .sum().reset_index().rename(columns={"daily_ots": "ots_after"}))

    cat_df = before_cat.merge(after_cat, on="CategoryNameDelivery", how="left").fillna(0)
    cat_df["pct_change"] = (cat_df["ots_after"] - cat_df["ots_before"]) / cat_df["ots_before"] * 100
    return cat_df.sort_values("pct_change")


def compute_daily_anomaly_count(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Количество аномальных респондентов по дням."""
    if anomalies.empty:
        return pd.DataFrame(columns=["researchdate", "n_anomalous"])
    counts = (anomalies.groupby("researchdate")["SubjectID"]
              .count().reset_index().rename(columns={"SubjectID": "n_anomalous"}))
    return counts.sort_values("researchdate")


# ─── Графики ─────────────────────────────────────────────────────────────────

def plot_total_ots_before_after(ots_daily: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ots_daily["researchdate"], ots_daily["ots_before"] / 1e6,
            label="До удаления", color="#2563EB", linewidth=1.5)
    ax.plot(ots_daily["researchdate"], ots_daily["ots_after"] / 1e6,
            label="После удаления", color="#16A34A", linewidth=1.5, linestyle="--")
    ax.fill_between(ots_daily["researchdate"],
                    ots_daily["ots_after"] / 1e6,
                    ots_daily["ots_before"] / 1e6,
                    alpha=0.15, color="#EF4444", label="Удалённый OTS")
    ax.set_title("Суммарный OTS по дням: до и после очистки аномалий", fontsize=14)
    ax.set_xlabel("Дата")
    ax.set_ylabel("OTS (млн)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Сохранено: {out_path}")


def plot_category_ots_change(cat_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(12, max(5, len(cat_df) * 0.4)))
    colors = ["#EF4444" if v < -5 else "#F59E0B" if v < 0 else "#16A34A"
              for v in cat_df["pct_change"]]
    bars = ax.barh(cat_df["CategoryNameDelivery"], cat_df["pct_change"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Изменение OTS по CategoryDelivery после очистки (%)", fontsize=13)
    ax.set_xlabel("Изменение OTS (%)")
    for bar, val in zip(bars, cat_df["pct_change"]):
        ax.text(val - 0.1 if val < 0 else val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="right" if val < 0 else "left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Сохранено: {out_path}")


def plot_daily_anomaly_count(counts: pd.DataFrame, out_path: str):
    if counts.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Аномалий не найдено", ha="center", va="center", fontsize=16)
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(counts["researchdate"], counts["n_anomalous"],
           width=0.8, color="#7C3AED", alpha=0.8)
    ax.set_title("Количество аномальных респондентов по дням", fontsize=14)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Кол-во аномальных субъектов")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Сохранено: {out_path}")


# ─── Аналитические функции ────────────────────────────────────────────────────

def plot_demography_before_after(df_raw: pd.DataFrame, anomalies: pd.DataFrame,
                                 column: str, out_dir: str):
    """График до/после по социально-демографическим характеристикам."""
    anom_keys = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    mask_clean = ~df_raw.apply(
        lambda r: (r["SubjectID"], r["researchdate"]) in anom_keys, axis=1)
    before = df_raw.groupby(column)["Weight"].sum()
    after = df_raw[mask_clean].groupby(column)["Weight"].sum()
    combined = pd.DataFrame({"До": before, "После": after}).fillna(0)
    combined = combined.sort_values("До", ascending=False).head(15)
    ax = combined.plot(kind="bar", figsize=(12, 5), color=["#2563EB", "#16A34A"])
    plt.title(f"Суммарный вес до/после по: {column}", fontsize=13)
    plt.xlabel(column)
    plt.ylabel("Суммарный Weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, f"demography_{column.replace('/', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Сохранено: {path}")


def show_anomalous_queries(df_raw: pd.DataFrame, subject_id: int,
                           date: str) -> pd.DataFrame:
    """Возвращает все поисковые запросы аномального респондента за день."""
    date_dt = pd.to_datetime(date)
    mask = (df_raw["SubjectID"] == subject_id) & (df_raw["researchdate"] == date_dt)
    return df_raw[mask][["QueryText", "Brand", "CategoryNameDelivery",
                          "ResourceName", "BrandID"]].reset_index(drop=True)


def plot_brand_ots_over_time(daily_ots: pd.DataFrame, anomalies: pd.DataFrame,
                              brand_id: str, out_dir: str):
    """График OTS по дням для конкретного бренда до и после очистки."""
    brand_data = daily_ots[daily_ots["BrandID"] == brand_id].copy()
    brand_name = brand_data["Brand"].iloc[0] if len(brand_data) > 0 else brand_id

    anom_keys = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    mask_clean = ~brand_data.apply(
        lambda r: (r["SubjectID"], r["researchdate"]) in anom_keys, axis=1)

    before = brand_data.groupby("researchdate")["daily_ots"].sum()
    after = brand_data[mask_clean].groupby("researchdate")["daily_ots"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(before.index, before.values / 1e3, label="До", color="#2563EB")
    ax.plot(after.index, after.values / 1e3, label="После", color="#16A34A", linestyle="--")
    ax.set_title(f"OTS бренда '{brand_name}' (ID={brand_id}) до/после очистки")
    ax.set_ylabel("OTS (тыс.)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"brand_ots_{brand_id}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Сохранено: {path}")


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Поиск аномальных респондентов SoS")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n[1/6] Загрузка данных...")
    df_raw = load_data(DATA_DIR)

    # 2. Предобработка
    print("\n[2/6] Предобработка...")
    df = preprocess(df_raw.copy())

    # 3. Вычисление daily_ots
    print("\n[3/6] Вычисление daily_ots...")
    daily_ots = compute_daily_ots(df)
    print(f"  Уникальных комбинаций (субъект, бренд, день): {len(daily_ots):,}")

    # 4. Поиск аномалий
    print("\n[4/6] Поиск аномалий...")
    reasons_df = detect_anomalies(daily_ots)
    print(f"  Найдено строк с причинами аномалий: {len(reasons_df):,}")

    anomalies = build_anomalies_csv(reasons_df)
    print(f"  Уникальных пар (SubjectID, date) для удаления: {len(anomalies):,}")
    print(f"  Уникальных аномальных респондентов: {anomalies['SubjectID'].nunique():,}")

    # 5. Создание выходных директорий
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    # 6. Сохранение основных файлов
    print("\n[5/6] Сохранение результатов...")
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)
    print(f"  Сохранено: {os.path.join(OUTPUT_DIR, 'anomalies.csv')}")

    if not reasons_df.empty:
        reasons_out = reasons_df.rename(columns={"CategoryDelivery": "CategoryDelivery"})
        reasons_out.to_csv(os.path.join(OUTPUT_DIR, "anomaly_reasons.csv"), index=False)
        print(f"  Сохранено: {os.path.join(OUTPUT_DIR, 'anomaly_reasons.csv')}")
    else:
        pd.DataFrame(columns=["SubjectID", "researchdate", "BrandID", "Brand",
                               "CategoryDelivery", "daily_ots", "score",
                               "threshold", "reason"]
                     ).to_csv(os.path.join(OUTPUT_DIR, "anomaly_reasons.csv"), index=False)

    # 7. Построение графиков
    print("\n[6/6] Построение обязательных графиков...")
    ots_daily, _ = compute_ots_totals(daily_ots, anomalies)

    plot_total_ots_before_after(
        ots_daily, os.path.join(OUTPUT_DIR, "plots", "total_ots_before_after.png"))

    cat_df = compute_category_ots_change(daily_ots, anomalies)
    plot_category_ots_change(
        cat_df, os.path.join(OUTPUT_DIR, "plots", "category_ots_change.png"))

    counts = compute_daily_anomaly_count(anomalies)
    plot_daily_anomaly_count(
        counts, os.path.join(OUTPUT_DIR, "plots", "daily_anomaly_count.png"))

    # ─── Итоговая статистика ───────────────────────────────────────────────
    total_subjects = df["SubjectID"].nunique()
    anom_subjects = anomalies["SubjectID"].nunique()
    total_ots_before = daily_ots["daily_ots"].sum()
    anom_keys_set = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    mask_clean = ~daily_ots.apply(
        lambda r: (r["SubjectID"], r["researchdate"]) in anom_keys_set, axis=1)
    total_ots_after = daily_ots[mask_clean]["daily_ots"].sum()

    print("\n" + "=" * 60)
    print("  Итоговая статистика")
    print("=" * 60)
    print(f"  Всего респондентов:          {total_subjects:,}")
    print(f"  Аномальных респондентов:     {anom_subjects:,} "
          f"({anom_subjects / total_subjects:.1%})")
    print(f"  Аномальных пар субъект-день: {len(anomalies):,}")
    print(f"  OTS до очистки:              {total_ots_before:,.0f}")
    print(f"  OTS после очистки:           {total_ots_after:,.0f}")
    print(f"  Сохранено OTS:               {total_ots_after / total_ots_before:.1%}")
    print("=" * 60)
    print(f"\nГотово! Результаты в папке: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
