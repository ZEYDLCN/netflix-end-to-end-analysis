"""
eda_bivariate.py
================
KÜME 1 - Adım 4: İki Değişkenli Analiz (Bivariate EDA)

Öğrenme hedefi: Artık kolonları TEK TEK değil, İKİŞER İKİŞER inceliyoruz.
Asıl soru şu: "İki değişken birlikte hareket ediyor mu?"
  1. Ülke × Tür: Hindistan gerçekten drama mı çekiyor, Kore romantik mi?
  2. Yıl × Tür: Belgesel türü son yıllarda arttı mı?
  3. Süre × Rating: TV-MA içerikler gerçekten daha mı uzun?
  4. Korelasyon matrisi: sayısal kolonlar birbiriyle ilişkili mi?
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLEAN_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles_clean.csv"
FIGURES_PATH = Path(__file__).resolve().parent.parent / "figures"


def load_clean_data() -> pd.DataFrame:
    """eda_univariate.py'deki ile aynı yükleme mantığı."""
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date_added"])
    df["genres_list"] = df["genres_list"].apply(ast.literal_eval)
    print(f"✅ Temiz veri yüklendi: {df.shape}")
    return df


def country_genre_heatmap(df: pd.DataFrame, top_countries: int = 8, top_genres: int = 8) -> pd.DataFrame:
    """
    Ülke × Tür crosstab'ı: "Hindistan drama mı çekiyor, Kore romantik mi?"

    ADIM ADIM MANTIK:
    1. Hem country hem genres_list çoklu değer içeriyor -> ikisini de
       explode() etmemiz lazım.
    2. Önce country'yi listeye çeviriyoruz (.str.split), sonra iki kez
       art arda explode() ile satırları çoğaltıyoruz, ardından index'i
       reset_index(drop=True) ile sıfırlıyoruz (yukarıdaki hatayı hatırla).
    3. pd.crosstab() -> iki kategorik kolon arasında "kaç kez birlikte
       göründüler" tablosu çıkarır (Excel'deki pivot table'ın pandas hali).
    4. En sık geçen top_countries ülke ve top_genres tür ile sınırlıyoruz.

    Heatmap'te koyu renk = o ülke-tür kombinasyonu sık görülüyor demek.
    """
    df = df.copy()
    df["country_list"] = df["country"].str.split(", ")

    exploded = df.explode("country_list").explode("genres_list").reset_index(drop=True)
    exploded = exploded[exploded["country_list"] != "Bilinmiyor"]

    top_c = exploded["country_list"].value_counts().head(top_countries).index
    top_g = exploded["genres_list"].value_counts().head(top_genres).index

    subset = exploded[exploded["country_list"].isin(top_c) & exploded["genres_list"].isin(top_g)]

    ct = pd.crosstab(subset["country_list"], subset["genres_list"])
    # Her ülkenin toplam içerik sayısına göre normalize ediyoruz (satır bazında %),
    # çünkü ABD zaten en çok içeriğe sahip - ham sayılarla kıyaslarsak her
    # kategoride "ABD kazanır" gibi yanıltıcı bir sonuç çıkar.
    ct_normalized = ct.div(ct.sum(axis=1), axis=0) * 100

    print("\nÜlke × Tür (satır bazında %):")
    print(ct_normalized.round(1))

    plt.figure(figsize=(10, 6))
    sns.heatmap(ct_normalized, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "% (ülke içi)"})
    plt.title("Ülke × Tür İlişkisi (satır yüzdesi)")
    plt.xlabel("Tür")
    plt.ylabel("Ülke")
    plt.tight_layout()
    _save(plt, "country_genre_heatmap.png")

    return ct_normalized


def year_genre_trend(df: pd.DataFrame, genres_to_track=None) -> pd.DataFrame:
    """
    Yıl × Tür trendi: "Belgesel türü son yıllarda arttı mı?"

    groupby(['release_year', 'genres_list']) -> hem yıla hem türe göre
    grupla, her grup için kaç satır olduğunu say.

    .unstack() -> groupby sonucundaki iç index'i (tür) kolonlara çevirir,
    "yıl satır, tür kolon" formatına getirir - çizgi grafik için ideal.
    """
    if genres_to_track is None:
        genres_to_track = ["Documentaries", "Dramas", "Comedies", "International Movies"]

    exploded = df.explode("genres_list")
    yearly = (
        exploded[exploded["genres_list"].isin(genres_to_track)]
        .groupby(["release_year", "genres_list"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    # Son 15 yılla sınırlıyoruz - eski yıllar grafiği gereksiz uzatır.
    yearly_recent = yearly[yearly.index >= 2006]

    plt.figure(figsize=(11, 5))
    for genre in genres_to_track:
        if genre in yearly_recent.columns:
            plt.plot(yearly_recent.index, yearly_recent[genre], marker="o", label=genre)
    plt.xlabel("Yıl")
    plt.ylabel("İçerik Sayısı")
    plt.title("Yıllara Göre Tür Trendi (2006+)")
    plt.legend()
    plt.tight_layout()
    _save(plt, "genre_year_trend.png")

    doc_2016 = yearly_recent.get("Documentaries", pd.Series(dtype=int)).get(2016, 0)
    doc_2020 = yearly_recent.get("Documentaries", pd.Series(dtype=int)).get(2020, 0)
    print(f"\nBelgesel sayısı 2016: {doc_2016} -> 2020: {doc_2020}")

    return yearly_recent


def duration_rating_relationship(df: pd.DataFrame) -> None:
    """
    Süre × Rating: "TV-MA içerikler gerçekten daha mı uzun?"

    Sadece Movie'lere bakıyoruz çünkü duration_minutes sadece onlarda dolu.
    Rating kategorisine göre gruplandırıp boxplot çiziyoruz.
    """
    movies = df[df["type"] == "Movie"].copy()

    top_ratings = movies["rating"].value_counts().head(6).index
    movies_top = movies[movies["rating"].isin(top_ratings)]

    order = movies_top.groupby("rating")["duration_minutes"].median().sort_values(ascending=False).index

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=movies_top, x="rating", y="duration_minutes", order=order, color="#E50914")
    plt.xlabel("Rating")
    plt.ylabel("Film Süresi (dakika)")
    plt.title("Rating'e Göre Film Süresi Dağılımı")
    plt.tight_layout()
    _save(plt, "duration_vs_rating.png")

    print("\nRating bazında medyan süre (azalan sıra):")
    print(movies_top.groupby("rating")["duration_minutes"].median().sort_values(ascending=False))


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sayısal kolonlar arası korelasyon matrisi.

    .corr() -> her sayısal kolon çifti için Pearson korelasyon katsayısı
    hesaplar (-1 ile +1 arası). +1'e yakın: biri artınca diğeri de artıyor.
    -1'e yakın: biri artınca diğeri azalıyor. 0'a yakın: doğrusal ilişki yok.
    """
    numeric_cols = ["release_year", "duration_minutes", "duration_seasons", "n_genres"]
    corr = df[numeric_cols].corr()

    print("\nKorelasyon matrisi:")
    print(corr.round(2))

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Sayısal Kolonlar Arası Korelasyon")
    plt.tight_layout()
    _save(plt, "correlation_matrix.png")

    return corr


def _save(plt_module, filename: str) -> None:
    FIGURES_PATH.mkdir(exist_ok=True)
    out_path = FIGURES_PATH / filename
    plt_module.savefig(out_path, dpi=120)
    plt_module.close()
    print(f"📊 Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    df = load_clean_data()

    print("\n" + "=" * 60)
    print("1) ÜLKE × TÜR")
    print("=" * 60)
    country_genre_heatmap(df)

    print("\n" + "=" * 60)
    print("2) YIL × TÜR TRENDİ")
    print("=" * 60)
    year_genre_trend(df)

    print("\n" + "=" * 60)
    print("3) SÜRE × RATING")
    print("=" * 60)
    duration_rating_relationship(df)

    print("\n" + "=" * 60)
    print("4) KORELASYON MATRİSİ")
    print("=" * 60)
    correlation_matrix(df)