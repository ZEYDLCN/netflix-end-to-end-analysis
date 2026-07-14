"""
eda_univariate.py
==================
KÜME 1 - Adım 3: Tek Değişkenli Analiz (Univariate EDA)

Öğrenme hedefi: Her bir kolonu TEK BAŞINA incelemek (henüz kolonlar arası
ilişkiye bakmıyoruz, o Adım 4'te). 5 soru soruyoruz:
  1. En popüler türler hangileri?
  2. En çok içerik üreten ülkeler hangileri?
  3. Hangi yıllarda kaç içerik eklenmiş?
  4. Film süreleri / dizi sezon sayıları nasıl dağılıyor?
  5. Rating (yaş sınırı) dağılımı nasıl?

Her fonksiyon aynı 3 adımı izliyor: veriyi say -> grafik çiz -> kaydet.
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLEAN_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles_clean.csv"
FIGURES_PATH = Path(__file__).resolve().parent.parent / "figures"


def load_clean_data() -> pd.DataFrame:
    """
    Temizlenmiş veriyi yükler.

    ÖNEMLİ BİR PANDAS TUZAĞI: cleaning.py içinde 'genres_list' kolonunu
    gerçek bir Python listesi olarak oluşturmuştuk (["Dramas", "Comedies"]).
    Ama CSV formatı sadece DÜZ METİN tutabilir - Python nesnelerini (liste,
    dict vb.) bilmez. Bu yüzden CSV'ye kaydedince liste, görünüşte listeye
    benzeyen ama aslında STRING olan "['Dramas', 'Comedies']" haline dönüşür.

    CSV'den geri okuyunca bunu tekrar gerçek bir listeye çevirmemiz lazım.
    Bunun için ast.literal_eval kullanıyoruz - bu fonksiyon "bu string bir
    Python literal'i mi (liste, sayı, dict...), öyleyse onu gerçek nesneye
    çevir" der.

    NOT: Bu yüzden pratikte, ara adımlarda ("checkpoint" olarak) CSV yerine
    Parquet formatı tercih edilir - Parquet, veri tiplerini (liste dahil)
    olduğu gibi saklar. Ama CSV her yerde okunabildiği için öğrenirken CSV
    ile devam ediyoruz; bu satırı bir "gerçek dünya" notu olarak düşün.
    """
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date_added"])
    df["genres_list"] = df["genres_list"].apply(ast.literal_eval)
    print(f"✅ Temiz veri yüklendi: {df.shape}")
    return df


def genre_distribution(df: pd.DataFrame, top_n: int = 10, save_fig: bool = True) -> pd.Series:
    """
    En popüler türleri bulur.

    .explode() -> bir kolonda liste olan her satırı, listenin her elemanı
    için AYRI BİR SATIRA "patlatır". Örnek:
        title="X", genres_list=["Dramas","Comedies"]
    şu iki satıra dönüşür:
        title="X", genre="Dramas"
        title="X", genre="Comedies"
    Böylece her tür kendi başına bir kayıt olur ve .value_counts() ile
    kaç kez geçtiğini doğru sayabiliriz. (explode yapmadan value_counts
    çağırırsan, "Dramas, Comedies" ikilisini TEK bir kategori sayar - yanlış
    olur çünkü "Dramas" tek başına da başka satırlarda geçiyor olabilir.)

    .value_counts() -> bir Series'teki her benzersiz değerin kaç kez
    geçtiğini sayar ve çoktan aza sıralı döner.
    """
    genre_counts = df["genres_list"].explode().value_counts().head(top_n)

    print(f"\nEn popüler {top_n} tür:")
    print(genre_counts)

    if save_fig:
        plt.figure(figsize=(9, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, color="#E50914")
        plt.xlabel("İçerik Sayısı")
        plt.title(f"En Popüler {top_n} Tür")
        plt.tight_layout()
        _save(plt, "genre_distribution.png")

    return genre_counts


def country_distribution(df: pd.DataFrame, top_n: int = 15, save_fig: bool = True) -> pd.Series:
    """
    En çok içerik üreten ülkeleri bulur.

    'country' kolonu da genres_list gibi çoklu değer içerebiliyor
    ("United States, India" gibi) ama bunu liste haline getirmemiştik.
    Burada aynı mantığı .str.split(", ") + .explode() ile tek satırda
    yapıyoruz - genres_list'i neden Adım 2'de ayrı bir kolon olarak
    hazırladığımızı, country'de ise anlık split yaptığımızı karşılaştır:
    ikisi de aynı sonucu verir, farklı yollardan.

    'Bilinmiyor' kategorisini grafikten çıkarıyoruz çünkü bu gerçek bir
    ülke değil, eksik veri işaretimiz (Adım 2'de biz koymuştuk) - onu
    saymak yanıltıcı olurdu.
    """
    country_counts = (
        df["country"]
        .str.split(", ")
        .explode()
        .pipe(lambda s: s[s != "Bilinmiyor"])
        .value_counts()
        .head(top_n)
    )

    print(f"\nEn çok içerik üreten {top_n} ülke:")
    print(country_counts)

    if save_fig:
        plt.figure(figsize=(9, 7))
        sns.barplot(x=country_counts.values, y=country_counts.index, color="#221F1F")
        plt.xlabel("İçerik Sayısı")
        plt.title(f"En Çok İçerik Üreten {top_n} Ülke")
        plt.tight_layout()
        _save(plt, "country_distribution.png")

    return country_counts


def year_distribution(df: pd.DataFrame, save_fig: bool = True) -> None:
    """
    release_year'a göre içerik sayısı dağılımı (histogram).

    Histogram, sürekli/sayısal bir değişkenin (burada yıl) hangi aralıklarda
    ne kadar yoğunlaştığını gösterir - bar chart'tan farkı, kategorileri
    değil ARALIKLARI (bin) gösterir. bins parametresi kaç aralığa
    bölüneceğini belirler.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df["release_year"], bins=30, color="#E50914")
    plt.xlabel("Yapım Yılı (release_year)")
    plt.ylabel("İçerik Sayısı")
    plt.title("Yıllara Göre İçerik Dağılımı")
    plt.tight_layout()
    if save_fig:
        _save(plt, "yearly_trend.png")

    print(f"\nEn eski içerik: {df['release_year'].min()}")
    print(f"En yeni içerik: {df['release_year'].max()}")
    print(f"En yoğun yıl: {df['release_year'].mode().iloc[0]}")


def duration_distribution(df: pd.DataFrame, save_fig: bool = True) -> None:
    """
    Film süresi ve dizi sezon sayısı dağılımı (boxplot).

    Boxplot bir dağılımın 5 önemli noktasını tek grafikte gösterir:
    min, Q1 (%25), medyan (%50), Q3 (%75), max - ve kutunun dışındaki
    noktalar "aykırı değer" (outlier) adayıdır. Histogram "şekle", boxplot
    "yayılıma ve aykırı değerlere" odaklanır - ikisi birbirini tamamlar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df["duration_minutes"].dropna(), ax=axes[0], color="#E50914")
    axes[0].set_title("Film Süresi (dakika)")
    axes[0].set_ylabel("Dakika")

    sns.boxplot(y=df["duration_seasons"].dropna(), ax=axes[1], color="#221F1F")
    axes[1].set_title("Dizi Sezon Sayısı")
    axes[1].set_ylabel("Sezon")

    plt.tight_layout()
    if save_fig:
        _save(plt, "duration_boxplot.png")

    print(f"\nFilm süresi -> medyan: {df['duration_minutes'].median():.0f} dk, "
          f"IQR: {df['duration_minutes'].quantile(0.25):.0f}-{df['duration_minutes'].quantile(0.75):.0f} dk")
    print(f"Dizi sezon -> medyan: {df['duration_seasons'].median():.0f}, "
          f"max: {df['duration_seasons'].max():.0f}")


def rating_distribution(df: pd.DataFrame, save_fig: bool = True) -> pd.Series:
    """
    Rating (yaş sınırı kategorisi) dağılımı.
    """
    rating_counts = df["rating"].value_counts()

    print("\nRating dağılımı:")
    print(rating_counts)

    if save_fig:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, color="#E50914")
        plt.xlabel("Rating")
        plt.ylabel("İçerik Sayısı")
        plt.title("Rating Dağılımı")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        _save(plt, "rating_distribution.png")

    return rating_counts


def _save(plt_module, filename: str) -> None:
    """Grafikleri figures/ klasörüne kaydeden ortak yardımcı fonksiyon."""
    FIGURES_PATH.mkdir(exist_ok=True)
    out_path = FIGURES_PATH / filename
    plt_module.savefig(out_path, dpi=120)
    plt_module.close()
    print(f"📊 Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    df = load_clean_data()

    print("\n" + "=" * 60)
    print("1) TÜR DAĞILIMI")
    print("=" * 60)
    genre_distribution(df)

    print("\n" + "=" * 60)
    print("2) ÜLKE DAĞILIMI")
    print("=" * 60)
    country_distribution(df)

    print("\n" + "=" * 60)
    print("3) YIL DAĞILIMI")
    print("=" * 60)
    year_distribution(df)

    print("\n" + "=" * 60)
    print("4) SÜRE DAĞILIMI")
    print("=" * 60)
    duration_distribution(df)

    print("\n" + "=" * 60)
    print("5) RATING DAĞILIMI")
    print("=" * 60)
    rating_distribution(df)