"""
hypothesis_testing.py
======================
KÜME 1 - Adım 5: İstatistiksel Hipotez Testleri

Öğrenme hedefi: Adım 4'te gözle gördüğümüz farkların "gerçek mi yoksa
tesadüf mü" olduğunu istatistiksel olarak test etmek. Her testte aynı
mantığı izliyoruz:

  H0 (null hipotez)   -> "Aslında fark yok, gördüğümüz şey tesadüf"
  H1 (alternatif)     -> "Gerçekten bir fark/ilişki var"
  p-value             -> H0 doğruyken, gözlemlediğimiz kadar (veya daha
                         aşırı) bir sonucu tesadüfen görme olasılığı
  Karar kuralı        -> p < 0.05 ise H0'ı reddet ("istatistiksel olarak
                         anlamlı bir fark/ilişki var" de)

ÖNEMLİ: p < 0.05, "fark BÜYÜK" demek değildir, "fark tesadüf DEĞİL" demektir.
Çok büyük veri setlerinde (bizde 7770 satır var) çok küçük farklar bile
"anlamlı" çıkabilir - bu yüzden p-value'nun yanına her zaman gerçek farkın
büyüklüğünü (effect size) de yazıyoruz.
"""

import ast
import pandas as pd
from scipy import stats
from pathlib import Path

CLEAN_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles_clean.csv"


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date_added"])
    df["genres_list"] = df["genres_list"].apply(ast.literal_eval)
    return df


def hypothesis_1_chi_square(df: pd.DataFrame) -> None:
    """
    HİPOTEZ 1: "Film/dizi oranı yıllar içinde değişti mi?"

    H0: release_year dönemi ile type (Movie/TV Show) arasında ilişki yoktur
    H1: release_year dönemi ile type arasında ilişki vardır

    NEDEN CHI-SQUARE? İki kategorik değişken arasındaki ilişkiyi test
    etmenin standart yolu budur. t-test/ANOVA sayısal bir ortalamayı
    kıyaslar, chi-square ise "kategori dağılımları birbirinden farklı mı"
    sorusuna cevap verir.

    Yılları 4 döneme grupladık (chi-square çok sayıda küçük hücreyle
    güvenilirliğini kaybeder, tek tek yıl yerine dönemlere böldük).
    """
    df = df.copy()
    df["period"] = pd.cut(
        df["release_year"],
        bins=[0, 2015, 2017, 2019, 2025],
        labels=["<=2015", "2016-2017", "2018-2019", "2020+"]
    )

    contingency = pd.crosstab(df["period"], df["type"])
    print("\nDönem × Tip kontenjans tablosu:")
    print(contingency)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\nChi-square istatistiği: {chi2:.2f}")
    print(f"Serbestlik derecesi (dof): {dof}")
    print(f"p-value: {p_value:.6f}")

    _print_verdict(
        p_value,
        h0="Dönem ile içerik tipi (Movie/TV Show) arasında ilişki yoktur",
        h1="Dönem ile içerik tipi arasında ilişki vardır"
    )

    proportions = contingency.div(contingency.sum(axis=1), axis=0) * 100
    print("\nDönem bazında Movie/TV Show oranı (%):")
    print(proportions.round(1))


def hypothesis_2_ttest(df: pd.DataFrame) -> None:
    """
    HİPOTEZ 2: "ABD filmleri diğer ülkelerin filmlerinden süre olarak farklı mı?"

    H0: ABD yapımı filmlerin ortalama süresi ile diğer ülkelerin ortalama
        süresi arasında fark yoktur
    H1: İki grubun ortalama süresi arasında fark vardır

    NEDEN T-TEST? İki bağımsız grubun (ABD vs diğer) SAYISAL bir ölçümünün
    ortalamasını kıyaslıyoruz - bağımsız örneklem t-testinin klasik
    kullanım alanı.

    equal_var=False -> Welch's t-test kullanıyoruz çünkü iki grubun
    varyansının eşit olduğunu VARSAYMIYORUZ. Bu daha güvenli/muhafazakar
    bir tercih - "varyanslar eşit" varsayımı yanlışsa klasik t-test
    yanıltıcı p-value verebilir.
    """
    movies = df[df["type"] == "Movie"].dropna(subset=["duration_minutes"])

    us_movies = movies[movies["country"].str.contains("United States", na=False)]["duration_minutes"]
    other_movies = movies[~movies["country"].str.contains("United States", na=False)]["duration_minutes"]

    print(f"\nABD filmleri: n={len(us_movies)}, ortalama={us_movies.mean():.1f} dk, std={us_movies.std():.1f}")
    print(f"Diğer ülkeler: n={len(other_movies)}, ortalama={other_movies.mean():.1f} dk, std={other_movies.std():.1f}")

    t_stat, p_value = stats.ttest_ind(us_movies, other_movies, equal_var=False)

    print(f"\nt-istatistiği: {t_stat:.3f}")
    print(f"p-value: {p_value:.6f}")

    _print_verdict(
        p_value,
        h0="ABD filmleri ile diğer ülkelerin filmlerinin ortalama süresi eşittir",
        h1="ABD filmleri ile diğer ülkelerin filmlerinin ortalama süresi farklıdır"
    )

    diff = us_movies.mean() - other_movies.mean()
    print(f"\nGerçek fark (effect size): {diff:.1f} dakika "
          f"({'ABD daha uzun' if diff > 0 else 'ABD daha kısa'})")


def hypothesis_3_anova(df: pd.DataFrame) -> None:
    """
    HİPOTEZ 3: "Rating kategorileri arasında film süresi farkı var mı?"

    H0: Tüm rating gruplarının ortalama film süresi eşittir
    H1: En az bir rating grubunun ortalama süresi diğerlerinden farklıdır

    NEDEN ANOVA? t-test SADECE İKİ grup kıyaslar. Burada 6+ rating
    kategorisi var - ikiden fazla grubun ortalamasını AYNI ANDA kıyaslamak
    istediğimizde One-way ANOVA kullanılır. (ANOVA sadece "en az biri
    farklı" der, HANGİSİNİN farklı olduğunu söylemez - onun için post-hoc
    test - örn. Tukey HSD - gerekir, bu bonus bir adım.)
    """
    movies = df[df["type"] == "Movie"].dropna(subset=["duration_minutes"])
    top_ratings = movies["rating"].value_counts().head(6).index
    movies_top = movies[movies["rating"].isin(top_ratings)]

    groups = [movies_top[movies_top["rating"] == r]["duration_minutes"] for r in top_ratings]

    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\nİncelenen rating kategorileri: {list(top_ratings)}")
    print(f"F-istatistiği: {f_stat:.3f}")
    print(f"p-value: {p_value:.6f}")

    _print_verdict(
        p_value,
        h0="Rating kategorileri arasında ortalama film süresi farkı yoktur",
        h1="En az bir rating kategorisinin ortalama film süresi diğerlerinden farklıdır"
    )

    print("\nGrup ortalamaları:")
    print(movies_top.groupby("rating")["duration_minutes"].mean().sort_values(ascending=False).round(1))


def _print_verdict(p_value: float, h0: str, h1: str, alpha: float = 0.05) -> None:
    """Ortak karar/yorum yazdırma fonksiyonu."""
    print(f"\nH0: {h0}")
    print(f"H1: {h1}")
    if p_value < alpha:
        print(f"✅ SONUÇ: p={p_value:.6f} < {alpha} -> H0 REDDEDİLDİ. {h1}")
    else:
        print(f"❌ SONUÇ: p={p_value:.6f} >= {alpha} -> H0 REDDEDİLEMEDİ. Yeterli kanıt yok.")


if __name__ == "__main__":
    df = load_clean_data()

    print("=" * 60)
    print("HİPOTEZ 1 — Chi-Square: Dönem × İçerik Tipi")
    print("=" * 60)
    hypothesis_1_chi_square(df)

    print("\n" + "=" * 60)
    print("HİPOTEZ 2 — t-test: ABD vs Diğer Ülkeler (Film Süresi)")
    print("=" * 60)
    hypothesis_2_ttest(df)

    print("\n" + "=" * 60)
    print("HİPOTEZ 3 — ANOVA: Rating Kategorilerine Göre Film Süresi")
    print("=" * 60)
    hypothesis_3_anova(df)