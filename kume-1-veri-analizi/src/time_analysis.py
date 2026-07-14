"""
time_analysis.py
=================
KÜME 1 - Adım 6: Zaman Serisi Trend Analizi

Öğrenme hedefi: Şu ana kadar 'release_year' (içeriğin YAPIM yılı) ile
çalıştık. Bu adımda 'date_added' (içeriğin NETFLIX'E EKLENDİĞİ tarih)
kullanıyoruz - ikisi FARKLI şeyler! 2015 yapımı bir film 2019'da
Netflix'e eklenmiş olabilir. "Netflix hangi ayda/yılda ne kadar
büyüdü" sorusunun cevabı date_added'da saklı.

3 soru:
  1. Aylık/yıllık ekleme trendi nasıl?
  2. Pandemi (2020) öncesi ve sonrası fark var mı?
  3. Hangi türler en hızlı büyüyor/küçülüyor?
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles_clean.csv"
FIGURES_PATH = Path(__file__).resolve().parent.parent / "figures"


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date_added"])
    df["genres_list"] = df["genres_list"].apply(ast.literal_eval)
    return df


def monthly_yearly_trend(df: pd.DataFrame) -> pd.Series:
    """
    Aylık ekleme trendi.

    .set_index("date_added") -> tarihi index yapıyoruz çünkü .resample()
    SADECE datetime index üzerinde çalışır (tarih bazlı gruplama yapan
    özel bir groupby'dır).
    .resample("ME") -> "Month End" - her ayı bir grup say, o ay kaç kayıt
    var hesapla. ("YE" = Year End, yıllık için.)

    Bu, groupby("ay") yapmaktan farklıdır çünkü resample BOŞ ayları da
    (0 kayıt varsa) otomatik olarak tabloya dahil eder - groupby boş
    grupları atlar. Zaman serisinde "hiç eklenmemiş ay" bilgisi de önemlidir.
    """
    df = df.dropna(subset=["date_added"]).copy()
    monthly = df.set_index("date_added").resample("ME").size()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly.index, monthly.values, color="#E50914")
    plt.xlabel("Ay")
    plt.ylabel("Eklenen İçerik Sayısı")
    plt.title("Aylık İçerik Ekleme Trendi")
    plt.tight_layout()
    _save(plt, "monthly_addition_trend.png")

    print(f"En yoğun ay: {monthly.idxmax().strftime('%Y-%m')} ({monthly.max()} içerik)")
    print(f"Ortalama aylık ekleme: {monthly.mean():.1f}")

    return monthly


def pandemic_effect(df: pd.DataFrame) -> None:
    """
    Pandemi etkisi: 2020-03 öncesi vs sonrası karşılaştırma.

    ⚠️ TUZAK (ilk denemede düştüğüm hata): Veri setindeki TÜM geçmişi
    (2008'den beri) "öncesi" olarak, sadece son birkaç ayı "sonrası" olarak
    almak YANLIŞ bir karşılaştırma üretir. Netflix'in kataloğu zaten yıllar
    içinde katlanarak büyüyordu. 2008-2019 arası ortalamayı, zaten büyümüş
    olan 2020 ortalamasıyla kıyaslarsan, gördüğün "pandemi etkisi" aslında
    büyük ölçüde "zaten var olan büyüme trendi" - pandeminin kendisi değil.

    DOĞRU YAKLAŞIM: adil bir karşılaştırma için AYNI UZUNLUKTA iki pencere
    al - post-pandemic kaç ay ise (örn. 11 ay), pre-pandemic'i de son 11 ay
    ile sınırla.
    """
    df = df.dropna(subset=["date_added"]).copy()
    monthly = df.set_index("date_added").resample("ME").size()

    cutoff = pd.Timestamp("2020-03-01")
    post_pandemic = monthly[monthly.index >= cutoff]
    window = len(post_pandemic)

    pre_pandemic_all = monthly[monthly.index < cutoff]
    pre_pandemic_matched = pre_pandemic_all.tail(window)  # aynı uzunlukta pencere

    print(f"\n[YANLIŞ KIYAS - tüm geçmiş] Pandemi öncesi ({len(pre_pandemic_all)} ay) "
          f"ortalama: {pre_pandemic_all.mean():.1f} -> "
          f"yanlış değişim: %{(post_pandemic.mean() - pre_pandemic_all.mean()) / pre_pandemic_all.mean() * 100:+.1f}")

    print(f"\n[DOĞRU KIYAS - eşit pencere] Pandemi öncesi son {window} ay "
          f"ortalama: {pre_pandemic_matched.mean():.1f} içerik")
    print(f"Pandemi sonrası {window} ay ortalama: {post_pandemic.mean():.1f} içerik")

    change_pct = (post_pandemic.mean() - pre_pandemic_matched.mean()) / pre_pandemic_matched.mean() * 100
    print(f"Gerçek değişim (eşit pencereyle): %{change_pct:+.1f}")


def genre_growth_rates(df: pd.DataFrame, start_year: int = 2016, end_year: int = 2020) -> pd.Series:
    """
    Tür bazında yıllık büyüme oranı.

    .pct_change() -> pandas'ta bir Series/DataFrame'de her satırın bir
    öncekine göre yüzde değişimini hesaplayan çok kullanışlı bir fonksiyon.
    """
    df = df.dropna(subset=["date_added"]).copy()
    df["added_year"] = df["date_added"].dt.year

    exploded = df.explode("genres_list")
    yearly_counts = (
        exploded.groupby(["added_year", "genres_list"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    recent = yearly_counts.loc[start_year:end_year]

    # En az start_year'da 10+ içeriği olan türlerle sınırlıyoruz - çok
    # nadir türlerde yüzde değişim anlamsız derecede oynak sayılar üretir.
    eligible_genres = recent.columns[recent.loc[start_year] >= 10]
    recent = recent[eligible_genres]

    avg_growth = (recent.pct_change().mean() * 100).sort_values(ascending=False)

    print(f"\nEn hızlı büyüyen 5 tür ({start_year}-{end_year} ort. yıllık büyüme):")
    print(avg_growth.head(5).round(1))

    print(f"\nEn hızlı küçülen 5 tür ({start_year}-{end_year} ort. yıllık büyüme):")
    print(avg_growth.tail(5).round(1))

    return avg_growth


def _save(plt_module, filename: str) -> None:
    FIGURES_PATH.mkdir(exist_ok=True)
    out_path = FIGURES_PATH / filename
    plt_module.savefig(out_path, dpi=120)
    plt_module.close()
    print(f"📊 Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    df = load_clean_data()

    print("=" * 60)
    print("1) AYLIK/YILLIK EKLEME TRENDİ")
    print("=" * 60)
    monthly_yearly_trend(df)

    print("\n" + "=" * 60)
    print("2) PANDEMİ ETKİSİ")
    print("=" * 60)
    pandemic_effect(df)

    print("\n" + "=" * 60)
    print("3) TÜR BAZINDA YILLIK BÜYÜME ORANI")
    print("=" * 60)
    genre_growth_rates(df)