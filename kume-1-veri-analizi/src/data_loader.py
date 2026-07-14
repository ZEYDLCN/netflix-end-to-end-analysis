"""
data_loader.py
==============
KÜME 1 - Adım 1: Veri Yükleme & İlk Bakış

Öğrenme hedefi: Bir CSV dosyasını pandas ile yükleyip "bu veri kim, ne anlatıyor,
neresi eksik, neresi bozuk" sorularına cevap verecek ilk keşfi yapmak.

Her fonksiyonun ne işe yaradığını docstring'lerde açıkladım; kod çalıştıkça
terminaldeki çıktıyı okuyarak pandas'ın mantığını kavrayacaksın.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles.csv"
FIGURES_PATH = Path(__file__).resolve().parent.parent / "figures"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    CSV dosyasını DataFrame'e yükler.

    pd.read_csv -> pandas'ın en çok kullanılan fonksiyonu. Dosyayı satır satır
    okuyup, her satırı bir kayıt (row), her kolonu bir öznitelik (column) olacak
    şekilde bir tablo (DataFrame) haline getirir. Excel'deki bir sayfa gibi
    düşünebilirsin ama Python'da programatik olarak sorgulayabildiğin bir yapı.
    """
    df = pd.read_csv(path)
    print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} kolon")
    return df


def first_look(df: pd.DataFrame) -> None:
    """
    Veriyle "ilk tanışma" - her yeni veri setinde yapman gereken 3 komut.

    .head()     -> ilk 5 satırı gösterir, veri neye benziyor gör
    .info()     -> her kolonun veri tipini (dtype) ve kaç tanesinin dolu (non-null)
                   olduğunu gösterir. Burada 'object' demek genelde string demektir.
    .describe() -> sadece sayısal kolonlar için ortalama, std, min, max gibi
                   istatistikleri verir. Kategorik (metin) kolonlar için
                   describe(include='object') kullanılır.
    """
    print("\n" + "=" * 60)
    print("İLK 5 SATIR (.head())")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("KOLON TİPLERİ VE DOLULUK ORANI (.info())")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("SAYISAL KOLONLAR İÇİN İSTATİSTİK (.describe())")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("KATEGORİK KOLONLAR İÇİN ÖZET (.describe(include='object'))")
    print("=" * 60)
    print(df.describe(include="object"))


def missing_value_report(df: pd.DataFrame, save_fig: bool = True) -> pd.DataFrame:
    """
    Eksik veri oranlarını hesaplar ve bir bar chart olarak görselleştirir.

    df.isnull()        -> her hücre için True/False (eksik mi değil mi) döner
    .sum()              -> her kolon için True'ların (eksiklerin) toplamını sayar
    .sort_values()      -> en çok eksiği olan kolon en üstte olacak şekilde sıralar

    Neden önemli? Bir kolonun %90'ı eksikse (mesela 'director' Netflix verisinde
    genelde öyledir), o kolonu doldurmak yerine belki tamamen ayrı bir
    "bilgi var mı yok mu" (has_director) bayrağına çevirmek daha mantıklı olur.
    Bu kararı Adım 2'de (cleaning.py) vereceğiz; burada sadece durumu tespit ediyoruz.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    report = pd.DataFrame({
        "eksik_sayisi": missing_count,
        "eksik_yuzde": missing_pct
    }).sort_values("eksik_yuzde", ascending=False)

    report = report[report["eksik_sayisi"] > 0]

    print("\n" + "=" * 60)
    print("EKSİK VERİ RAPORU")
    print("=" * 60)
    print(report)

    if save_fig:
        FIGURES_PATH.mkdir(exist_ok=True)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=report["eksik_yuzde"], y=report.index, color="#E50914")
        plt.xlabel("Eksik Veri Yüzdesi (%)")
        plt.ylabel("")
        plt.title("Kolon Bazında Eksik Veri Oranı")
        plt.tight_layout()
        out_path = FIGURES_PATH / "missing_data_report.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n📊 Grafik kaydedildi: {out_path}")

    return report


def check_dtypes_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kolon tiplerini kontrol eder ve 'date_added' kolonunu gerçek bir
    tarih (datetime) tipine çevirir.

    Neden önemli? CSV'den okunduğunda tarihler pandas için sadece birer
    string'tir ("August 14, 2020" gibi). pd.to_datetime() ile bunu gerçek bir
    tarih nesnesine çevirmezsek, ileride "hangi yılda kaç içerik eklendi"
    gibi bir zaman analizi (Adım 6) yapamayız - çünkü string'lerin üzerinde
    "bu tarih bundan önce mi sonra mı" gibi matematiksel işlem yapılamaz.
    """
    print("\n" + "=" * 60)
    print("MEVCUT DTYPE'LAR")
    print("=" * 60)
    print(df.dtypes)

    df = df.copy()
    original_notna = df["date_added"].notna().sum()
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="mixed", errors="coerce")
    new_notna = df["date_added"].notna().sum()

    print("\n✅ 'date_added' kolonu datetime'a çevrildi.")
    print(f"   Örnek: {df['date_added'].iloc[0]}  (tipi: {df['date_added'].dtype})")
    print(f"   Dönüştürülemeyen (parse edilemeyen) kayıt sayısı: {original_notna - new_notna}")
    return df


if __name__ == "__main__":
    df = load_data()
    first_look(df)
    missing_value_report(df)
    df = check_dtypes_and_dates(df)
