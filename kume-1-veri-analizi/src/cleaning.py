"""
cleaning.py
===========
KÜME 1 - Adım 2: Veri Temizleme

Öğrenme hedefi: Ham veriyi analiz edilebilir hale getirmek. Bu adımda 4 problemi
çözüyoruz:
  1. Eksik director / cast / country -> ne yapmalı?
  2. duration ("90 min" / "4 Seasons") -> sayısal bir kolona nasıl çevrilir?
  3. listed_in ("Dramas, Comedies") -> tek bir içerik birden fazla türe ait,
     bunu nasıl parse ederiz?
  4. Duplike (tekrar eden) kayıt var mı?
"""

import pandas as pd
from data_loader import load_data, check_dtypes_and_dates


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik veri stratejisi.

    Neden silmiyoruz? director %30 eksik. Bu kolonu tamamen silersek (dropna),
    7787 satırdan sadece ~5400'ü kalır - veri setinin üçte biri çöpe gider.
    Oysa 'director bilgisi yok' de başlı başına bir bilgidir (örneğin bağımsız
    yapımlarda ya da bazı ülkelerin içeriklerinde bu bilgi hiç girilmemiş olabilir).

    Bu yüzden kategorik/metin kolonlarda strateji şu:
    - Eksik değeri "Bilinmiyor" ile doldur (silme, bilgiyi koru)
    - fillna() -> pandas'ta bir Series/DataFrame'deki NaN değerlerini
      belirttiğin bir değerle doldurur.

    rating ve date_added gibi satır sayısı çok az eksik olan (<%1) kolonlarda
    ise satırı silmek (dropna) veri kaybını önemsiz kılar, o yüzden onları
    satır bazında sileceğiz.
    """
    df = df.copy()

    for col in ["director", "cast", "country"]:
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna("Bilinmiyor")
        print(f"'{col}': {n_missing} eksik değer 'Bilinmiyor' ile dolduruldu")

    before = len(df)
    df = df.dropna(subset=["rating", "date_added"])
    print(f"'rating'/'date_added' eksik olan {before - len(df)} satır silindi "
          f"(veri setinin %{(before - len(df)) / before * 100:.2f}'i - önemsiz)")

    return df


def parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    'duration' kolonunu iki ayrı sayısal kolona ayırır.

    Sorun: 'duration' kolonu Movie'ler için "90 min", TV Show'lar için
    "4 Seasons" içeriyor. Bunlar farklı birimler (dakika vs sezon), aynı
    kolonda tutmak analizi imkansız hale getiriyor (90 ile 4'ü karşılaştıramazsın).

    Çözüm: type'a göre ayrı iki kolon üret:
      - duration_minutes  -> sadece Movie'lerde dolu
      - duration_seasons  -> sadece TV Show'larda dolu

    .str.extract() -> regex ile string içinden bir sayı grubu çeker.
    r'(\\d+)' -> "bir veya daha fazla rakam" demek, parantez bu grubu yakalar.
    """
    df = df.copy()

    is_movie = df["type"] == "Movie"
    is_tv = df["type"] == "TV Show"

    df["duration_minutes"] = pd.NA
    df["duration_seasons"] = pd.NA

    df.loc[is_movie, "duration_minutes"] = (
        df.loc[is_movie, "duration"].str.extract(r"(\d+)", expand=False).astype(float)
    )
    df.loc[is_tv, "duration_seasons"] = (
        df.loc[is_tv, "duration"].str.extract(r"(\d+)", expand=False).astype(float)
    )

    print(f"Movie ortalama süre: {df['duration_minutes'].mean():.1f} dakika")
    print(f"TV Show ortalama sezon sayısı: {df['duration_seasons'].mean():.2f}")

    return df


def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    'listed_in' kolonunu bir liste (multi-label) haline getirir.

    Örnek: "International TV Shows, TV Dramas, TV Sci-Fi & Fantasy"
        -> ["International TV Shows", "TV Dramas", "TV Sci-Fi & Fantasy"]

    Neden önemli? Bir içerik aynı anda birden fazla türe ait olabilir.
    Bunu ayrı bir liste kolonu haline getirirsek, Adım 3'te (eda_univariate.py)
    "en popüler 10 tür" analizini .explode() ile kolayca yapabiliriz - her
    türü kendi satırına "patlatıp" saydırabiliriz.
    """
    df = df.copy()
    df["genres_list"] = df["listed_in"].apply(lambda x: [g.strip() for g in x.split(",")])
    df["n_genres"] = df["genres_list"].apply(len)

    print(f"Ortalama tür sayısı/içerik: {df['n_genres'].mean():.2f}")
    print(f"En çok türe sahip içerik: {df['n_genres'].max()} tür")

    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplike kayıt kontrolü.

    .duplicated() -> her satır için "bu satır daha önce göründü mü" diye
    True/False döner. subset ile hangi kolonlara göre kontrol edeceğini
    belirtiyoruz - burada aynı title+type+release_year kombinasyonu
    tekrarlanıyorsa muhtemelen aynı içeriktir.
    """
    dup_mask = df.duplicated(subset=["title", "type", "release_year"], keep="first")
    n_dup = dup_mask.sum()
    print(f"Bulunan duplike kayıt: {n_dup}")

    if n_dup > 0:
        df = df[~dup_mask].copy()
        print(f"Duplikeler silindi, kalan satır: {len(df)}")

    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm temizlik adımlarını sırayla uygular."""
    print("\n" + "=" * 60)
    print("1) EKSİK VERİ STRATEJİSİ")
    print("=" * 60)
    df = handle_missing_values(df)

    print("\n" + "=" * 60)
    print("2) DURATION PARSE")
    print("=" * 60)
    df = parse_duration(df)

    print("\n" + "=" * 60)
    print("3) GENRE (listed_in) PARSE")
    print("=" * 60)
    df = parse_genres(df)

    print("\n" + "=" * 60)
    print("4) DUPLIKE KONTROLÜ")
    print("=" * 60)
    df = check_duplicates(df)

    print(f"\n✅ Temizlik bitti. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_data()
    df = check_dtypes_and_dates(df)
    df_clean = clean_pipeline(df)
    df_clean.to_csv("../data/netflix_titles_clean.csv", index=False)
    print("\n💾 Temiz veri kaydedildi: data/netflix_titles_clean.csv")
