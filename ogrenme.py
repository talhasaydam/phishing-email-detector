import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 📂 1. Veri kümesini oku
veri = pd.read_csv("turkish_phishing_dataset.csv")

# 🧹 2. Eksik verileri temizle
veri.dropna(inplace=True)

# 🧠 3. Özellik (X) ve etiket (y) sütunlarını ayır
X = veri["İçerik"]
y = veri["Kategori"]

# ✍️ 4. Türkçe stop word'leri tanımla
turkce_stop_kelimeler = [
    'acaba','ama','aslında','az','bazı','belki','biri','birkaç','birşey','biz','bu','çok','çünkü',
    'da','daha','de','defa','diye','eğer','en','gibi','hem','hep','hepsi','her','hiç','için','ile',
    'ise','kez','ki','kim','mı','mu','mü','nasıl','ne','neden','nerde','nerede','nereye','niçin',
    'niye','o','sanki','şayet','şey','siz','şu','tüm','ve','veya','ya','yani'
]

# 📈 5. TF-IDF vektörleştirme işlemi
vektorizer = TfidfVectorizer(stop_words=turkce_stop_kelimeler)
X_vektorel = vektorizer.fit_transform(X)

# 🧪 6. Veriyi eğitim ve test olarak böl
X_egitim, X_test, y_egitim, y_test = train_test_split(X_vektorel, y, test_size=0.2, random_state=42)

# 🤖 7. Modeli oluştur ve eğit
model = MultinomialNB()
model.fit(X_egitim, y_egitim)

# 📊 8. Test verisiyle tahmin yap
tahminler = model.predict(X_test)

# ✅ 9. Sonuçları yazdır
print("Doğruluk Oranı: ", accuracy_score(y_test, tahminler))
print("Sınıflandırma Raporu:\n", classification_report(y_test, tahminler))

# 💾 10. Model ve vectorizer'ı kaydet
joblib.dump(model, 'model.pkl')
joblib.dump(vektorizer, 'vectorizer.pkl')
