import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# 📂 1. Veri kümesini oku
veri = pd.read_csv("turkish_phishing_dataset.csv")

# 🧹 2. Eksik verileri temizle
veri.dropna(inplace=True)

# 🧼 3. Temizleme işlemi: küçük harfe çevirme ve özel karakterlerden arındırma
def temizle(text):
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldırma
    return text

veri['İçerik'] = veri['İçerik'].apply(temizle)

# 🧠 4. Özellik (X) ve etiket (y) sütunlarını ayır
X = veri["İçerik"]
y = veri["Kategori"]

# ✍️ 5. Türkçe stop kelimeler
turkce_stop_kelimeler = [
    'acaba','ama','aslında','az','bazı','belki','biri','birkaç','birşey','biz','bu','çok','çünkü',
    'da','daha','de','defa','diye','eğer','en','gibi','hem','hep','hepsi','her','hiç','için','ile',
    'ise','kez','ki','kim','mı','mu','mü','nasıl','ne','neden','nerde','nerede','nereye','niçin',
    'niye','o','sanki','şayet','şey','siz','şu','tüm','ve','veya','ya','yani'
]

# 📈 6. TF-IDF vektörleştirme işlemi
vektorizer = TfidfVectorizer(stop_words=turkce_stop_kelimeler, ngram_range=(1, 2))  # Kelime çiftleri
X_vektorel = vektorizer.fit_transform(X)

# 🧪 7. Veriyi eğitim ve test olarak böl
X_egitim, X_test, y_egitim, y_test = train_test_split(X_vektorel, y, test_size=0.2, random_state=42)

# 🤖 8. Modeli oluştur ve eğit
model = MultinomialNB(class_prior=[0.7, 0.3])  # Güvenilir e-postalara daha fazla ağırlık ver
model.fit(X_egitim, y_egitim)

# 📊 9. Test verisiyle tahmin yap
tahminler = model.predict(X_test)

# ✅ 10. Sonuçları yazdır
print("Doğruluk Oranı: ", accuracy_score(y_test, tahminler))
print("Sınıflandırma Raporu:\n", classification_report(y_test, tahminler))

# 💾 11. Model ve vectorizer'ı kaydet
joblib.dump(model, 'model.pkl')
joblib.dump(vektorizer, 'vectorizer.pkl')
