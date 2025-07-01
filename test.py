import pandas as pd
import joblib

# 📂 Model ve vektörleştiriciyi yükle
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 📩 E-posta içeriğini al (manuel olarak yapıştırılabilir)
email_content = input("E-posta içeriğini girin: ")

# 🧪 E-posta içeriğini vektörleştir
email_vector = vectorizer.transform([email_content])

# 🤖 E-posta içeriğini sınıflandır
prediction = model.predict(email_vector)

# 📊 Sonucu yazdır
if prediction[0] == 'Oltalama':
    print("Bu e-posta oltalama (phishing) olabilir!")
else:
    print("Bu e-posta güvenilirdir.")
