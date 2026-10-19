import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Verilen eğitim ve test setleri üzerinde birden fazla sınıflandırma modelini eğitir 
    ve performans metriklerini hesaplar.
    """
    
    # Kullanılacak Modelleri Tanımla
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Karar Ağacı (Decision Tree)": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB()
    }

    results = {}
    best_f1_score = -1
    best_model_name = ""

    print("\n[BİLGİ] Modeller eğitiliyor ve değerlendiriliyor...")
    
    for name, model in models.items():
        print(f"\n--- {name} Modeli ---")
        
        # 3. MODEL EĞİTİMİ
        model.fit(X_train, y_train)

        # 4. TAHMİN
        y_pred = model.predict(X_test)

        # 5. PERFORMANS METRİKLERİ
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Recall": rec,
            "F1 Score": f1
        }
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            
        print(f"Accuracy (Doğruluk) : {acc:.4f}")
        print(f"Recall (Duyarlılık) : {rec:.4f}")
        print(f"F1 Score             : {f1:.4f}")
        print("\nDetaylı Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=['Kaldı (0)', 'Geçti (1)']))
        
    return results, best_model_name

def main():
    print("--- ÖĞRENCİ PERFORMANSI SINIFLANDIRMA SİSTEMİ ---")
    
    # 1. VERİ YÜKLEME 
    df = pd.read_csv("StudentsPerformance.csv")
    
    # 2. VERİ ÖN İŞLEME
    df['Math_Pass'] = np.where(df['math score'] >= 60, 1, 0)
    print(f"\n[BİLGİ] Hedef değişken 'Math_Pass' oluşturuldu. Geçti (1) oranı: {df['Math_Pass'].mean():.2%}")

    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Not sütunlarını ve hedefi X'ten çıkarıyoruz.
    X = df_encoded.drop(['Math_Pass', 'math score', 'reading score', 'writing score'], axis=1) 
    y = df_encoded['Math_Pass'] 
    
    print(f"[BİLGİ] Özellik sayısı: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3-5. MODELLERİ EĞİTME VE DEĞERLENDİRME
    results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 6. KARŞILAŞTIRMA VE SONUÇ (BU KISIM TABLOYU GÖSTERİR)
    print("\n" + "="*50)
    print("TÜM MODELLERİN KARŞILAŞTIRMALI SONUÇLARI (F1 Score'a Göre)")
    print("="*50)
    
    # Sonuçları F1 Score'a göre sırala
    results_df = pd.DataFrame(results).T.sort_values(by='F1 Score', ascending=False)
    
    # Raporu formatlı bir şekilde yazdır ve ekrana bastır
    print(results_df.applymap(lambda x: f"{x:.4f}").to_markdown(numalign="left", stralign="left"))
    
    print("\n" + "*"*50)
    print(f"EN İYİ PERFORMANSI GÖSTEREN MODEL (F1 Score): **{best_model_name}**")
    print(f"F1 Score: {results_df.loc[best_model_name]['F1 Score']:.4f}")
    print("*"*50)


if __name__ == "__main__":
    main()