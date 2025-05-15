# Banka Kredi Onayı Tahmin Modeli

## Proje Açıklaması

Bu projede, banka müşterilerinin kredi başvurularının onaylanıp onaylanmayacağını tahmin eden bir makine öğrenmesi modeli geliştirdim. Gerçek dünya bankacılık senaryosunu simüle etmek için sentetik veri oluşturdum ve bu veri üzerinden bir RandomForest sınıflandırma modeli eğittim.

## Veri Seti Oluşturma

Projede 2000 adet sentetik müşteri kaydından oluşan bir veri seti oluşturdum. Bu veri seti aşağıdaki özelliklere sahiptir:

| Özellik            | Açıklama                            | Değer Aralığı    |
|--------------------|-------------------------------------|------------------|
| age                | Müşterinin yaşı                     | 18 - 65          |
| income             | Aylık gelir (₺)                     | 3.000 - 30.000   |
| debt               | Toplam kredi kartı/kredi borcu (₺)  | 0 - 50.000       |
| credit_score       | Kredi puanı                         | 0 - 1.000        |
| employment_years   | İş yerinde çalışma süresi (yıl)     | 0 - 40           |
| approved           | Kredi onayı (1=Onay, 0=Red)         | 0 veya 1         |

## Kredi Onay Mantığı

Veri setindeki kredi onayı durumunu belirlemek için aşağıdaki kuralı uyguladım:

```python
approved = (
    (income > 8000) &
    (debt < 20000) &
    (credit_score > 300) &
    (employment_years >= 2) &
    (age >= 21)
).astype(int)
```

Gerçek dünya senaryolarını daha iyi yansıtmak için verilere gürültü ekledim:

```python
income += np.random.normal(0, 3000, income.shape).astype(int)
debt += np.random.normal(0, 4000, debt.shape).astype(int)
credit_score += np.random.normal(0, 50, credit_score.shape).astype(int)
```

## Veri Analizi ve Görselleştirme

Veri setini oluşturduktan sonra çeşitli görselleştirme teknikleri kullanarak veriyi analiz ettim:

1. **Gelir Dağılımı Histogramı**: Müşterilerin gelir dağılımını göstermek için histogram kullandım.

2. **Kredi Puanı ve Onay Durumu**: Box plot ile kredi puanının onay durumuna göre nasıl değiştiğini görselleştirdim.

3. **Değişkenler Arası İlişkiler**: Seaborn pairplot ile tüm değişkenler arasındaki ilişkileri onay durumuna göre renklendirilmiş şekilde gösterdim.

## Model Geliştirme

Veri setini eğitim ve test setlerine ayırdım:

```python
X = df[['age', 'income', 'debt', 'credit_score', 'employment_years']]
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

RandomForestClassifier algoritmasını kullanarak bir sınıflandırma modeli eğittim:

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## Model Değerlendirme

Modelin performansını çeşitli metriklerle değerlendirdim:

1. **Doğruluk (Accuracy)**: Modelin genel tahmin başarısını ölçtüm.

2. **Karışıklık Matrisi (Confusion Matrix)**: Modelin yaptığı doğru ve yanlış tahminleri görselleştirdim.

3. **Precision ve Recall**: Modelin pozitif sınıfı tahmin etmedeki kesinliğini ve duyarlılığını ölçtüm.

4. **F1 Score**: Precision ve Recall metriklerinin harmonik ortalamasını hesaplayarak modelin genel performansını değerlendirdim.

## Kullanılan Teknolojiler

- Python 3.x
- Pandas ve NumPy (veri manipülasyonu)
- Scikit-learn (makine öğrenmesi)
- Matplotlib ve Seaborn (veri görselleştirme)

## Sonuçlar ve Çıkarımlar

Projede elde ettiğim sonuçlar, kredi onayı tahmininde RandomForest algoritmasının etkili olduğunu gösterdi. Özellikle kredi skoru ve gelir değişkenlerinin onay sürecinde en etkili faktörler olduğu görüldü.

Bu model, bankaların kredi risk değerlendirme süreçlerinde karar destek sistemi olarak kullanılabilir, böylece hem iş süreçlerini hızlandırabilir hem de daha tutarlı kararlar alınmasına yardımcı olabilir.

## Gelecek Çalışmalar

- Hyperparameter optimizasyonu ile model performansını artırma
- Farklı makine öğrenmesi algoritmalarının karşılaştırılması
- Daha karmaşık kredi onay kuralları ile model geliştirme
- ROC eğrisi ve AUC değeri analizi ekleme

---

Bu proje, makine öğrenmesi ve veri analizi yeteneklerimi geliştirmek amacıyla oluşturulmuştur. Tüm kod ve grafikler, gerçek bir banka veri analisti senaryosunu simüle etmek için tasarlanmıştır.
