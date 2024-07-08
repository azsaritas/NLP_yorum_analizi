# -*- coding: utf-8 -*-
#NLP - Doğal Dil İşleme - Duygu Analizi
# Aziz Sarıtaş 210757061 

import pandas as pd 
import keras  #keras kütüphanesi
import numpy as np 
import re    #regex kütüphanesi 

veriseti_yolu = "birlesik_veri_seti_yorum_yeni.csv"
#kullandığımız verisetinin .csv dosyası yolu

veri = pd.read_csv(veriseti_yolu, sep=";", names=["yorum","deger"] , encoding="utf-8")
#verisetini okuyup ";" (yani sütun) ifadesine göre etiketleri yorum ve deger olarak ayırıyor. 

veri.head() #veriyi okuyoruz

yorumlar=veri["yorum"] #yorum sütun
degerler=veri["deger"] #deger sütun olarak ayrı ayrı atadık

yorumlar=yorumlar.str.lower() #bütün metni küçük harflere dönüştürdük

yorumlar.head() #yorumları okur
# degerler.head() #degerleri okur

notr=np.where(np.array(degerler)==2)[0]  
print(notr)     #verisetindeki 2 olarak etiketlenmiş ayrı nötr değerleri 0.5 olarak değştirdik.
for index in notr:
    degerler[index]=0.5
    
# degerler.head()

#%%

def metni_temizle(metin): #aldığı metini regex modülü ile özel karakterlerden temizleme fonksiyonu
    temiz_metin = re.sub(r'[^a-zA-Zğüşıöç\s]', '', str(metin)) 
    #regex türkçe karakterler ve normal harfler dışındaki her karakteri siler
    return temiz_metin

yorumlar=yorumlar.apply(metni_temizle) #yorumlar sütunüna fonksiyonu uygular

yorumlar.head()

#yorumlar[16],degerler[16]


#%%

import sklearn.model_selection #scikit-learn kütüphanesi

X_egitim=yorumlar #x egitim verisine yorumları atar
X_egitim2=yorumlar #x egitim verisine yorumları atar
y_egitim=degerler #y eğitim verisine degerleri atar

X_egitim,X_test,y_egitim,y_test=sklearn.model_selection.train_test_split(X_egitim,y_egitim,test_size=0.20,random_state=80)
#verisetinin yüzde 80ini x_eğitim'e eğitim için , yüzde 20sini x_teste test için atar ve verileri karıştırır. 

# X_egitim.head()
# y_egitim.head()
# X_test.head()
# y_test.head()

# print(X_egitim,y_egitim)

#X_egitim[16],y_egitim[16]

print("X eğitim verisi: ",X_egitim.shape) #verilerin boyutunu görmek için
print("X test verisi  : ",X_test.shape)
print("Y eğitim verisi: ",y_egitim.shape)
print("Y test verisi  : ",y_test.shape)

#%%

from tensorflow.keras.preprocessing.text import Tokenizer #tokenizer modülü

tokenizer = Tokenizer() #tokenizer oluşturuyoruz. bu sayede metin tokenler olarak parçalanacak.

tokenizer.fit_on_texts(X_egitim2) #x_egitim verisini tokenize eder.

kelime_tokenleri = tokenizer.word_index #tokenizerin parçaladığı x_eğitim tokenlerini gösterir.
kelime_tokenleri_uzunluk = len(kelime_tokenleri) #tokenlerin toplam sayısını verir.
kelime_tokenleri_uzunluk

kelime_tokenleri #tokenleri görmek için

#%%
from keras.preprocessing.sequence import pad_sequences #sequences modülü

X_egitim_dizisi = tokenizer.texts_to_sequences(X_egitim)
#x_eğitim verisne göre oluşturulan tokenlere sayısal değerler atanır. her token benzersiz olur.

X_test_dizisi = tokenizer.texts_to_sequences(X_test)
#aynı şey test verisi için de uygulanıyor

X_egitim_dizisi[10450] #örnek olarak 10450.verinin tokenleşmiş hali 

max_uzunluk = max([len(x) for x in X_egitim_dizisi]) #token dizisinin en uzun tokeni
max_uzunluk

#%%

sirali_x_egitim_dizisi=pad_sequences(X_egitim_dizisi,maxlen=max_uzunluk)
#oluşturulan token dizisinin sequences ile max_uzunluk a göre sıfır eklenerek hepsinin aynı boyutta olması sağlanıyor

sirali_x_test_dizisi=pad_sequences(X_test_dizisi,maxlen=max_uzunluk)
#aynı şey test verisi için de uygulanıyor

sirali_x_egitim_dizisi[10450] #örnek olarak 10450.verinin 0 eklenip tokenleştirilmiş hali 


#sirali_x_test_dizisi[120]

#%%

ters_katman = dict(zip(kelime_tokenleri.values(), kelime_tokenleri.keys()))
#sequences ile oluşturulan token dizilerinin tersi alınarak tekrardan kelimeye çevirme fonksiyonu

def tokeni_kelimeye_donustur(dizi_degerleri):
    kelimeler = [ters_katman[token] for token in dizi_degerleri if token != 0]
    metin = " ".join(kelimeler)  #token değeri 0 olmayanları alır kelime oluşur
    return metin

X_egitim_dizisi[10450]  #10450. verinin token hali
tokeni_kelimeye_donustur(X_egitim_dizisi[10450]) #10450. verinin metin hali

#%%

#sinir ağları bölümü
from keras.layers import Embedding, LSTM, Dense, Dropout #katmanların import edilmesi
from keras import Sequential          
from keras.models import Sequential

optimizer_fonksiyonu=keras.optimizers.Adam(learning_rate=0.001)
#kullandığımız optimizasyon yöntemi. Adam optimizeri kullanıyoruz öğrenme oranı 0.001

model = Sequential() 
#sequental modeli oluşturduk

model.add(Embedding(input_dim=kelime_tokenleri_uzunluk, output_dim=300))
#giriş olarak Embedding katmanı kullandık giriş sayısı oluştuduğumuz tokenlerin tomplam sayısı. çıkış ise değişken.

model.add(LSTM(100)) 
#LSTM uzun kısa süreli bellek 2.katman olarak kullandık. giriş parametresi değişken.

model.add(Dropout(0.2))
#aşırı öğrenme olmaması çıkışın %20sini devre dışı bıraktık. (deneysel olarak)

model.add(Dense(1, activation='sigmoid'))
#çıkış katmanı olarak Dense kullandık. sigmoid fonksiyonu ile çıkan 1 değeri 0-1 arasına sıkıştırdık.

model.compile(loss = 'binary_crossentropy', optimizer = optimizer_fonksiyonu, metrics=['accuracy']) 
#modelin loss fonksiyonunu binary_crossentropy (ikili), optimizerini Adam, ve doğruluk değeri için accuracy kullandık.
model.summary() #model özeti

#test etmek için farklı modeller -----------
# embedding_size = 512
# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(input_dim=12136,embeddings_initializer="uniform",output_dim=embedding_size,name="embedding_layer"))
# model.add(tf.keras.layers.GRU(units = 128, return_sequences = True))
# model.add(tf.keras.layers.GRU(units = 64, return_sequences = True))
# model.add(tf.keras.layers.GRU(units = 32))
#model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

# model = Sequential()
# model = keras.Sequential([
#     keras.layers.Embedding(10000, 128),
#     keras.layers.Bidirectional(keras.layers.LSTM(64)),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
#model.compile(loss = "binary_crossentropy", optimizer = optimizer_fonksiyonu, metrics = ["accuracy"])

#%%

sirali_x_egitim_dizisi=np.array(sirali_x_egitim_dizisi) #x eğitim dizisi array türüne dönüştürülüyor
sirali_x_test_dizisi=np.array(sirali_x_test_dizisi) #x test dizisi array türüne dönüştürülüyor

y_egitim=np.array(y_egitim) #y eğitim dizisi array türüne dönüştürülüyor
y_test=np.array(y_test) #y test dizisi array türüne dönüştürülüyor

sirali_x_egitim_dizisi[120]
y_egitim[120]


#%%

#model eğitimi
model.fit(sirali_x_egitim_dizisi, y_egitim, validation_data=(sirali_x_test_dizisi, y_test), epochs=6, batch_size=64)
#fit komutu ile giriş değeri olarak sıralanmış x ve y eğitim verileri giriyor
#doğrulama olarak her turda x ve y test verileri ile eğitime göre doğrulama yapıyor
#epoch değeri kaç çevrim yapacağını belirliyor. (değişken)
#batch_size değeri her çevrimde kullanılacak veri miktarını belirliyor. bellek için önemli.

#%%

model_degerlendir = model.evaluate(sirali_x_test_dizisi, y_test)
#modelin doğruluk ve loss oranını test verilerine göre değerlendiriyor.

model.save('duygu_analizi.keras')
#modeli kaydediyor.

#%%

def tahmin_yap(cumle): #kendi cümlelerimizle tahminde bulunmak için

    tahmin_cumlesi_tokeni = tokenizer.texts_to_sequences( [cumle] ) 
    #fonksiyona aldığımız cümleyi parçalayıyoruz (tokenleştiriyoruz)
    
    sirali_tahmin_cumlesi_tokeni = pad_sequences(tahmin_cumlesi_tokeni , max_uzunluk)
    #tokenleştirdiğimiz cümleyi padding yapıyoruz (0 ekleyerek sıralıyoruz)
    
    tahmin = model.predict(sirali_tahmin_cumlesi_tokeni)
    #modele tahmin yaptırıyoruz
    
    #tahmin değerine göre duyguyu sınıflanırıyor
    if tahmin>0 and tahmin<0.2:
        deger="Aşırı olumsuz"
    elif tahmin>0.2 and tahmin<0.5:
        deger="Olumsuz"
    elif tahmin>0.4 and tahmin<0.6:
        deger="Ortalama"
    elif tahmin>0.6 and tahmin<0.8:
        deger="Olumlu"
    elif tahmin>0.8 and tahmin<1:
        deger="Aşırı olumlu"
        
    tahmin = float(tahmin[0]) #tahmin değeri floatlaştırma
    
    print("Cumle: ", cumle)
    print(f"Tahmin Degeri: {(tahmin):6f}")
    print("Yorum Duygusu: ", deger)
    print("-------------------")
    
#örnek cümleler
cumle1="beklentimi tam olarak karşılamadı"
cumle2="oldukça iyi bir makine"
cumle3="kulaklığın kablo kalitesi iyi ama ses kalitesi biraz düşük"
cumle4="bu telefon hakkında ne diyeceğimden emin değilim idare eder gibi"
cumle5="bu bilgisayarı bedava versen bile tercih etmem"

tahmin_yap(cumle1)
tahmin_yap(cumle2)
tahmin_yap(cumle3)
tahmin_yap(cumle4)
tahmin_yap(cumle5)




#%%



