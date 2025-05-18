import gdown

# file_id = "1FQb1MlER1s1sImeZgTmMVD8hqVkkTk4m"
# url = f"https://drive.google.com/uc?id={file_id}"
# output = "nama_file_yang_ingin_disimpan"

# gdown.download(url, output, quiet=False)


# Install Library

!conda install -y gdown

# Import Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Connect to Google Drive

# !gdown --id 1FQb1MlER1s1sImeZgTmMVD8hqVkkTk4m

data = pd.read_excel('dataset siga.xlsx')

print('Jumlah Baris dan Kolom Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 :\n', data.shape)
print('Jumlah Baris Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 : \n', data.shape[0])
print('Jumlah Kolom Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 :\n', data.shape[1])

# Data Exploratory Analysis

# show object variable

categorical = [var for var in data.columns if data[var].dtype=='O']
print('Terdapat {} categorical variabel\n'.format(len(categorical)))
print('Variabel categorical adalah:\n', categorical)

# show categorical and numerical variable

numerikal =[]
for num in data.columns:
  if data[num].dtypes == 'int64':
    numerikal.append(num)
print('\nAda {} variabel numerik:\n'.format(len(numerikal)))
print('Variabel numerik adalah:\n', numerikal)

Bagian kode ini digunakan untuk mengidentifikasi kolom-kolom dalam dataset data yang bertipe kategorikal, yaitu kolom yang memiliki tipe data object (umumnya berupa string seperti nama wilayah, jenis, atau kategori lain). Hasilnya disimpan dalam list categorical. Selanjutnya, jumlah variabel kategorikal dicetak ke layar bersama daftar nama kolomnya. Ini berguna untuk memahami struktur data secara kualitatif sebelum melakukan analisis lebih lanjut.
Lalu bagian selanjutnya berfungsi untuk mengidentifikasi kolom-kolom yang bertipe data numerik dengan tipe int64. Setiap kolom pada dataset dicek, dan jika bertipe integer (int64), maka nama kolom tersebut dimasukkan ke dalam list numerikal. Setelah seluruh kolom diperiksa, jumlah variabel numerik dan nama-namanya akan dicetak. Identifikasi ini penting untuk menentukan kolom mana yang dapat dihitung atau dianalisis secara statistik.



print('Ringkasan Data Kekerasan Provinsi Sumatera Utara: \n')
data.info()

fungsi data.info() digunakan untuk menampilkan ringkasan struktur dataset, yang meliputi jumlah total entri, nama kolom, jumlah data non-null per kolom, dan tipe data masing-masing kolom. Ini memberikan gambaran menyeluruh mengenai kelengkapan data serta tipe masing-masing fitur, sehingga sangat membantu dalam menentukan langkah preprocessing selanjutnya.

data['Tahun'] = pd.to_datetime(data['Tahun'].astype(str) + '-01-01')

Bagian kode ini mengubah kolom Tahun, yang awalnya bertipe numerik atau string, menjadi format tanggal (datetime) dengan menetapkan setiap nilai tahun ke tanggal 1 Januari pada tahun tersebut. Ini penting untuk analisis berbasis waktu (time-series), seperti tren kasus kekerasan dari tahun ke tahun. Dengan format datetime, kolom ini dapat dimanfaatkan dalam visualisasi dan manipulasi data waktu secara efisien menggunakan pustaka pandas.

print('Statistik Deskriptif Data Kasus Kekerasan Provinsi Sumatera Utara: \n')
desc_data = data.describe()
np.transpose(desc_data)

fungsi data.describe() digunakan untuk menghasilkan statistik deskriptif dari seluruh kolom numerik dalam dataset. Informasi yang diperoleh meliputi nilai minimum, maksimum, mean (rata-rata), standar deviasi, serta kuartil. Data tersebut kemudian ditranspos menggunakan np.transpose() agar lebih mudah dibaca, dengan kolom sebagai baris. Statistik deskriptif ini memberikan gambaran awal mengenai sebaran dan karakteristik data, serta bisa digunakan untuk mendeteksi nilai ekstrem atau anomali.

print('Korelasi dataset: \n')
# Drop the non-numeric 'Kabupaten/Kota' and 'Jenis Kelamin' columns before calculating correlation
corr = data.drop(['Kabupaten/Kota', 'Jenis Kelamin'], axis=1).corr(method='pearson')
corr

Bagian ini menghitung matriks korelasi Pearson antar variabel numerik. Sebelum perhitungan, dua kolom kategorikal, yaitu Kabupaten/Kota dan Jenis Kelamin, dihapus dari dataset karena tidak relevan untuk analisis korelasi numerik. Korelasi Pearson mengukur hubungan linear antara dua variabel, dengan nilai berkisar dari -1 (korelasi negatif sempurna) hingga 1 (korelasi positif sempurna). Matriks korelasi ini penting untuk memahami hubungan antar jenis kasus kekerasan, serta membantu mengidentifikasi pola atau redundansi antar fitur dalam data.

plt.figure(figsize=(24,15))
sns.heatmap(data=corr, annot=True)
plt.title('Korelasi Pearson Heatmap', pad=20, fontsize=25)
plt.savefig('korelasi.png')
plt.show()

>>
**1. Korelasi Positif yang Kuat**

Beberapa variabel menunjukkan korelasi positif yang sangat tinggi (mendekati 1), yang artinya keduanya cenderung meningkat atau menurun secara bersamaan. Misalnya:

  Kekerasan Seksual ↔ Eksploitasi: Korelasinya sangat tinggi (sekitar 0.88), menunjukkan bahwa kasus eksploitasi sering muncul beriringan dengan kekerasan seksual.

  Kekerasan Lainnya ↔ Penelantaran ↔ Trafficking: Korelasi di atas 0.75, menandakan keterkaitan atau pola terjadinya beberapa jenis kekerasan sekaligus.

  Kelompok Usia 0–5 Tahun ↔ 6–12 Tahun ↔ 13–17 Tahun: Ketiganya memiliki korelasi sangat tinggi satu sama lain (> 0.8), bisa jadi karena data usia anak-anak dan remaja sering diklasifikasikan bersama dalam pelaporan.




>>
**2. Korelasi Negatif atau Lemah**

Sebagian besar variabel memiliki korelasi lemah atau mendekati nol, misalnya antara variabel Tahun dengan jenis kekerasan tertentu, yang artinya tidak ada hubungan linear yang kuat antar tahun dan banyaknya kasus.

Contoh korelasi negatif kecil: Tahun ↔ Lingkungan Masyarakat (-0.33). Ini bisa menunjukkan bahwa jumlah kasus di lingkungan masyarakat sedikit menurun dari tahun ke tahun.

>>
**3. Korelasi antar Sumber Pelaku**

Orang Tua ↔ Keluarga memiliki korelasi kuat (~0.76), yang masuk akal karena pelaku dalam lingkup keluarga dekat bisa saja dicatat dalam kedua kategori.

Sekolah ↔ Petugas Sekolah (Guru dll) juga menunjukkan hubungan yang cukup kuat (~0.6), mendukung asumsi bahwa kasus di sekolah sering melibatkan guru atau staf.

>>
**4. Korelasi antar Jenis Penanganan**

Pengaduan ↔ Kesehatan ↔ Bantuan Hukum menunjukkan korelasi positif yang kuat (> 0.7), menunjukkan bahwa jika ada pengaduan, biasanya juga disertai dengan intervensi hukum dan kesehatan.

Rehabilitasi Sosial ↔ Pemulangan dan Reintegrasi juga saling terkait, karena proses ini biasanya terjadi dalam satu alur penanganan korban.

print('Mengecek keseimbangan data: \n')
print(data['Jenis Kelamin'].value_counts())

Data pada variabel Jenis Kelamin menunjukkan distribusi yang seimbang, masing-masing 99 sampel untuk Laki-laki dan Perempuan. Keseimbangan ini memudahkan analisis dan pelatihan model karena tidak memerlukan penyesuaian terhadap ketimpangan kelas.

laki = len(data[data['Jenis Kelamin'] == 'Laki-laki'])
perempuan = len(data[data['Jenis Kelamin'] == 'Perempuan'])

lakipersen = laki/(laki+perempuan)
print('Persentase dari Laki-laki:', lakipersen * 100)

perempuanpersen = perempuan/(laki+perempuan)
print('Persentase dari Perempuan:', perempuanpersen*100)

Distribusi data berdasarkan variabel Jenis Kelamin menunjukkan bahwa masing-masing kategori, yaitu Laki-laki dan Perempuan, memiliki proporsi yang seimbang, yaitu 50,0%. Keseimbangan ini menunjukkan bahwa tidak terdapat dominasi kelas tertentu, sehingga analisis atau pemodelan yang dilakukan tidak perlu penanganan ketidakseimbangan data.

print('rata-rata korban dan pelaku berdasarkan jenis kelamin: \n')
observ = data.copy()
# Drop the non-numeric 'Tahun' and 'Kabupaten/Kota' columns before calculating the mean
observ.drop(['Tahun', 'Kabupaten/Kota'], axis=1, inplace=True)
observ.groupby(['Jenis Kelamin']).mean()

Berdasarkan rata-rata data korban dan pelaku yang dikelompokkan berdasarkan jenis kelamin, terlihat adanya perbedaan pola antara laki-laki dan perempuan:

Perempuan memiliki rata-rata yang jauh lebih tinggi pada kasus kekerasan seksual (13.18) dibandingkan laki-laki (1.74), menunjukkan bahwa perempuan lebih rentan menjadi korban kekerasan seksual.

Pada jenis kekerasan psikis, perempuan juga mencatat rata-rata yang lebih tinggi (3.77) dibandingkan laki-laki (2.08).

Sebaliknya, laki-laki sedikit lebih tinggi dalam kekerasan fisik (3.81) dibanding perempuan (3.18), yang bisa mengindikasikan peran laki-laki lebih dominan sebagai pelaku atau korban dalam kategori ini.

Untuk kelompok usia korban, perempuan mendominasi hampir semua kelompok usia, terutama pada usia 13–17 tahun (9.51) dan 6–12 tahun (6.90), dibandingkan laki-laki dengan masing-masing 3.97 dan 3.54.

Dalam konteks layanan atau intervensi, perempuan memiliki angka lebih tinggi pada pengaduan (14.26), kesehatan (3.12), dan bantuan hukum serta penegakan (6.54), yang menunjukkan kebutuhan bantuan dan perlindungan lebih besar pada kasus yang melibatkan perempuan.

Secara keseluruhan, hasil ini menegaskan bahwa perempuan cenderung lebih banyak terlibat sebagai korban dalam berbagai bentuk kekerasan dan memerlukan lebih banyak akses ke layanan pemulihan dan perlindungan.

print('rata-rata banyaknya kasus berdasarkan tahun:\n')
# Select only the numeric columns before calculating the mean
numeric_data = data.select_dtypes(include=np.number)
numeric_data.groupby(data['Tahun']).mean()

datatahun = data.loc[:,'Tahun':'Kekerasan Lainnya']
datatahun.drop(['Kabupaten/Kota'], axis=1, inplace=True)
datatahun['Tahun'] = datatahun['Tahun'].dt.year.astype(str)
datatahun['Total'] = datatahun.loc[:,'Kekerasan Fisik':'Kekerasan Lainnya'].sum(axis=1)
datagroup = datatahun.groupby(['Tahun']).sum().reset_index()
datagroup

Berikut adalah analisis rata-rata banyaknya kasus kekerasan berdasarkan tahun:

Pada tahun 2017, jumlah rata-rata kasus kekerasan seksual cukup tinggi, yaitu sekitar 8.33 kasus, disusul oleh kekerasan psikis (4.24) dan kekerasan fisik (4.12). Ini menunjukkan bahwa pada tahun ini, kekerasan seksual menjadi kasus yang paling menonjol.

Pada tahun 2018, rata-rata kasus kekerasan seksual hampir sama dengan tahun sebelumnya, yakni 8.27 kasus, namun terjadi penurunan pada kekerasan fisik (3.73) dan kekerasan psikis (2.61). Jumlah pengaduan meningkat cukup signifikan hingga mencapai 14.80, menandakan peningkatan kesadaran atau pelaporan kasus.

Pada tahun 2019, terjadi penurunan signifikan pada semua jenis kekerasan, terutama kekerasan seksual yang turun menjadi 5.77 kasus, kekerasan fisik turun ke 2.64, dan kekerasan psikis turun ke 1.92. Hal ini mungkin menunjukkan perbaikan situasi atau perubahan dalam pelaporan kasus.

Secara umum, data menunjukkan tren penurunan kasus kekerasan dari 2017 ke 2019, khususnya pada kekerasan seksual, fisik, dan psikis. Namun, pengaduan justru meningkat pada 2018, yang bisa berarti adanya peningkatan pelaporan atau penanganan kasus.

import matplotlib.dates as mdates

plt.figure(figsize=(14,7))
sns.set()
var = datagroup['Tahun']
num = datagroup['Total']

# Menggunakan lineplot di sini
lines = sns.lineplot(x=var, y=num, data=datagroup, palette='Blues_r', marker='o')  # menambahkan marker untuk titik data

lines.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xlabel('Tahun', fontsize=18, labelpad=10)
plt.ylabel('Jumlah Kasus', fontsize=18)
plt.title('Kasus Kekerasan Provinsi Sumatera Utara', pad=40, fontsize=25, color='blue')
plt.xticks(datagroup['Tahun'].unique())
plt.savefig('jumlah kasus berdasarkan tahun.png', bbox_inches='tight')  # Memperbaiki typo dari bbox_tight menjadi bbox_inches='tight'
plt.show()

datagroup=data.loc[:,'Tahun':'Kekerasan Lainnya']
datagroup.drop(['Kabupaten/Kota'], axis=1, inplace=True)
datagroup['Tahun'] = datagroup['Tahun'].dt.year
datagroup['Total'] = datagroup.loc[:,'Kekerasan Fisik':'Kekerasan Lainnya'].sum(axis=1)
datagrouptahun = datagroup.groupby(['Tahun'])[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Eksploitasi','Trafficking','Penelantaran','Kekerasan Lainnya']].sum()
datagrouptahun

Pada tahun 2017, kasus kekerasan seksual paling tinggi dengan jumlah 550 kasus, diikuti oleh kekerasan psikis (280) dan kekerasan fisik (272). Kasus eksploitasi dan trafficking belum tercatat.

Pada tahun 2018, kekerasan seksual tetap tinggi dengan 546 kasus, meskipun ada penurunan pada kekerasan fisik (246) dan kekerasan psikis (172). Kasus penelantaran meningkat cukup signifikan menjadi 210 kasus.

Pada tahun 2019, terjadi penurunan signifikan pada semua jenis kekerasan utama: kekerasan seksual turun menjadi 381 kasus, kekerasan fisik menjadi 174, dan kekerasan psikis menjadi 127. Namun, mulai muncul kasus eksploitasi (13 kasus) dan trafficking (5 kasus), sementara kekerasan lainnya meningkat sedikit menjadi 142 kasus.

Secara keseluruhan, data menunjukkan tren penurunan kasus kekerasan fisik, psikis, dan seksual selama tiga tahun, namun ada kemunculan kasus eksploitasi dan trafficking di 2019 serta peningkatan kecil pada kekerasan lainnya.

fig, ax = plt.subplots(figsize=(12,8))
plt.style.use('ggplot')
labels = ['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual',
          'Eksploitasi','Trafficking','Penelantaran','Kekerasan Lainnya']
x = np.arange(7)
datagroup_tjenis = np.transpose(datagrouptahun)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_tjenis.columns):
  bar = plt.bar(x+(a*width), datagroup_tjenis[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x-0, labels=datagroup_tjenis.index, rotation=15)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Jenis Kekerasan \nProvinsi Sumatera Utara', fontsize=22, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Jenis Kekerasan', fontsize=14)
plt.legend(ncol=3, fancybox=True, shadow=True, loc='upper right', title='Tahun')
plt.savefig('jumlah kasus berdasrkan jenis kekerasan.png')
plt.show()

datagroupusia=data[['Tahun','0-5 tahun','6-12 tahun','13-17 tahun']]
datagroupusia['Tahun'] = datagroupusia['Tahun'].dt.year
datausia = datagroupusia.groupby(['Tahun'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()
datausia


Data menunjukkan jumlah kasus kekerasan pada tiga kelompok usia berbeda selama tiga tahun, yaitu usia 0-5 tahun, 6-12 tahun, dan 13-17 tahun.

Pada tahun 2017, kasus kekerasan terbanyak terjadi pada kelompok usia 13-17 tahun dengan 485 kasus, diikuti oleh usia 6-12 tahun sebanyak 360 kasus, dan usia 0-5 tahun sebanyak 173 kasus.

Tahun 2018 mengalami peningkatan pada semua kelompok usia, dengan kasus tertinggi pada kelompok 6-12 tahun (394 kasus), diikuti kelompok 13-17 tahun (525 kasus), dan 0-5 tahun (194 kasus).

Pada tahun 2019, jumlah kasus pada ketiga kelompok usia menurun secara signifikan, terutama di kelompok usia 13-17 tahun yang turun menjadi 324 kasus, kemudian 6-12 tahun menjadi 279 kasus, dan 0-5 tahun menjadi 134 kasus.

Secara umum, kelompok usia remaja (13-17 tahun) dan anak-anak usia sekolah dasar (6-12 tahun) merupakan kelompok dengan jumlah kasus kekerasan tertinggi, dengan tren menurun pada tahun 2019.

fig, ax = plt.subplots(figsize=(11,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
datagroup_age = np.transpose(datausia)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_age.columns):
  bar = plt.bar(x+(a*width), datagroup_age[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.2, labels=datagroup_age.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia \n Provinsi Sumatera Utara', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Usia Korban', fontsize=14, labelpad=10)
plt.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', title='Tahun')
plt.savefig('jumlah kasus berdasarkan usia.png')
plt.show()

datagroupjk=data[['Tahun','Jenis Kelamin','Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']]
datagroupjk['Total'] = datagroupjk[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']].sum(axis=1)
datagroupjk = datagroupjk.groupby(['Jenis Kelamin'])[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']].sum()
datagroupjk

Data menunjukkan perbandingan jumlah kasus kekerasan fisik, psikis, seksual, penelantaran, dan kekerasan lainnya antara laki-laki dan perempuan.

Kekerasan Fisik lebih banyak dialami oleh laki-laki dengan 377 kasus, dibandingkan perempuan sebanyak 315 kasus.

Untuk Kekerasan Psikis, perempuan mengalami jumlah kasus yang lebih tinggi, yaitu 373 kasus, sementara laki-laki hanya 206 kasus.

Kasus Kekerasan Seksual sangat dominan dialami oleh perempuan, dengan jumlah kasus mencapai 1305, jauh lebih tinggi dibandingkan laki-laki yang hanya 172 kasus.

Pada kategori Penelantaran, kasusnya relatif seimbang antara laki-laki (225 kasus) dan perempuan (228 kasus).

Untuk Kekerasan Lainnya, perempuan juga mengalami lebih banyak kasus (219) dibandingkan laki-laki (180).

Secara keseluruhan, perempuan lebih banyak menjadi korban kekerasan psikis, seksual, dan kekerasan lainnya, sedangkan laki-laki lebih banyak mengalami kekerasan fisik.

fig, ax = plt.subplots(figsize=(12,7))
plt.style.use('ggplot')
labels = ['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Lainnya']
x = np.arange(5)
datagroup_jkt = np.transpose(datagroupjk)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_jkt.columns):
  bar = plt.bar(x+(a*width), datagroup_jkt[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.2, labels=datagroup_jkt.index)
ax.set_xticklabels(labels)
ax.set_title('Jenis Kekerasan \nBerdasarkan Jenis Kelamin Provinsi Sumatera Utara', fontsize=25, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Jenis Kekerasan', fontsize=14, labelpad=10)
plt.legend(ncol=3, fancybox=True)

data17 = data[data['Tahun'] == '2017-01-01']
usia17=data17[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia17 = usia17.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()


fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia17t = np.transpose(usia17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia17t.columns):
  bar = plt.bar(x+(a*width), usia17t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia17t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2017 \n Provinsi Sumatera Utara', fontsize=22, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.68,-0.12), title='Jenis Kelamin')
# Change bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2017.png', bbox_inches='tight')
plt.show()

data18 = data[data['Tahun'] == '2018-01-01']
usia18=data18[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia18 = usia18.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()
data18[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]

Terdapat variasi jumlah kasus di setiap kelompok usia dan jenis kelamin.

Pada beberapa baris, perempuan menunjukkan angka kasus yang lebih tinggi di kelompok usia 13-17 tahun, misalnya baris 67 dan 129, yang memiliki jumlah kasus cukup signifikan.

Kasus pada kelompok usia 0-5 tahun cenderung lebih kecil, baik untuk laki-laki maupun perempuan.

Untuk kelompok usia 6-12 tahun dan 13-17 tahun, terdapat beberapa baris dengan angka yang cukup tinggi, terutama untuk perempuan di usia remaja (13-17 tahun).

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia18t = np.transpose(usia18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia18t.columns):
  bar = plt.bar(x+(a*width), usia18t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia18t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2018 \n Provinsi Sumatera Utara', fontsize=21, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14, labelpad=3)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.69,-0.12), title='Jenis Kelamin')
# Changed bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2018.png', bbox_inches='tight')
plt.show()

data19 = data[data['Tahun'] == '2019-01-01']
usia19=data19[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia19 = usia19.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia19t = np.transpose(usia19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia19t.columns):
  bar = plt.bar(x+(a*width), usia19t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia19t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2019 \n Provinsi Sumatera Utara', fontsize=21, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.69,-0.12), title='Jenis Kelamin')
# Changed bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2019.png', bbox_inches='tight')
plt.show()

datatempat17 = data[data['Tahun'] == '2017-01-01']
tempat17=datatempat17[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya']]
tempat17 = tempat17.groupby(['Jenis Kelamin'])[['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']
x = np.arange(4)
tempat17_t = np.transpose(tempat17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat17_t.columns):
  bar = plt.bar(x+(a*width), tempat17_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat17_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2017', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datatempat18 = data[data['Tahun'] == '2018-01-01']
tempat18=datatempat17[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya']]
tempat18 = tempat18.groupby(['Jenis Kelamin'])[['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']
x = np.arange(4)
tempat18_t = np.transpose(tempat18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat18_t.columns):
  bar = plt.bar(x+(a*width), tempat18_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat18_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2018', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datatempat19 = data[data['Tahun'] == '2019-01-01']
tempat19=datatempat19[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya','Fasilitas Umum']]
tempat19 = tempat19.groupby(['Jenis Kelamin'])[['Rumah','Tempat Bekerja','Sekolah','Lainnya','Fasilitas Umum']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Tempat Bekerja','Sekolah','Lainnya','Fasilitas Umum']
x = np.arange(5)
tempat19_t = np.transpose(tempat19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat19_t.columns):
  bar = plt.bar(x+(a*width), tempat19_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat19_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2019', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan17 = data[data['Tahun'] == '2017-01-01']
layanan17=datalayanan17[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan17 = layanan17.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Rehabilitasi','Pemulangan dan \nReintegrasi','Pendampingan','Mediasi']
x = np.arange(7)
layanan17_t = np.transpose(layanan17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan17_t.columns):
  bar = plt.bar(x+(a*width), layanan17_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan17_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2017', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan18 = data[data['Tahun'] == '2018-01-01']
layanan18=datalayanan18[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan18 = layanan18.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Rehabilitasi','Pemulangan dan \nReintegrasi','Pendampingan','Mediasi']
x = np.arange(7)
layanan18_t = np.transpose(layanan18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan18_t.columns):
  bar = plt.bar(x+(a*width), layanan18_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan18_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2018', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan19 = data[data['Tahun'] == '2019-01-01']
layanan19=datalayanan19[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan19 = layanan19.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan \nReintegrasi Sosial','Pendampingan','Mediasi']
x = np.arange(8)
layanan19_t = np.transpose(layanan19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan19_t.columns):
  bar = plt.bar(x+(a*width), layanan19_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan19_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2019', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

# Data Pre-Processing

## Data Cleaning

missing_values = data.isnull()

print('Check Missing Values:\n')
for column in missing_values.columns.values.tolist():
  print(column)
  print(missing_values[column].value_counts())
  print('')

Dari hasil pemeriksaan tersebut, dapat disimpulkan bahwa tidak ada nilai yang hilang di seluruh kolom dataset. Semua kolom, mulai dari variabel tanggal ('Tahun'), kategori lokasi ('Kabupaten/Kota'), berbagai jenis kekerasan, kelompok usia, sampai variabel pendukung lain seperti 'Jenis Kelamin', memiliki 198 baris data lengkap tanpa ada nilai kosong.

from scipy import stats
datazscore = data.copy()
datazscore.drop(['Kabupaten/Kota','Jenis Kelamin','Rekan Kerja','Tahun','Na'], axis=1, inplace=True)

# Find the outliers using Z score
zscore = np.abs(stats.zscore(datazscore))
print('Nilai Zscore:\n',zscore)
print('\n')

threshold = 3
thres_zscore = zscore>3
loc = np.where(thres_zscore)
print('Lokasi Outliers:\n',loc)
print('Jumlah Outliers Pada Dataset:\n', thres_zscore.sum())

Nilai Z-score menunjukkan seberapa jauh nilai sebuah data dari rata-rata dalam satuan standar deviasi. Biasanya, data dengan |Z-score| > 3 dianggap outlier.

Dari dataset ini, ditemukan 146 data yang dikategorikan sebagai outlier.

Lokasi outlier ditampilkan sebagai indeks baris dan kolom pada dataset Anda. Misalnya, baris ke-7 kolom ke-26, baris ke-8 kolom ke-18, dan seterusnya.

Jumlah 146 outlier dari total 198 data berarti sekitar 74% data mengandung nilai ekstrem. Ini cukup tinggi, yang bisa jadi menunjukkan:

Data memang memiliki banyak nilai ekstrim yang perlu diperhatikan atau dianalisis lebih lanjut.

Atau mungkin distribusi data sangat tidak normal, sehingga deteksi outlier menggunakan Z-score kurang cocok tanpa transformasi data terlebih dahulu.

## Encoding

encoder = LabelEncoder()
data['Kabupaten/Kota'] = encoder.fit_transform(data['Kabupaten/Kota'])
data['Jenis Kelamin'] = encoder.fit_transform(data['Jenis Kelamin'])
print('Data setelah LabelEncoder:')
data.head(10)

Data yang telah melalui proses LabelEncoder ini menunjukkan bahwa variabel kategori seperti 'Kabupaten/Kota' dan 'Jenis Kelamin' telah diubah menjadi bentuk numerik (misalnya 'Kabupaten/Kota' menjadi angka 0, 1, 2, dst., dan 'Jenis Kelamin' menjadi 0 dan 1).

Beberapa poin analisis terkait data ini:

Tanggal Tahun tetap dalam format datetime (2017-01-01), memungkinkan analisis waktu.

Semua fitur lain sudah dalam bentuk numerik, memudahkan penerapan algoritma machine learning yang memerlukan input numerik.

Data ini berisi berbagai jenis kekerasan dan kategori demografis (kelompok usia dan jenis kelamin).

LabelEncoder memungkinkan model mengolah data kategorikal tanpa kehilangan informasi yang sebelumnya berbentuk teks.

Contoh baris pertama menunjukkan data untuk satu entri dengan nilai-nilai numerik pada semua kolom fitur, seperti jumlah kasus kekerasan fisik 4, kekerasan psikis 0, dan jenis kelamin dikodekan sebagai 0 (misal laki-laki).

# Data Transformation

## Feature Selection

data.head()

1. Distribusi Kasus Kekerasan per Jenis (2017) :
Kekerasan Psikis dan Kekerasan Seksual terlihat sering muncul, contohnya baris kedua dan ketiga kasusnya cukup banyak.

Kekerasan Fisik juga signifikan, terutama pada baris ke-5 (25 kasus).

2. Distribusi Kasus Berdasarkan Usia
Ada kasus di rentang usia 0-5 tahun, 6-12 tahun, dan 13-17 tahun.

usia 6-12 dan 13-17 tahun sering lebih banyak kasusnya dibandingkan usia 0-5 tahun.

3. Distribusi Kasus Berdasarkan Lokasi
Data lokasi menunjukkan kasus tersebar di beberapa tempat seperti Rumah, Tempat Bekerja, Lingkungan Masyarakat, dll.

Biasanya rumah dan sekolah adalah lokasi yang cukup sering muncul pada kasus kekerasan anak.

4. Jenis Kelamin
Ada kolom jenis kelamin yang sudah di-encode (0 dan 1).

Dari data summary sebelumnya, perempuan cenderung mengalami lebih banyak kekerasan seksual dibanding laki-laki.

5. Outliers
Dari hasil Z-score, terdapat 146 outlier yang tersebar di beberapa baris dan kolom.

Outliers ini kemungkinan besar adalah kasus yang sangat tinggi pada satu atau beberapa variabel tertentu, bisa jadi data kasus yang ekstrem.



data = data.drop(['Kabupaten/Kota','Tahun','Kekerasan Lainnya','Lainnya'], axis=1)
datadrop=data.iloc[:,9:24]
data = data.drop(datadrop, axis=1)
data.head()

Berdasarkan langkah pembersihan data yang dilakukan, dataset ini telah melalui proses seleksi fitur dengan menghapus kolom-kolom yang dianggap kurang relevan atau berlebihan untuk analisis, seperti informasi geografis (Kabupaten/Kota), waktu (Tahun), serta beberapa kategori kekerasan dan atribut lain yang tidak spesifik (Kekerasan Lainnya dan Lainnya). Selain itu, sejumlah kolom lain yang berada di indeks 9 hingga 23 juga dihapus, yang kemungkinan besar berisi variabel yang berkorelasi tinggi atau tidak memiliki kontribusi signifikan dalam analisis lebih lanjut. Dengan mengurangi jumlah kolom tersebut, data menjadi lebih ringkas dan fokus pada variabel utama yang berpotensi memberikan insight lebih mendalam mengenai pola kekerasan dan karakteristik korban berdasarkan variabel seperti jenis kekerasan (fisik, psikis, seksual, dll), rentang usia korban, serta faktor pendukung lainnya. Proses ini penting untuk meminimalkan noise dan meningkatkan efisiensi dalam analisis statistik atau pemodelan machine learning, sehingga hasil yang diperoleh lebih valid dan mudah diinterpretasikan

# Data Modelling

X = data.drop(['Jenis Kelamin'], axis=1)
y = data[['Jenis Kelamin']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print('X_train shape :\n', X_train.shape)
print('X_test shape :\n', X_test.shape)
print('Baris X_train shape:\n', X_train.shape[0])
print('Kolom X_train shape:\n', X_train.shape[1])

print('Baris X_test shape:\n', X_test.shape[0])
print('Kolom X_test shape:\n', X_test.shape[1])

Berdasarkan hasil dari output ukuran data X_train dan X_test, dapat dianalisis bahwa data pelatihan (X_train) memiliki jumlah baris yang lebih banyak dibandingkan data pengujian (X_test). Hal ini menunjukkan bahwa data sudah dibagi dengan proporsi yang tepat antara pelatihan dan pengujian, yang umumnya mengikuti pembagian standar seperti 70:30 atau 80:20, sehingga model dapat belajar dari sebagian besar data dan diuji pada data yang belum pernah dilihat sebelumnya. Selain itu, jumlah kolom pada kedua dataset adalah sama, yang berarti fitur yang digunakan untuk pelatihan dan pengujian konsisten dan tidak ada kehilangan variabel penting selama pembagian data. Dengan bentuk data seperti ini, proses pelatihan model diharapkan dapat berjalan optimal karena representasi fitur lengkap dan jumlah data pelatihan cukup memadai untuk menangkap pola dalam dataset. Selanjutnya, data pengujian yang cukup juga memungkinkan evaluasi model secara valid dan akurat terhadap performa generalisasi model pada data baru.









print('y_train shape :\n', y_train.shape)
print('y_test shape :\n', y_test.shape)
print('Baris y_train shape:\n', y_train.shape[0])
print('Baris y_train shape:\n', y_test.shape[0])

## Data Transformation (Normalization)

scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

print('Normalisasi X_train 80%:\n', X_train_scaler)
print('Normalisasi X_test 20%:\n', X_test_scaler)

Hasil normalisasi data pada X_train dan X_test menunjukkan bahwa nilai fitur telah diubah ke dalam skala yang seragam, biasanya dalam rentang 0 hingga 1. Hal ini penting dilakukan untuk memastikan bahwa setiap fitur memiliki kontribusi yang seimbang saat digunakan dalam model pembelajaran mesin, terutama yang sensitif terhadap skala data seperti KNN, SVM, atau regresi logistik. Pada X_train, nilai-nilai fitur tersebar dengan variasi yang cukup kecil dan proporsional, yang menandakan bahwa data sudah siap untuk proses pelatihan model tanpa adanya bias akibat skala fitur yang berbeda. Sementara itu, pada X_test, meskipun secara umum juga berada pada rentang yang sama, terlihat adanya beberapa nilai yang lebih besar dari 1 (misalnya 1.25 dan 1.0869), yang bisa mengindikasikan adanya nilai ekstrem atau outlier pada subset pengujian. Namun, ini tidak selalu negatif asalkan sesuai dengan distribusi asli data dan model mampu menangani variasi tersebut.

## Data Modelling (Implementation Algorithm)

### Split 80%:20%

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaler, y_train.values.ravel())

Model Gaussian Naive Bayes yang diterapkan menggunakan data training yang sudah melalui proses normalisasi bertujuan untuk mengklasifikasikan data berdasarkan probabilitas fitur-fitur yang diasumsikan mengikuti distribusi Gaussian (normal). Proses pelatihan model ini melibatkan estimasi parameter-parameter distribusi (rata-rata dan varians) dari setiap fitur untuk tiap kelas target, sehingga model dapat memprediksi kelas baru berdasarkan probabilitas gabungan dari fitur-fiturnya. Penggunaan normalisasi pada fitur input (X_train_scaler) membantu memastikan fitur memiliki skala yang konsisten, yang dapat meningkatkan kinerja model Naive Bayes meskipun model ini secara teoritis kurang sensitif terhadap skala fitur. Dengan demikian, pelatihan model ini siap untuk digunakan dalam prediksi klasifikasi pada data testing guna mengukur performa model dalam konteks masalah klasifikasi yang sedang Anda kerjakan.

gnbscore_train = gnb.score(X_train_scaler, y_train)
print('Accuracy of Gaussian NB classifier on training set: {:.2f}'
     .format(gnbscore_train))

gnbscore_test = gnb.score(X_test_scaler, y_test)
print('Accuracy of Gaussian NB classifier on test set: {:.2f}'
     .format(gnbscore_test))

scores = cross_val_score(gnb, X, y.values.ravel(), cv=10)
print('Keakuratan Gaussian NB dengan dataset:', scores.mean())
print('%0.2f akurasi dengan standar deviasi sebesar %0.2f' %(scores.mean(), scores.std()))

Hasil evaluasi model Gaussian Naive Bayes menunjukkan bahwa akurasi pada data training sebesar 65% dan pada data testing sebesar 70%. Ini mengindikasikan bahwa model cukup baik dalam mengklasifikasikan data baru, dengan performa yang sedikit lebih baik pada data testing dibandingkan data training. Nilai akurasi sekitar 65% dengan standar deviasi 0.06 juga menunjukkan bahwa hasil model relatif konsisten dan stabil, tanpa fluktuasi besar pada akurasi di berbagai percobaan atau subset data. Secara keseluruhan, model ini memiliki kemampuan prediksi yang memadai, walaupun masih ada ruang untuk peningkatan, misalnya dengan mengoptimasi fitur, menggunakan teknik pemilihan fitur, atau mencoba algoritma klasifikasi lain untuk memperoleh hasil yang lebih baik.









print('0=Laki-laki\n1=Perempuan\n')
print('Peluang tiap class pada dataset:\n', gnb.class_prior_)

Distribusi kelas pada dataset menunjukkan bahwa proporsi data laki-laki (label 0) adalah sekitar 48,7%, sedangkan perempuan (label 1) sekitar 51,3%. Hal ini berarti dataset relatif seimbang antara kedua kelas tersebut, dengan sedikit lebih banyak data perempuan daripada laki-laki. Kondisi ini baik untuk proses pelatihan model karena keseimbangan kelas dapat membantu model belajar tanpa bias ke salah satu kelas, sehingga prediksi yang dihasilkan cenderung lebih adil dan akurat dalam mengklasifikasikan kedua kategori tersebut.

print('Mean tiap feature: \n', gnb.theta_)
print('\nVariansi tiap feature: \n', gnb.var_)

gnb.theta_ → menyimpan mean (rata-rata) tiap fitur untuk setiap kelas (baris pertama untuk kelas 0, baris kedua untuk kelas 1, dst.)

gnb.var_ → menyimpan variansi tiap fitur untuk setiap kelas

y_pred_gaussian = gnb.predict(X_test_scaler)
print('Angka kesalahan label point dari total %d points : %d' %(X_test_scaler.shape[0], (y_test.values.ravel() != y_pred_gaussian).sum()))

Dari hasil prediksi menggunakan model Gaussian Naive Bayes pada data uji sebanyak 40 titik data, ditemukan bahwa model melakukan kesalahan prediksi pada 12 titik data. Artinya, tingkat kesalahan model adalah 30%, yang berarti akurasi model pada data uji sebesar 70%. Hal ini menunjukkan bahwa meskipun model mampu memprediksi dengan tingkat keberhasilan yang cukup baik, masih ada sekitar sepertiga data yang tidak berhasil diklasifikasikan dengan benar. Kesalahan ini bisa disebabkan oleh beberapa faktor, seperti ketidaksesuaian asumsi distribusi Gaussian pada fitur, overlap antar kelas, atau fitur yang kurang informatif.

y_pred_gaussian_ds = gnb.predict([[0.608696,0.217391,0,0.25,0,
                                   0.041667,0,0.166667,0.125,
                                   0.170732,0,0.032609,0,0.090909,
                                   0,0,0]])
print('Prediksi Data Sampel Pertama:', y_pred_gaussian_ds)

y_pred_gauss_ds2 = gnb.predict([[1,1.086957,0.052632,0,0,0.541667,
                                 0.791667,0.6875,0.458333,0.012195,
                                 0,0.01087,0,0,0,0,0]])
print('Prediksi Data Sampel Kedua:', y_pred_gauss_ds2)

Dari hasil prediksi dua sampel data menggunakan model Gaussian Naive Bayes, keduanya diprediksi termasuk dalam kelas 1 (Perempuan). Artinya, berdasarkan fitur-fitur yang diberikan pada masing-masing sampel, model memperkirakan bahwa kedua data tersebut lebih cenderung milik kelas perempuan daripada laki-laki.

Prediksi ini didasarkan pada probabilitas tertinggi yang dihitung model dari distribusi Gaussian setiap fitur untuk masing-masing kelas. Dengan kata lain, karakteristik nilai fitur sampel tersebut lebih sesuai dengan pola fitur yang dimiliki kelas perempuan menurut model.

print('Prediksi seluruh data sampel:\n', y_pred_gaussian)
print('Seluruh data aktual:\n', y_test)

Prediksi model (y_pred_gaussian) adalah array label kelas yang diprediksi oleh Gaussian Naive Bayes untuk setiap sampel di data uji.

Data aktual (y_test) adalah label sebenarnya dari masing-masing sampel tersebut.

Jika kita lihat, prediksi model tidak selalu cocok dengan label aktual. Misalnya, pada indeks pertama, model memprediksi kelas 1 (Perempuan), tetapi label sebenarnya adalah 0 (Laki-laki). Begitu juga beberapa indeks lain, ada ketidaksesuaian antara prediksi dan label sebenarnya.

Hal ini menunjukkan bahwa model memiliki tingkat kesalahan dalam klasifikasi, yang sudah terlihat dari sebelumnya kamu menyebutkan ada 12 kesalahan prediksi dari total 40 data uji. Artinya model memiliki akurasi sekitar 70% pada data uji, yang menunjukkan performa model cukup baik tetapi masih ada ruang untuk perbaikan.

confusion_mat = confusion_matrix(y_test, y_pred_gaussian)
confusion_mat

>>
Baris pertama (kelas 0 = Laki-laki):

18 data laki-laki diklasifikasikan dengan benar sebagai laki-laki (True Negative).

4 data laki-laki salah diklasifikasikan sebagai perempuan (False Positive).

>>
Baris kedua (kelas 1 = Perempuan):

8 data perempuan salah diklasifikasikan sebagai laki-laki (False Negative).

10 data perempuan diklasifikasikan dengan benar sebagai perempuan (True Positive).



fig, ax = plt.subplots()
class_names=[0,1] # name  of classes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_mat), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix Data Split 80:20', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('confusionmatriks 8020.png')
plt.show()

Confusion matrix tersebut menunjukkan bahwa model klasifikasi ini memiliki performa yang lebih baik dalam mengenali kelas 0 dibandingkan kelas 1. Dari total 22 data kelas 0, model berhasil memprediksi dengan benar sebanyak 18 data (82% recall), sedangkan untuk kelas 1, dari 18 data hanya 10 yang berhasil diprediksi dengan tepat (56% recall). Hal ini menunjukkan bahwa model lebih akurat dalam mengidentifikasi kelas 0 dan kurang sensitif terhadap kelas 1. Selain itu, terdapat 8 data kelas 1 yang salah diklasifikasikan sebagai kelas 0 (false negative), sementara 4 data kelas 0 salah diklasifikasikan sebagai kelas 1 (false positive). Kesalahan yang lebih banyak pada false negative ini mengindikasikan model masih kesulitan mendeteksi sebagian data kelas 1. Secara keseluruhan, model mencapai akurasi sekitar 70%, yang tergolong cukup baik.

print('Pengujian Evaluasi Data Split 80:20')
print(classification_report(y_test, y_pred_gaussian))

>>
1. Precision

Kelas 0 (Laki-laki): 0.69

Artinya, dari semua data yang diprediksi sebagai laki-laki, 69% benar-benar laki-laki.

Kelas 1 (Perempuan): 0.71

Dari semua data yang diprediksi sebagai perempuan, 71% benar-benar perempuan.

>>
2. Recall

Kelas 0 (Laki-laki): 0.82

Model mampu mengenali 82% data laki-laki yang sebenarnya (True Positive Rate cukup tinggi).

Kelas 1 (Perempuan): 0.56

Model hanya mengenali 56% data perempuan yang sebenarnya, artinya masih banyak perempuan yang tidak terdeteksi (False Negative cukup tinggi).

>>
3. F1-score

Kelas 0 (Laki-laki): 0.75

Kelas 1 (Perempuan): 0.62

F1-score menggabungkan precision dan recall, nilai ini menunjukkan keseimbangan antara keduanya. Model lebih baik dalam mengenali laki-laki daripada perempuan.

>>
4. Accuracy

Akurasi keseluruhan adalah 0.70 atau 70%, artinya 70% prediksi model sesuai dengan label sebenarnya.

>>
5. Support

Jumlah data uji untuk kelas 0 = 22

Jumlah data uji untuk kelas 1 = 18

gnb.fit(X_train.values, y_train.values.ravel())
print('Training set score: {:.4f}'.format(gnb.score(X_train.values, y_train.values.ravel())))
print('Test set score: {:.4f}'.format(gnb.score(X_test.values, y_test.values.ravel())))

**Training Score (63.9%)**

Model mampu belajar dengan cukup baik dari data pelatihan, tapi tidak terlalu tinggi. Ini bisa jadi karena model GaussianNB dengan asumsi distribusi Gaussian untuk tiap fitur mungkin kurang cocok atau fitur yang digunakan belum maksimal.

Test Score (70.0%) **teks tebal**

Model justru performa di data test sedikit lebih baik daripada training. Ini menunjukkan model tidak mengalami overfitting (tidak terlalu “hapal” data training sampai mengorbankan generalisasi). Justru performa di test set lebih stabil.

**Interpretasi**

Model cenderung underfit (kurang fit dengan data training) karena skor training tidak terlalu tinggi, tapi tetap generalisasi cukup baik.

Mungkin fitur perlu diperkaya atau model yang lebih kompleks diperlukan untuk meningkatkan performa.

### Process of visualization the model

testpred = pd.DataFrame({'Nilai Test':y_test.values.ravel(),'Nilai Prediksi':y_pred_gaussian})

testpred[testpred['Nilai Prediksi']==1].count()

Nilai test dan nilai prediksi sama-sama 14 menunjukkan bahwa pada kasus tersebut, model berhasil memprediksi dengan tepat label kelas dari data uji yang diberikan. Ini berarti untuk sampel tersebut, model Gaussian Naive Bayes berhasil melakukan klasifikasi dengan benar.

predtest = pd.DataFrame({'Nilai Prediksi': ['0','1'], 'Total': [26,14], 'Nilai Test': ['0','1'],'Total Test':[22,18]}, index=[0,1])
predtestgroup = predtest.groupby(['Nilai Prediksi'])[['Total']].mean().reset_index()
pred_testgroup = predtest.groupby(['Nilai Test'])[['Total Test']].mean().reset_index()

valuespred20 = predtestgroup['Total']
valuesact20 = pred_testgroup['Total Test']

fig, ax = plt.subplots(figsize=(12,8))
explode = (0,0.1)
my_colors = ['#66b3ff','#ff9999']
piechart = ax.pie(predtestgroup['Total'], labels=predtestgroup['Nilai Prediksi'],
          autopct=lambda p:f'{p:.2f}%, {p*sum(valuespred20)/100 :.0f} Anak', explode=explode, shadow=True, colors=my_colors,
          startangle=0)

ax.set_title('Hasil Prediksi Analisis Data Kasus Kekerasan Anak \nVariasi 80:20', color='black',weight='bold',fontsize=20)

plt.legend(bbox_to_anchor = (1,0.6),labels=['Laki-laki','Perempuan'])

plt.savefig('visualisasi data akhir 8020.png')
plt.show()

>>
**Persentase Laki-laki (0): 65% (26 anak)**

Mayoritas prediksi model adalah anak laki-laki, yang menunjukkan bahwa dari total sampel data yang dianalisis, 65% diprediksi sebagai laki-laki. Ini bisa jadi mencerminkan distribusi data asli atau hasil dari model yang cenderung memprediksi kelas laki-laki lebih banyak.

>>
**Persentase Perempuan (1): 35% (14 anak)**

Sisanya, 35%, adalah anak perempuan. Proporsi ini lebih kecil dibandingkan anak laki-laki.

>>
**Warna dan visualisasi:**

Warna biru mewakili laki-laki, merah muda mewakili perempuan. Slice untuk perempuan sedikit diangkat (explode) yang menarik perhatian dan memperjelas perbedaan kedua kelas tersebut.

>>
**Kesimpulan:**

Model prediksi mendeteksi anak laki-laki sebagai kelas dominan dalam kasus kekerasan anak pada dataset yang dianalisis.

Jika data asli memang proporsional seperti ini, model sudah cukup representatif. Namun jika data asli lebih seimbang, model perlu dievaluasi apakah ada bias prediksi.

