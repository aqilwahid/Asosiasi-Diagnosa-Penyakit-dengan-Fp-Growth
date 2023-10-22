# ------------Import Library------------------#
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------Global Directory------------#
directory_ori = "D:/UTY/Semester 6/Proyek Profesional/Data/2021.csv"
directory = "D:/UTY/Semester 6/Proyek Profesional/Data/Hasil Cluster/kec_"
# ---------------------------------------#
# ------------Global Daerah------------#
daerah = {
    "Ambunten": "0",
    "Arjasa": "1",
    "Batang - Batang": "2",
    "Batu Putih": "3",
    "Bluto": "4",
    "Dasuk": "5",
    "Dungkek": "6",
    "Gapura": "7",
    "Gayam": "8",
    "Guluk - Guluk": "9",
    "Kalianget": "10",
    "Kota Sumenep": "11",
    "Masalembu": "12",
    "Pasongsongan": "13",
    "Pragaan": "14",
    "Raas": "15",
    "Rubaru": "16",
    "Kabupaten Sumenep": "2021",
}
# --------------------------------------#
# ------------Global Preprocessing------------#
labelencoder = preprocessing.LabelEncoder()
data = pd.read_csv(directory_ori)
data = data.sort_values("Kecamatan", ascending=True).drop(["No_REG", "Alamat"], axis=1)
data["Kecamatan"] = labelencoder.fit_transform(data["Kecamatan"])

destination_folder = "D:/UTY/Semester 6/Proyek Profesional/Data/Hasil Cluster/"  # Destination folder for grouping

data.to_csv(
    destination_folder + "Kec_2021.csv", index=False
)  # Save origial dataset as Kec_2021

grouped_data = data.groupby("Kecamatan").agg(
    {"Usia": list, "Diagnosa": list}
)  # Grouping data based on 'Kecamatan' column and aggregating data

for kec, diag in grouped_data.iterrows():  # Save each group to csv file
    filename = "kec_{}.csv".format(kec)
    filepath = destination_folder + filename
    pd.DataFrame({"Usia": diag["Usia"], "Diagnosa": diag["Diagnosa"]}).to_csv(
        filepath, index=False
    )
# -------------------------------------------#


# -----------------------------------------------BREAK FUNCTION----------------------------------------------------------#
def importdata():
    data = pd.read_csv(directory_ori)
    st.write(data)


def preprocessing():
    pilihan = st.selectbox("Pilih halaman", daerah)
    # ----menampilkan konten berdasarkan pilihan daerah----#
    if pilihan in daerah:
        data_daerah = pd.read_csv(directory + daerah[pilihan] + ".csv")
        st.write(data_daerah)
    else:
        st.write("Pilihan tidak ada")
    # -----------------------------------------------------#


def transaction_encoder(dataset):
    unique_items = set()
    encoded_dataset = []

    # Mengumpulkan semua item unik dalam dataset
    for transaction in dataset:
        unique_items.update(transaction)

    # Membuat dictionary untuk memetakan setiap item ke indeks kolom
    item_to_column = {item: i for i, item in enumerate(unique_items)}

    # Mengkodekan setiap transaksi dalam bentuk biner
    for transaction in dataset:
        encoded_transaction = [0] * len(unique_items)
        for item in transaction:
            column_index = item_to_column[item]
            encoded_transaction[column_index] = 1
        encoded_dataset.append(encoded_transaction)

    return encoded_dataset, list(unique_items)


def transfrom():
    # ------------Transfrom------------#
    # Pemetaan Umur
    # Balita: usia 0-4 tahun
    # Anak-anak: usia 5-12 tahun
    # Remaja: usia 13-17 tahun
    # Dewasa: usia 18-59 tahun
    # Lansia: usia 60 tahun ke atas.
    data_usia = data["Usia"]
    data_diag = data["Diagnosa"]

    balita = []
    anak = []
    remaja = []
    dewasa = []
    lansia = []

    for i in range(len(data)):
        if data_usia[i] < 5:
            balita.append(data_diag[i])
        elif data_usia[i] >= 5 and data_usia[i] <= 12:
            anak.append(data_diag[i])
        elif data_usia[i] >= 13 and data_usia[i] <= 17:
            remaja.append(data_diag[i])
        elif data_usia[i] >= 18 and data_usia[i] <= 59:
            dewasa.append(data_diag[i])
        elif data_usia[i] >= 60:
            lansia.append(data_diag[i])

    # Membuat Dataset Baru
    dataset = [balita, anak, remaja, dewasa, lansia]
    te_ary, unique_items = transaction_encoder(dataset)
    df = pd.DataFrame(te_ary, columns=unique_items)
    st.write(df)


def fp_growth():
    # ----------Select Box-----------#
    pilihan = st.selectbox("Pilih halaman", daerah)
    trs = st.number_input(
        "Input Treshold", min_value=0.3, max_value=0.9, step=0.1, value=0.4
    )  # treshold
    # --------------------------------#
    # menampilkan konten berdasarkan pilihan daerah
    if pilihan in daerah:
        data_daerah = pd.read_csv(directory + daerah[pilihan] + ".csv")
        # --------Pengelompokan by Umur--------#
        data_usia = data_daerah["Usia"]
        data_diag = data_daerah["Diagnosa"]

        balita = []
        anak = []
        remaja = []
        dewasa = []
        lansia = []

        for i in range(len(data_daerah)):
            if data_usia[i] < 5:
                balita.append(data_diag[i])
            elif data_usia[i] >= 5 and data_usia[i] <= 12:
                anak.append(data_diag[i])
            elif data_usia[i] >= 13 and data_usia[i] <= 17:
                remaja.append(data_diag[i])
            elif data_usia[i] >= 18 and data_usia[i] <= 59:
                dewasa.append(data_diag[i])
            elif data_usia[i] >= 60:
                lansia.append(data_diag[i])
        # ---------------------------------------#
        # Membuat Dataset Baru
        dataset = [balita, anak, remaja, dewasa, lansia]
        te_ary, unique_items = transaction_encoder(dataset)
        df = pd.DataFrame(te_ary, columns=unique_items)
        # -------------------------------------#

        # -----------FP_Growth----------#
        res = fpgrowth(
            df, min_support=trs, use_colnames=True, verbose=1
        )  # menjalankan algoritma fp-growth
        res["itemsets"] = res["itemsets"].apply(
            lambda x: set(x)
        )  # mengubah frozenset menjadi set pada kolom itemsets
        ress = association_rules(
            res, metric="lift", min_threshold=1
        )  # membuat aturan asosiasi

        # Mengubah kolom antecedents dan consequents pada dataframe ress menjadi set
        ress["antecedents"] = ress["antecedents"].apply(lambda x: set(x))
        ress["consequents"] = ress["consequents"].apply(lambda x: set(x))

        # menampilkan Fp-growth & aturan asosiasi
        st.write(res)
        st.write(ress)
        # -------------------------------------#

    else:
        st.write("Pilihan tidak ada")


def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu Pengolahan Data RSUD Dr. H. Moh Anwar",  # Judul
            menu_icon="none",
            options=["Import Data", "Preprocessing", "Transfrom", "Fp-Growth"],
            icons=["-", "-", "-", "-"],
            default_index=0,
        )

    if selected == "Import Data":
        importdata()
        os.system("cls")
    if selected == "Preprocessing":
        preprocessing()
        os.system("cls")
    if selected == "Transfrom":
        transfrom()
        os.system("cls")
    if selected == "Fp-Growth":
        fp_growth()
        os.system("cls")


if __name__ == "__main__":
    main()
