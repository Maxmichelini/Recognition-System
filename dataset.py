import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose

random.seed(42)

pd.set_option('display.max_columns', None)

class AppleBrowningDataset(Dataset):
    def __init__(self, excel_file:str, sheet_name:str, transform=None):
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)
        #print("Colonne disponibili nel DataFrame:", self.data.columns)

        # Converti l'oggetto Index delle colonne in una lista
        # columns_list = self.data.columns.tolist()

        # Stampa tutte le colonne
        # for col in columns_list:
        #     print(col)

        self.transform = transform

        # self.data = self.data.fillna(0)  

        # Seleziona tutte le colonne di magnitude e phase
        unnamed_columnsP = [f'Unnamed: {i}' for i in range(3, 202)]  # Trova le colonne 'Unnamed: i' per i da 3 a 201
        unnamed_columnsM = [f'Unnamed: {i}' for i in range(204, 403)]  # Trova le colonne 'Unnamed: i' per i da 204 a 402

        # magnitude_columns = [col for col in self.data.columns if 'Magnitude (ohm)' in col] + unnamed_columnsM
        # phase_columns = [col for col in self.data.columns if 'Phase Angel (degree)' in col] + unnamed_columnsP

        magnitude_columns = []
        start_addingM = False

        for col in self.data.columns:
            if 'Magnitude (ohm)' in col:
                start_addingM = True
            if start_addingM:
                if self.data[col].isna().all() or 'FELIX' in col:
                    break
                magnitude_columns.append(col)



        phase_columns = []
        start_addingP = False

        for col in self.data.columns:
            if 'Phase Angel (degree)' in col:
                start_addingP = True
            if start_addingP:
                if self.data[col].isna().all() or 'FELIX' in col:
                    break
                phase_columns.append(col)

        

        self.data = self.data.dropna(subset=magnitude_columns + phase_columns)


        #DEBUG contenuto delle colonne
        # print(self.data.iloc[:15, :])  # Mostra le prime 10 righe e tutte le colonne

        # print("Magnitude Columns:", magnitude_columns)
        # print("Phase Columns:", phase_columns)
        
        # print("Contenuto delle colonne 'Phase Angel (degree)':")
        # print(self.data[phase_columns])
        # print("Contenuto delle colonne 'Magnitude (ohm)':")
        # print(self.data[magnitude_columns])

        self.data = self.data[magnitude_columns + phase_columns + ['Internal Browning']]  # Filtra le colonne e riorganizzale nell'ordine desiderato
        self.data['Internal Browning'] = self.data['Internal Browning'].apply(lambda x: 0 if x == 'No Browning' else 1)  # Trasforma la colonna 'Internal Browning' in 0 o 1

        self.features = self.data[magnitude_columns + phase_columns].values  # Conserva le features senza trasformarle
        self.labels = self.data['Internal Browning'].values  # Conserva le etichette senza trasformarle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

if __name__ == '__main__':  # Assicura che il codice al suo interno venga eseguito solo se il file viene eseguito direttamente, non se viene importato come modulo in un altro file
    excel_file = "Full Data Analysis- file, new revised_Sundus 1.xlsx"  # Assicurati che il percorso sia corretto
    sheet_name = "Exp. 2"  # Assicurati che il nome del foglio sia corretto
    transform = None  # Non applicare alcuna trasformazione

    # Aggiungi un try-except per catturare errori e fare il debug
    try:
        dataset = AppleBrowningDataset(excel_file, sheet_name, transform=transform)
        #print("Numero di elementi nel dataset:", len(dataset))
        #print("Primo elemento del dataset:", dataset[0])
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    BS = 32
    dataLoader = DataLoader(dataset=dataset, batch_size=BS, shuffle=True)

    batch = next(iter(dataLoader))
    features, labels = batch
    # print(features)
    # print(labels)
