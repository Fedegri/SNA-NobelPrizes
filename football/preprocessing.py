import pandas as pd

# Percorsi dei file di input
clubs_path = 'clubs.csv'
transfers_path = 'transfers.csv'

# Carica i file CSV
clubs_df = pd.read_csv(clubs_path)
transfers_df = pd.read_csv(transfers_path)

# Creazione del file nodes.csv
# Utilizziamo club_id come id e name come label per Gephi
nodes = clubs_df[['club_id', 'name']].rename(columns={'club_id': 'id', 'name': 'label'})

# Creazione del file edges.csv
# Verifica che i club coinvolti nei trasferimenti esistano nel file dei club
valid_transfers = transfers_df[
    transfers_df['from_club_id'].isin(clubs_df['club_id']) & 
    transfers_df['to_club_id'].isin(clubs_df['club_id'])
]

# Raggruppa per coppia di club e conta i trasferimenti
edges = valid_transfers.groupby(['from_club_id', 'to_club_id']).size().reset_index(name='weight')

# Rinomina le colonne per Gephi
edges = edges.rename(columns={'from_club_id': 'source', 'to_club_id': 'target'})

# Salva il file edges.csv
edges.to_csv('edges.csv', index=False)