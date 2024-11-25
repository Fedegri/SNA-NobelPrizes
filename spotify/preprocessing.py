import pandas as pd

# Carica il file CSV originale
file_path = 'filename.csv'  # Specifica il percorso del file CSV originale
df = pd.read_csv(file_path)

# Mantieni solo le colonne 'name' e 'artists'
df_filtered = df[['name', 'artists']]

# Dividi i campi "artists" in più righe per separare i collaboratori
df_expanded = df_filtered.assign(artists=df_filtered['artists'].str.split(', ')).explode('artists')

# Filtra le tracce con "feat" nel nome
feat_tracks = df_expanded[df_expanded['name'].str.contains('feat', case=False)]

# Genera un elenco unico di artisti per tracce con "feat"
artists_feat = feat_tracks['artists'].unique()
nodes_df = pd.DataFrame({
    'Id': range(len(artists_feat)),
    'Label': artists_feat
})

# Mappa artisti a ID
artist_to_id = {artist: idx for idx, artist in enumerate(artists_feat)}

# Genera archi per collaborazioni nelle tracce con "feat"
collaborations = feat_tracks.groupby('name')['artists'].apply(list)
edges = []
for collaborators in collaborations:
    if len(collaborators) > 1:
        # Crea coppie di artisti (collaboratori)
        pairs = [(artist_to_id[a], artist_to_id[b]) for i, a in enumerate(collaborators) for b in collaborators[i + 1:]]
        edges.extend(pairs)

# Converti gli archi in un DataFrame e aggrega i pesi
edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
edges_df['Weight'] = 1
edges_df = edges_df.groupby(['Source', 'Target'], as_index=False).sum()

# Rimuovi gli archi che puntano a sé stessi
edges_df = edges_df[edges_df['Source'] != edges_df['Target']]

# Aggiungi la colonna 'Type' per Gephi
edges_df['Type'] = 'Undirected'

# Salva i nuovi file
nodes_path = 'spotify_nodes.csv'
edges_path = 'spotify_edges.csv'

nodes_df.to_csv(nodes_path, index=False)
edges_df.to_csv(edges_path, index=False)

print(f"Nodi salvati in: {nodes_path}")
print(f"Archi salvati in: {edges_path}")