import pandas as pd
import numpy as np

def tratar_dataset_aeromotion(caminho_csv: str) -> pd.DataFrame:
    # 1. Carregamento com parsing temporal
    df = pd.read_csv(caminho_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Ordenação cronológica por equipamento
    df = df.sort_values(by=['id_equipamento', 'timestamp']).reset_index(drop=True)
    
    # 3. Tratamento de ruídos eletromagnéticos (conversão para NaN para posterior interpolação)
    df.loc[df['temperatura_mancal_c'] >= 900.0, 'temperatura_mancal_c'] = np.nan
    df.loc[df['vibracao_rms_mms'] < 0.0, 'vibracao_rms_mms'] = np.nan
    
    # 4. Interpolação linear temporal por equipamento
    colunas_continuas = [
        'temperatura_mancal_c', 'vibracao_rms_mms', 'pressao_oleo_bar',
        'rotacao_rpm', 'emissao_acustica_db', 'corrente_eletrica_amp'
    ]
    
    for col in colunas_continuas:
        df[col] = df.groupby('id_equipamento')[col].transform(
            lambda s: s.interpolate(method='linear').bfill().ffill()
        )
        
    # 5. Engenharia de Features (Rolling Features e Razão Física)
    # Janelas: 6 passos (1h de leituras do ativo) e 36 passos (6h)
    for equip, grupo in df.groupby('id_equipamento'):
        idx = grupo.index
        # Temperatura: médias e desvios
        df.loc[idx, 'temp_rolling_mean_1h'] = grupo['temperatura_mancal_c'].rolling(6, min_periods=1).mean()
        df.loc[idx, 'temp_rolling_std_1h'] = grupo['temperatura_mancal_c'].rolling(6, min_periods=1).std().fillna(0)
        df.loc[idx, 'temp_rolling_mean_6h'] = grupo['temperatura_mancal_c'].rolling(36, min_periods=1).mean()
        df.loc[idx, 'temp_rolling_std_6h'] = grupo['temperatura_mancal_c'].rolling(36, min_periods=1).std().fillna(0)
        
        # Vibração: médias e desvios
        df.loc[idx, 'vib_rolling_mean_1h'] = grupo['vibracao_rms_mms'].rolling(6, min_periods=1).mean()
        df.loc[idx, 'vib_rolling_std_1h'] = grupo['vibracao_rms_mms'].rolling(6, min_periods=1).std().fillna(0)
        df.loc[idx, 'vib_rolling_mean_6h'] = grupo['vibracao_rms_mms'].rolling(36, min_periods=1).mean()
        df.loc[idx, 'vib_rolling_std_6h'] = grupo['vibracao_rms_mms'].rolling(36, min_periods=1).std().fillna(0)
        
    # Severidade Dinâmica (Vibração RMS / RPM)
    df['severidade_dinamica'] = df['vibracao_rms_mms'] / df['rotacao_rpm']
    
    return df

if __name__ == "__main__":
    df_limpo = tratar_dataset_aeromotion(r"C:\Users\mateus_s_cavalheiro\Documents\GitHub\AeroMotion-Dynamics\data\raw\dados_sensores_aeromotion.csv")
    df_limpo.to_csv(r"C:\Users\mateus_s_cavalheiro\Documents\GitHub\AeroMotion-Dynamics\data\processed\dados_sensores_aeromotion_tratado.csv", index=False)