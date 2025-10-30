# RELATÓRIO DE ANÁLISE EXPLORATÓRIA - SARIMAX

## CRISP-DM Fase 2: Data Understanding

**Data:** 2025-10-29 11:48:01

### 1. Estatísticas Básicas

| Métrica | Valor    |
| -------- | -------- |
| count    | 120      |
| mean     | 10308.81 |
| std      | 5018.10  |
| min      | 2503     |
| 25%      | 5153.50  |
| 50%      | 10580.50 |
| median   | 10580.50 |
| 75%      | 14279.00 |
| max      | 19367    |
| range    | 16864    |
| cv       | 48.68    |
| skewness | -0.04    |
| kurtosis | -1.42    |

### 2. Testes de Estacionariedade

- **ADF Test:** p-value = 0.9012 (Não-estacionária)
- **KPSS Test:** p-value = 0.0155

### 3. Decomposição Sazonal

- **Força da Sazonalidade:** 0.553
- **Modelo Recomendado:** MULTIPLICATIVE

### 4. Análise de Autocorrelação

- **Período Sazonal Detectado:** 12

### 5. Variáveis Exógenas 

- atendimento_pre_hospitalar
- busca_e_salvamento
- emissao_de_alvaras_de_licenca
- realizacao_de_vistorias
- roubo_de_veiculo
- combate_a_incendios
- homicidio_doloso
- tentativa_de_homicidio
- furto_de_veiculo
- roubo_de_carga
- morte_no_transito_ou_em_decorrencia_dele_exceto_homicidio_doloso
- estupro
- mortes_a_esclarecer_sem_indicio_de_crime
- pessoa_desaparecida
- suicidio
- roubo_seguido_de_morte_latrocinio
- trafico_de_drogas
- arma_de_fogo_apreendida
- roubo_a_instituicao_financeira
- mandado_de_prisao_cumprido
- lesao_corporal_seguida_de_morte
- apreensao_de_maconha
- feminicidio
- pessoa_localizada
- morte_por_intervencao_de_agente_do_estado

## Recomendações para Modelagem SARIMAX

### Configuração do auto_arima:

```python

from pmdarima import auto_arima


# Baseado na análise:

seasonal = True  # Sazonalidade detectada

seasonal_periods = 12  # Mensal

d = 1  # Série não-estacionária

exogenous = df[['atendimento_pre_hospitalar', 'busca_e_salvamento', 'emissao_de_alvaras_de_licenca', 'realizacao_de_vistorias', 'roubo_de_veiculo', 'combate_a_incendios', 'homicidio_doloso', 'tentativa_de_homicidio', 'furto_de_veiculo', 'roubo_de_carga', 'morte_no_transito_ou_em_decorrencia_dele_exceto_homicidio_doloso', 'estupro', 'mortes_a_esclarecer_sem_indicio_de_crime', 'pessoa_desaparecida', 'suicidio', 'roubo_seguido_de_morte_latrocinio', 'trafico_de_drogas', 'arma_de_fogo_apreendida', 'roubo_a_instituicao_financeira', 'mandado_de_prisao_cumprido', 'lesao_corporal_seguida_de_morte', 'apreensao_de_maconha', 'feminicidio', 'pessoa_localizada', 'morte_por_intervencao_de_agente_do_estado']]  # Features recomendadas

```
