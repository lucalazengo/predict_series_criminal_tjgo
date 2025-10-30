# 📚 Índice da Documentação do Projeto

## 🗺️ Navegação Rápida

### Para Começar

1. **[README.md](README.md)** ⭐
   - Visão geral do projeto
   - Instalação rápida
   - Estrutura do projeto
   - Links para todos os documentos

2. **[GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)** 📘 **RECOMENDADO PARA INICIANTES**
   - Instalação passo a passo detalhada
   - Como executar o pipeline
   - Troubleshooting completo
   - Exemplos práticos
   - Checklist de execução

### Documentação Técnica

3. **[DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)**
   - Arquitetura do sistema
   - Descrição detalhada de cada módulo
   - APIs e interfaces
   - Fluxo de dados

4. **[RELATORIO_EXECUCAO_FINAL.md](RELATORIO_EXECUCAO_FINAL.md)**
   - Resultados da execução
   - Métricas obtidas
   - Análise dos resultados

5. **[RESUMO_IMPLEMENTACOES.md](RESUMO_IMPLEMENTACOES.md)**
   - Resumo de todas as funcionalidades
   - Status de implementação
   - Melhorias realizadas

### Relatórios Técnicos

Os relatórios técnicos detalhados são gerados automaticamente após a execução do pipeline e ficam em:
- `outputs/reports/RELATORIO_DETALHADO_COMPLETO_*.md`

---

## 📋 Guia por Objetivo

### Quero executar o pipeline pela primeira vez

1. Leia: [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Seção "Instalação"
2. Siga: Passos 1-4 do guia
3. Execute: `python3 execute_pipeline.py`
4. Consulte: Seção "Interpretação dos Resultados"

### Encontrei um erro durante a execução

1. Consulte: [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Seção "Troubleshooting"
2. Verifique: Logs em `logs/prophet_pipeline.log`
3. Revise: [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md) para entender a arquitetura

### Quero entender como o projeto funciona

1. Leia: [README.md](README.md) - Visão geral
2. Estude: [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md) - Arquitetura
3. Explore: Código-fonte em `src/` com comentários detalhados

### Quero personalizar a configuração

1. Consulte: [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Seção "Configuração"
2. Revise: `configs/criminal_cases_config.yaml`
3. Teste: Execute com configuração personalizada

### Quero entender os resultados

1. Analise: Relatório HTML em `outputs/reports/report_*.html`
2. Leia: Relatório técnico em `outputs/reports/RELATORIO_DETALHADO_COMPLETO_*.md`
3. Consulte: [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Seção "Interpretação dos Resultados"

### Quero contribuir ou modificar o código

1. Entenda: [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md) - Arquitetura
2. Explore: Código em `src/` com documentação inline
3. Teste: Use `python -m pytest tests/`

---

## 🎯 Documentos por Tipo

### Guias Práticos
- [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Execução passo a passo
- [README.md](README.md) - Introdução e quick start

### Documentação Técnica
- [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md) - Arquitetura e design
- Código-fonte em `src/` - Documentação inline

### Relatórios
- [RELATORIO_EXECUCAO_FINAL.md](RELATORIO_EXECUCAO_FINAL.md) - Resultados de execução
- [RESUMO_IMPLEMENTACOES.md](RESUMO_IMPLEMENTACOES.md) - Status das implementações
- `outputs/reports/RELATORIO_DETALHADO_COMPLETO_*.md` - Relatórios técnicos gerados

---

## 🔍 Busca Rápida

### Instalação
→ [GUIA_EXECUCAO.md - Seção Instalação](GUIA_EXECUCAO.md#instalação)

### Execução
→ [GUIA_EXECUCAO.md - Seção Execução Básica](GUIA_EXECUCAO.md#execução-básica)

### Configuração
→ [GUIA_EXECUCAO.md - Seção Configuração](GUIA_EXECUCAO.md#configuração)
→ `configs/criminal_cases_config.yaml`

### Troubleshooting
→ [GUIA_EXECUCAO.md - Seção Troubleshooting](GUIA_EXECUCAO.md#troubleshooting)

### Arquitetura
→ [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)

### Métricas e Resultados
→ [GUIA_EXECUCAO.md - Seção Interpretação](GUIA_EXECUCAO.md#interpretação-dos-resultados)

---

## 📖 Ordem Recomendada de Leitura

### Para Usuários (Executar o Pipeline)

1. **README.md** (5 min) - Visão geral
2. **GUIA_EXECUCAO.md** (15 min) - Instruções completas
3. Execute o pipeline
4. Leia relatório gerado em `outputs/reports/`

### Para Desenvolvedores

1. **README.md** - Visão geral do projeto
2. **DOCUMENTACAO_TECNICA.md** - Arquitetura e design
3. Explore código em `src/` com documentação inline
4. **RESUMO_IMPLEMENTACOES.md** - Funcionalidades implementadas

### Para Analistas

1. **RELATORIO_EXECUCAO_FINAL.md** - Resultados obtidos
2. Relatórios técnicos em `outputs/reports/`
3. **GUIA_EXECUCAO.md** - Seção "Interpretação dos Resultados"

---

## 📝 Notas

- Todos os documentos estão em **português**
- Os relatórios técnicos são gerados automaticamente após executar o pipeline
- O código-fonte contém documentação inline detalhada
- Logs são salvos em `logs/prophet_pipeline.log`

---

**Última atualização:** Outubro 2025



