# Roadmap — Bias in Embedding Models

## Status Atual

- 8 modelos avaliados (3 comerciais, 5 open-source)
- 2 idiomas (PT-BR, EN)
- 14 testes WEAT/SEAT (PT-BR), 5 testes (EN)
- Paper: 12 páginas, 4 tabelas, compilação funcional

## Próximos Passos

### Alto Impacto (melhoram qualidade do paper)

| # | Tarefa | Esforço | Prioridade |
|---|--------|---------|------------|
| 1 | Intervalos de confiança para effect sizes | Médio | Alta |
| 2 | Correção de Bonferroni para testes múltiplos | Baixo | Alta |
| 3 | Validação de listas de palavras (mais palavras por grupo) | Médio | Alta |

### Médio Impacto (expandem escopo)

| # | Tarefa | Esforço | Prioridade |
|---|--------|---------|------------|
| 4 | Novos modelos (Cohere, Voyage, mais recentes) | Baixo | Média |
| 5 | Bias interseccional (gênero × raça) | Médio | Média |
| 6 | Testes de sentença (SEAT) para pares IBGE | Médio | Média |
| 7 | Análise de correlação tamanho do modelo × viés | Baixo | Média |

### Baixo Impacto (refinamentos)

| # | Tarefa | Esforço | Prioridade |
|---|--------|---------|------------|
| 8 | Mais idiomas (espanhol, francês) | Alto | Baixa |
| 9 | Teste de robustez (variar nº de palavras por grupo) | Médio | Baixa |
| 10 | Gráficos adicionais (scatter PT-BR × EN) | Baixo | Baixa |

## Observações

- Itens 1-3 devem ser feitos antes de submeter a um venue
- Itens 4-7 são extensões naturais que fortalecem a contribuição
- Itens 8-10 são opcionais e dependem do escopo desejado
