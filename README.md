# RAG PoC Corporativo — FastAPI + ChromaDB + OpenAI + Streamlit + Docker

PoC de Retrieval-Augmented Generation (RAG) baseada em um projeto real, pronta para rodar localmente ou via Docker com uma UI simples em Streamlit. O objetivo é reduzir o tempo entre ideia e demo, oferecer um baseline técnico claro e facilitar a evolução colaborativa.

## Sumário
- Visão geral do RAG
- Funcionalidades
- Arquitetura
- Requisitos
- Variáveis de ambiente (.env)
- Execução com Docker (recomendado)
- Execução local (sem Docker)
- Estrutura do projeto
- API (formatos)
- Troubleshooting
- Roadmap
- Licença e uso interno

## Visão geral do RAG
RAG combina um LLM com uma camada de recuperação de contexto sobre documentos internos. Em vez de “alucinar”, o modelo responde com base em trechos relevantes recuperados de um índice vetorial e cita as fontes. Fluxo: ingestão → chunking → embeddings → indexação → recuperação top-k → re-ranking (opcional) → geração condicionada ao contexto.

## Funcionalidades
- Ingestão e chunking: leitura de PDFs/MD/TXT, chunks ~400 tokens, metadados enriquecidos (title, doc_id, section e, quando disponível, page e source).
- Indexação: embeddings com OpenAI (text-embedding-3-large/small) e persistência local no ChromaDB.
- Recuperação: consulta vetorial top-k na coleção persistida e formatação de fontes sem duplicatas.
- Geração com contexto: endpoint /ask monta prompt com os trechos recuperados e chama o LLM (gpt-4o-mini por padrão), retornando citações inline e uma lista de fontes com snippets.
- UI Streamlit: perguntas/respostas com exibição de fontes, status da API e métricas rápidas.
- Docker Compose: sobe API e UI com um comando, montando data/ e .env dentro dos containers.

## Arquitetura
``` mermaid
---
config:
  flowchart:
    htmlLabels: false
---
flowchart LR
  U[Usuário - UI/Swagger] -->|question| UI[Streamlit] -->|HTTP| B[FastAPI /ask]
  B --> C[Retriever<br/>ChromaDB query]
  C -->|top-k docs + metadados| D[Reranker opcional]
  D --> E[Prompt Builder<br/>Context + Question]
  E --> F[LLM Chat Completions]
  F -->|answer + citations| B --> UI

  subgraph Persistência
    G[ChromaDB<br/>data/chroma]
  end

  subgraph Indexação
    H[Ingestão + Chunking]
    I[Embeddings<br/>OpenAI]
    H --> I --> G
  end

  C <-- query_embeddings --> I
```
## Requisitos

* Linguagem/Runtime: Python 3.11, FastAPI, Uvicorn.
* Vetor store: ChromaDB (persistência local em data/chroma).
* Embeddings: OpenAI (text-embedding-3-large/small) — com opção futura de SentenceTransformers local.
* LLM: OpenAI Chat (gpt-4o-mini padrão, ajustável via .env).
* Docker Desktop (para execução com Docker).

## Variáveis de ambiente (.env)

Crie um arquivo .env na raiz do projeto com, por exemplo:
```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
CHROMA_PATH=data/chroma
CHROMA_COLLECTION=docs
```
Dica: salve o .env como UTF-8 sem BOM. O projeto possui fallback para BOM, mas o ideal é sem BOM.

## Execução com Docker (recomendado)

Pré-requisito: Docker Desktop rodando ("Docker Desktop is running").

1) Subir serviços (API + UI)
```
docker compose up --build -d
```
Verifique:
```
docker compose ps
```
Containers rag-api e rag-ui devem estar "Up".

2) Indexar documentos dentro do container da API (se ainda não indexou)

Coloque seus PDFs/MD/TXT em data/raw

Rode no container:
```
docker compose exec rag-api python -m src.index.
build_index
```
Confirme a coleção:
```
docker compose exec rag-api sh -c "ls -la /app/data/chroma"
```
3) Acessar

  * UI: http://localhost:8501
  * API (Swagger): http://localhost:8000/docs
  * Health: http://localhost:8000/health

Observações:

  * O compose monta ./data em /app/data e também monta o .env (somente leitura).
  * A UI dentro do Docker fala com a API pelo host rag-api (configurado via RAG_API_URL). No navegador, acesse a UI em http://localhost:8501.

## Estrutura do projeto

```
rag-genai-poc/
├─ src/
│  ├─ api/
│  │  └─ main.py            # FastAPI: /ask, retriever, prompt builder, formatação de fontes
│  ├─ generator/
│  │  └─ llm.py             # Wrapper de chat OpenAI; carrega .env de forma robusta
│  ├─ index/
│  │  └─ build_index.py     # Ingestão, chunking, embeddings, upsert no Chroma
│  ├─ ingest/               # Parsers utilitários, se aplicável
│  └─ ui/
│     └─ app.py             # UI Streamlit
├─ data/
│  ├─ raw/                  # Documentos de entrada; não versionados
│  ├─ processed/            # Opcional
│  └─ chroma/               # Persistência ChromaDB
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .env                     # não versionado
├─ .gitignore
└─ README.md
```

Dica: para evitar erro de import no container, inclua __init__.py em src/, src/index/ e src/ingest/, e defina PYTHONPATH=/app no serviço rag-api do docker-compose.

## API (formatos)

Request (POST /ask):

      {
      "question": "Quais são os principais pontos do documento X?"
      }
Resposta:
  * answer: texto ancorado em trechos dos documentos, com citações inline do tipo [Arquivo.pdf#pX-cY] quando a página for conhecida.
  * sources: lista com metadados (title, page, section, source, doc_id) e snippet recortado do chunk.

## Troubleshooting

* UI com erro “API Error: ... rag-api:8000 timeout”
  * Verifique se a API está “Up”: docker compose ps e docker compose logs -f rag-api.
  * Teste conectividade dentro do container da UI: docker compose exec rag-ui curl -s http://rag-api:8000/health.
  * Se rodar UI fora do Docker, use RAG_API_URL=http://localhost:8000.
* Swagger responde com sources: []
  * Índice inexistente: rode a indexação dentro do container: docker compose exec rag-api python -m src.index.build_index.
  * Confirme data/chroma populado: docker compose exec rag-api sh -c "ls -la /app/data/chroma".
* OPENAI_API_KEY not found
  * Garanta .env na raiz e montado no container (./.env:/app/.env:ro).
  * Salve .env como UTF-8 sem BOM.
* Erro de import “No module named src” no container
  * Inclua __init__.py em src/, src/index/, src/ingest/.
  * Defina PYTHONPATH=/app no docker-compose.yml do serviço rag-api.
  * Rode scripts como módulo: python -m src.index.build_index.
* Portas em uso
  * Edite docker-compose.yml e troque mapeamentos, por exemplo: "8080:8000" e "8601:8501". Depois: docker compose up -d --force-recreate.

## Roadmap

* Re-ranking com cross-encoder (SentenceTransformers) para melhor precisão.
* Filtros por metadados (título, recência, tipo) e boosting.
* Observabilidade: métricas de latência/custo, logging de Q/A em .jsonl.
* Avaliação: conjunto de Q/A, EM/F1, groundedness/faithfulness.
* Suporte a provedores alternativos (Azure OpenAI, embeddings locais).

## Licença e uso interno

PoC criada para uso interno/educacional e rápida validação. Adapte livremente para seus casos.  
Contribua enviando PRs e abrindo issues com ideias de melhoria. Se quiser discutir aplicações específicas, me chame!