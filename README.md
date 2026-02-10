# RAG-Financial-Named-Entity-Recognition
A retrieval-augmented financial NER system that extracts structured financial entities from large collections of financial news and reports.

## Installation and Launch

### Install dependencies

```bash
# Create virtual environment
py -3.11 -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database and GEMINI_API_KEY configuration

Create a `.env` file:

```env
DB_HOST= your_HOST
DB_PORT= your_PORT
DB_NAME= your_DB_NAME
DB_USER= your_DB_USER
DB_PASSWORD= your_PASSWORD

GEMINI_API_KEY = your_API
```

Create a `docker-compose.yml` file:
```yml
services:
  db:
    image: ankane/pgvector:latest
    container_name: your_DB_NAME
    restart: always
    environment:
      POSTGRES_DB: your_DB_NAME
      POSTGRES_USER:  your_DB_USER
      POSTGRES_PASSWORD: your_PASSWORD
    ports:
      - your_PORT
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Start the database

```bash
docker-compose up -d
```
#### Creating an extension

```bash
docker exec -it financial_ner_db psql -U postgres -d FinancialNER
```

---
## Running the Server

```bash
uvicorn app.main:app --reload
```

Server will be available at: `http://localhost:8000`

---

## API Documentation
`http://localhost:8000/docs`

---

## API Usage
### 1. Ingestion

**POST `upload_pdf`**

Select the required PDF file and click `Execute`

 ---
#### The results
```json
{
  "document_id": 12,
  "extracted_at": "2026-02-10T11:15:27.479121Z",
  "entities": [
    {
      "entity_type": "REVENUE",
      "value": "$57.0 billion",
      "id": 6
    },
    {
      "entity_type": "NET_INCOME",
      "value": "$31,910 million",
      "id": 7
    },
    {
      "entity_type": "EPS",
      "value": "$1.30",
      "id": 8
    },
    {
      "entity_type": "OPERATING_MARGIN",
      "value": "63.2%",
      "id": 9
    },
    {
      "entity_type": "CASH",
      "value": "$60.6 billion",
      "id": 10
    }
  ]
}
```
---


### 2. Chat

**POST `chat`**

```json
{
  "query": "What is this file about?"
}
```
---
#### The results
```json
{
  "answer": "This file is about the risks and uncertainties that could cause actual results to differ materially from expectations. It details important factors that could lead to such discrepancies, including:\n\n*   Global economic and political conditions\n*   Reliance on third parties for manufacturing, assembly, packaging, and testing\n*   The impact of technological development and competition\n*   Development of new products/technologies or enhancements to existing ones\n*   Market acceptance of products\n*   Design, manufacturing, or software defects\n*   Changes in consumer preferences or demands\n*   Changes in industry standards and interfaces\n*   Unexpected loss of performance of products when integrated into systems\n*   Changes in applicable laws",
  "sources": [
    "created by those sections based on management’s beliefs and assumptions and on information currently available to \nmanagement and are subject to risks and uncertainties that could cause results to be materially different than \nexpectations. Important factors that could cause actual results to differ materially include: global economic and political \nconditions; our reliance on third parties to manufacture, assemble, package and test our products; the impact of \ntechnological development and competition; development of new products and technologies or enhancements to our \nexisting product and technologies; market acceptance of our products or our partners’ products; design, manufacturing or \nsoftware defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected \nloss of performance of our products or technologies when integrated into systems; and changes in applicable laws and",
    "created by those sections based on management’s beliefs and assumptions and on information currently available to \nmanagement and are subject to risks and uncertainties that could cause results to be materially different than \nexpectations. Important factors that could cause actual results to differ materially include: global economic and political \nconditions; our reliance on third parties to manufacture, assemble, package and test our products; the impact of \ntechnological development and competition; development of new products and technologies or enhancements to our \nexisting product and technologies; market acceptance of our products or our partners’ products; design, manufacturing or \nsoftware defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected \nloss of performance of our products or technologies when integrated into systems; and changes in applicable laws and",
    "created by those sections based on management’s beliefs and assumptions and on information currently available to \nmanagement and are subject to risks and uncertainties that could cause results to be materially different than \nexpectations. Important factors that could cause actual results to differ materially include: global economic and political \nconditions; our reliance on third parties to manufacture, assemble, package and test our products; the impact of \ntechnological development and competition; development of new products and technologies or enhancements to our \nexisting product and technologies; market acceptance of our products or our partners’ products; design, manufacturing or \nsoftware defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected \nloss of performance of our products or technologies when integrated into systems; and changes in applicable laws and",
    "created by those sections based on management’s beliefs and assumptions and on information currently available to \nmanagement and are subject to risks and uncertainties that could cause results to be materially different than \nexpectations. Important factors that could cause actual results to differ materially include: global economic and political \nconditions; our reliance on third parties to manufacture, assemble, package and test our products; the impact of \ntechnological development and competition; development of new products and technologies or enhancements to our \nexisting product and technologies; market acceptance of our products or our partners’ products; design, manufacturing or \nsoftware defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected \nloss of performance of our products or technologies when integrated into systems; and changes in applicable laws and",
    "created by those sections based on management’s beliefs and assumptions and on information currently available to \nmanagement and are subject to risks and uncertainties that could cause results to be materially different than \nexpectations. Important factors that could cause actual results to differ materially include: global economic and political \nconditions; our reliance on third parties to manufacture, assemble, package and test our products; the impact of \ntechnological development and competition; development of new products and technologies or enhancements to our \nexisting product and technologies; market acceptance of our products or our partners’ products; design, manufacturing or \nsoftware defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected \nloss of performance of our products or technologies when integrated into systems; and changes in applicable laws and"
  ]
}
```
---
### 3. Analytics

**GET `dashboard`**

Click `Execute`

---
#### The results
```json
{
    "document_name": "Q3FY26-CFO-Commentary.pdf",
    "upload_date": "2026-02-10 07:09",
    "financials": [
      {
        "metric": "REVENUE",
        "value": "$57.0 billion"
      },
      {
        "metric": "NET_INCOME",
        "value": "$31.910 billion"
      },
      {
        "metric": "EPS",
        "value": "$1.30"
      },
      {
        "metric": "OPERATING_MARGIN",
        "value": "63.17%"
      },
      {
        "metric": "CASH",
        "value": "$60.6 billion"
      }
    ]
}
```