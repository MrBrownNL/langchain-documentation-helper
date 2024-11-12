# LangChain Documentation Helper

Copy the .env-example to .env and fill in your keys/models etc.
```shell
cp .env-example .env
```

## Init Pinecone vectorstore
Run the ingestion.py once to fill the vector database.

## Streamlit runner
Get the current Streamlit binary:
```shell
which streamlit
```
Add a run configuration, fill in the Streamlit path to the script field.
- Parameters: run main.py
- Working directory: project root path
- Envionment variables: All models/keys/tracing

Run the streamlit runner and the application will show up in the browser.
