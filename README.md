# Vanna.AI with Supabase Vector Storage

This project demonstrates how to use Vanna.AI with Supabase (PostgreSQL + pgvector) for vector storage instead of the default ChromaDB.

## Overview

By default, Vanna.AI uses ChromaDB for vector storage, which creates local files to persist embeddings. This implementation replaces ChromaDB with a custom vector store that uses Supabase's PostgreSQL database with the pgvector extension to store all vector embeddings and training data.

## Benefits of Using Supabase

1. **Centralized Storage**: All data is stored in your Supabase PostgreSQL database, eliminating the need for local files.
2. **Scalability**: PostgreSQL can handle large amounts of data with proper indexing.
3. **Persistence**: Data is reliably stored in your database and won't be lost if local files are deleted.
4. **Accessibility**: The same vector data can be accessed from multiple instances or deployments.
5. **Backup & Recovery**: Built-in PostgreSQL backup features can be used.

## Prerequisites

- Python 3.8+
- A Supabase instance with pgvector extension enabled
- PostgreSQL connection details
- OpenAI API key

## Installation

```bash
pip install pandas numpy psycopg2-binary openai python-dotenv vanna
```

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_HOST=your_supabase_host
SUPABASE_DBNAME=your_supabase_database_name
SUPABASE_USER=your_supabase_user
SUPABASE_PASSWORD=your_supabase_password
SUPABASE_PORT=your_supabase_port
```

## Project Structure

- `supabase_vector_store.py` - Custom implementation of Vanna's VectorStore interface using Supabase/PostgreSQL
- `start_server.py` - Entry point for starting the Vanna.AI web server
- `train_supabase.py` - Script to train the model by loading data into Supabase

## Usage

### 1. Enable pgvector in Supabase

Make sure the pgvector extension is enabled in your Supabase project. You can enable it by running the following SQL in the Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Train the Model

Fill in the necessary variables with relevant SQL question pairs, DDL and extraneous documentation. 
Then, run the training script to load example data into Supabase:

```bash
python train_supabase.py
```

This will create the necessary tables, add your SQL examples, DDL, and documentation to the Supabase database.

### 3. Start the Server

Run the server script to start the Vanna.AI web interface:

```bash
python start_server.py
```

The web interface will be available at http://localhost:8084

## How It Works

1. **Vector Storage**: The `Supabase_VectorStore` class uses PostgreSQL with pgvector to store and retrieve vector embeddings.
2. **Vector Tables**: Three tables are created in your Supabase database:
   - `vanna_sql` - Stores question-SQL pair embeddings
   - `vanna_ddl` - Stores database schema (DDL) embeddings
   - `vanna_documentation` - Stores documentation embeddings
3. **Training Process**: The training script:
   - Creates these tables if they don't exist
   - Adds vector indexes for similarity search
   - Generates embeddings for your training data
   - Stores the embeddings in the appropriate tables
4. **Query Process**: When you ask a question:
   - The question is embedded using the same embedding model
   - Similar vectors are retrieved from Supabase using vector similarity search
   - The most relevant context is provided to the LLM to generate SQL

## Customization

You can customize the implementation by modifying the following:
- Change the embedding model in the `generate_embedding` method
- Adjust the number of results returned in similarity searches
- Add more SQL examples, DDL, or documentation to improve results

## Troubleshooting

If you encounter issues:

1. **Connection Problems**: Check your Supabase credentials and make sure the pgvector extension is installed.
2. **Missing Embeddings**: Run the training script to ensure data is loaded into Supabase.
3. **Poor Results**: Add more examples or improve the quality of your training data.

## Conclusion

By using Supabase for vector storage, you eliminate the dependency on local ChromaDB files while gaining all the benefits of a robust, managed PostgreSQL database for your vector embeddings. 