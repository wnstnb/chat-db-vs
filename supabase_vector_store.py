from vanna.base import VannaBase
import pandas as pd
import json
import psycopg2
import os
from vanna.utils import deterministic_uuid
from openai import OpenAI

class Supabase_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        
        if config is None:
            config = {}
            
        self.config = config
        self.embedding_function = config.get("embedding_function")
        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 10))
        self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 10))
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 10))
        
        # Connection will be established later with connect_to_postgres
        self.conn = None
        
    def connect_to_postgres(self, host, dbname, user, password, port, **kwargs):
        """Connect to Postgres database"""
        # Save connection parameters for potential reconnection
        self._connection_params = {
            'host': host,
            'dbname': dbname,
            'user': user,
            'password': password,
            'port': port,
            **kwargs
        }
        
        self.conn = psycopg2.connect(**self._connection_params)
        
        # Create tables if they don't exist
        self._create_tables()
        
        # Define the function to run SQL queries
        def run_sql_postgres(sql):
            try:
                return pd.read_sql_query(sql, self.conn)
            except psycopg2.InterfaceError:
                # Connection might be closed, try to reconnect
                self.conn = psycopg2.connect(**self._connection_params)
                return pd.read_sql_query(sql, self.conn)
            except Exception as e:
                self.conn.rollback()
                raise e
        
        # Set the attributes that VannaBase.ask checks for
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres
        
    def _create_tables(self):
        """Create necessary tables for vector storage"""
        with self.conn.cursor() as cur:
            # Check if pgvector extension is installed
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create tables for SQL, DDL, and documentation
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vanna_sql (
                    id TEXT PRIMARY KEY,
                    document TEXT,
                    embedding vector(1536)
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vanna_ddl (
                    id TEXT PRIMARY KEY,
                    document TEXT,
                    embedding vector(1536)
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vanna_documentation (
                    id TEXT PRIMARY KEY,
                    document TEXT,
                    embedding vector(1536)
                );
            """)
            
            # Create indexes for similarity search
            cur.execute("CREATE INDEX IF NOT EXISTS vanna_sql_embedding_idx ON vanna_sql USING ivfflat (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS vanna_ddl_embedding_idx ON vanna_ddl USING ivfflat (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS vanna_documentation_embedding_idx ON vanna_documentation USING ivfflat (embedding vector_l2_ops);")
            
            self.conn.commit()
    
    def generate_embedding(self, data, **kwargs):
        """Generate embedding using OpenAI API"""
        if self.embedding_function:
            embedding = self.embedding_function([data])
            if len(embedding) == 1:
                return embedding[0]
            return embedding
        else:
            # If no embedding function is provided, use OpenAI's API directly
            # Get API key from config or environment variable
            api_key = self.config.get("api_key") if hasattr(self, "config") and self.config else None
            
            # Create a client instance
            client = OpenAI(api_key=api_key)
            
            response = client.embeddings.create(
                input=data,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
    
    def add_question_sql(self, question, sql, **kwargs):
        """Add question-SQL pair to the database"""
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        
        id = deterministic_uuid(question_sql_json) + "-sql"
        embedding = self.generate_embedding(question_sql_json)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vanna_sql (id, document, embedding) VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET document = %s, embedding = %s",
                    (id, question_sql_json, embedding, question_sql_json, embedding)
                )
                self.conn.commit()
            return id
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def add_ddl(self, ddl, **kwargs):
        """Add DDL to the database"""
        id = deterministic_uuid(ddl) + "-ddl"
        embedding = self.generate_embedding(ddl)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vanna_ddl (id, document, embedding) VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET document = %s, embedding = %s",
                    (id, ddl, embedding, ddl, embedding)
                )
                self.conn.commit()
            return id
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def add_documentation(self, documentation, **kwargs):
        """Add documentation to the database"""
        id = deterministic_uuid(documentation) + "-doc"
        embedding = self.generate_embedding(documentation)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vanna_documentation (id, document, embedding) VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET document = %s, embedding = %s",
                    (id, documentation, embedding, documentation, embedding)
                )
                self.conn.commit()
            return id
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_similar_question_sql(self, question, **kwargs):
        """Get similar question-SQL pairs"""
        embedding = self.generate_embedding(question)
        
        try:
            with self.conn.cursor() as cur:
                # Cast the embedding array to vector type
                cur.execute(
                    "SELECT document FROM vanna_sql ORDER BY embedding <-> %s::vector LIMIT %s",
                    (embedding, self.n_results_sql)
                )
                results = cur.fetchall()
                
            return [json.loads(row[0]) for row in results]
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_related_ddl(self, question, **kwargs):
        """Get related DDL statements"""
        embedding = self.generate_embedding(question)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT document FROM vanna_ddl ORDER BY embedding <-> %s::vector LIMIT %s",
                    (embedding, self.n_results_ddl)
                )
                results = cur.fetchall()
                
            return [row[0] for row in results]
        except Exception as e:
            self.conn.rollback()
            raise e

    def get_related_documentation(self, question, **kwargs):
        """Get related documentation"""
        embedding = self.generate_embedding(question)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT document FROM vanna_documentation ORDER BY embedding <-> %s::vector LIMIT %s",
                    (embedding, self.n_results_documentation)
                )
                results = cur.fetchall()
                
            return [row[0] for row in results]
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_training_data(self, **kwargs):
        """Get all training data"""
        df = pd.DataFrame()
        
        with self.conn.cursor() as cur:
            # Get SQL data
            cur.execute("SELECT id, document FROM vanna_sql")
            sql_data = cur.fetchall()
            
            if sql_data:
                documents = [json.loads(row[1]) for row in sql_data]
                ids = [row[0] for row in sql_data]
                
                df_sql = pd.DataFrame({
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                })
                df_sql["training_data_type"] = "sql"
                df = pd.concat([df, df_sql])
            
            # Get DDL data
            cur.execute("SELECT id, document FROM vanna_ddl")
            ddl_data = cur.fetchall()
            
            if ddl_data:
                documents = [row[1] for row in ddl_data]
                ids = [row[0] for row in ddl_data]
                
                df_ddl = pd.DataFrame({
                    "id": ids,
                    "question": [None for _ in documents],
                    "content": documents,
                })
                df_ddl["training_data_type"] = "ddl"
                df = pd.concat([df, df_ddl])
            
            # Get documentation data
            cur.execute("SELECT id, document FROM vanna_documentation")
            doc_data = cur.fetchall()
            
            if doc_data:
                documents = [row[1] for row in doc_data]
                ids = [row[0] for row in doc_data]
                
                df_doc = pd.DataFrame({
                    "id": ids,
                    "question": [None for _ in documents],
                    "content": documents,
                })
                df_doc["training_data_type"] = "documentation"
                df = pd.concat([df, df_doc])
        
        return df
    
    def remove_training_data(self, id, **kwargs):
        """Remove training data by ID"""
        if id.endswith("-sql"):
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM vanna_sql WHERE id = %s", (id,))
                self.conn.commit()
            return True
        elif id.endswith("-ddl"):
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM vanna_ddl WHERE id = %s", (id,))
                self.conn.commit()
            return True
        elif id.endswith("-doc"):
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM vanna_documentation WHERE id = %s", (id,))
                self.conn.commit()
            return True
        else:
            return False
    
    def reset_connection(self):
        """Reset the database connection if it's in a bad state"""
        if self.conn:
            try:
                self.conn.rollback()  # Try to rollback any pending transaction
            except:
                pass
            
            try:
                self.conn.close()  # Close the current connection
            except:
                pass
        
        # Reconnect using the saved connection parameters
        if hasattr(self, '_connection_params') and self._connection_params:
            self.conn = psycopg2.connect(**self._connection_params)
        else:
            raise Exception("Cannot reset connection: connection parameters not saved") 