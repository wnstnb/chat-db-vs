from supabase_vector_store import Supabase_VectorStore
from vanna.openai import OpenAI_Chat
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Define your training data here or do it in the Vanna UI
sql_pairs = [
    # {"question": "How many rows are in table 1?", "sql": "select count(*) from table1;"},
]

# Add all DDL here or do it in the Vanna UI, or both
all_ddl = None # """
# CREATE TABLE IF NOT EXISTS pages (
#     id BIGSERIAL PRIMARY KEY,
#     filename TEXT,
#     preprocessed TEXT,
#     page_number INTEGER,
#     image_width DOUBLE PRECISION,
#     image_height DOUBLE PRECISION,
#     lines TEXT,
#     words TEXT,
#     bboxes TEXT,
#     normalized_bboxes TEXT,
#     tokens TEXT,
#     words_for_clf TEXT,
#     processing_time DOUBLE PRECISION,
#     clf_type TEXT,
#     page_label TEXT,
#     page_confidence DOUBLE PRECISION,
#     created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
# );
# ...
# """

# Loose form information you want 
documentation_model_crosswalk = None # '''
# ...
# '''

# This will be your main training function
def train_vanna():
    # Create an instance of MyVanna that uses Supabase for vector storage
    class MyVanna(Supabase_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            Supabase_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)
    
    # Initialize with OpenAI API key
    vn = MyVanna(config={'api_key': os.getenv('OPENAI_API_KEY'), 'model': 'gpt-4o'})
    
    # Connect to Supabase
    vn.connect_to_postgres(
        host=os.getenv('SUPABASE_HOST'),
        dbname=os.getenv('SUPABASE_DBNAME'),
        user=os.getenv('SUPABASE_USER'),
        password=os.getenv('SUPABASE_PASSWORD'),
        port=os.getenv('SUPABASE_PORT')
    )
    
    # Train with SQL pairs
    print("Training with SQL pairs...")
    for pair in sql_pairs:
        vn.add_question_sql(pair["question"], pair["sql"])
    
    # Train with DDL
    print("Training with DDL...")
    vn.add_ddl(all_ddl)
    
    # Train with documentation
    print("Training with documentation...")
    vn.add_documentation(documentation_model_crosswalk)
    
    # Get training data to confirm it's loaded
    df_training = vn.get_training_data()
    print(f"Training data loaded: {len(df_training)} records")
    
    return vn

if __name__ == "__main__":
    print("Starting training...")
    vn = train_vanna()
    print("Training completed successfully!")
    
    # Test a simple query to confirm everything works
    try:
        result = vn.ask("what are the entity types and their counts?")
        print("Test query result:")
        print(result)
    except Exception as e:
        print(f"Error in test query: {str(e)}") 