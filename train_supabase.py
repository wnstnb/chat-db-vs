from supabase_vector_store import Supabase_VectorStore
from vanna.openai import OpenAI_Chat
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Define your training data
sql_pairs = [
    {"question": "what businesses have 1065 documents uploaded with us?", "sql": "SELECT DISTINCT e.entity_name\nFROM entities e\nJOIN page_entity_crosswalk pec ON e.entity_id = pec.entity_id\nJOIN pages p ON pec.page_id = p.id\nWHERE e.entity_type = 'business'\n  AND (p.page_label = '1065_p1' OR p.page_label = '1065_k1' OR p.page_label = '1065_bal_sheet');"},
    {"question": "What is the address for RAINFLOW DEVELOPMENTS, LLC?", "sql": "SELECT STRING_AGG(ex.value, ', ') AS address\nFROM entities e\nJOIN page_entity_crosswalk pec ON e.entity_id = pec.entity_id\nJOIN pages p ON pec.page_id = p.id\nJOIN extracted2 ex ON p.preprocessed = ex.filename\nWHERE e.entity_name ILIKE '%rainflow developments, llc%'\n  AND e.entity_type = 'business'\n  AND ex.key in ('street_address','city_state')\ngroup by e.entity_name"},
    {"question": "who are the people that we have drivers licenses for?", "sql": "SELECT DISTINCT STRING_AGG(ex.value, ' ') AS full_name\nFROM extracted2 ex\nJOIN pages p ON ex.filename = p.preprocessed\nWHERE p.page_label = 'drivers_license' AND ex.key IN ('first_name', 'last_name')\nGROUP BY ex.filename\nORDER BY full_name;"},
    {"question": "what are the entity types and their counts?", "sql": "SELECT entity_type, COUNT(*) AS count\nFROM entities\nGROUP BY entity_type;"},
    {"question": "do we have any businesses with the string \"rain\" in it?", "sql": "SELECT entity_name\nFROM entities\nWHERE entity_type = 'business' AND entity_name ILIKE '%rain%';"},
    {"question": "what documents do we have for RAINFLOW DEVELOPMENTS, LLC?", "sql": "SELECT DISTINCT p.page_label\nFROM entities e\nJOIN page_entity_crosswalk pec ON e.entity_id = pec.entity_id\nJOIN pages p ON pec.page_id = p.id\nWHERE e.entity_name ILIKE '%rainflow developments, llc%';"},
    {"question": "who are the business owners of RAINFLOW DEVELOPMENTS, LLC?", "sql": "SELECT DISTINCT e.entity_name, e.entity_type, e2.entity_name AS shareholder_or_owner\nFROM entities e\nJOIN page_entity_crosswalk pec ON e.entity_id = pec.entity_id\nJOIN pages p ON pec.page_id = p.id\nJOIN extracted2 ex ON p.preprocessed = ex.filename\nJOIN entities e2 ON ex.value ILIKE '%' || e2.entity_name || '%'\nWHERE e.entity_name ILIKE '%rain%' -- Using loose match, but feel free to use exact match as well\n  AND e.entity_type = 'business'\n  AND (p.page_label = '1120S_k1' OR p.page_label = '1065_k1')\n  AND ex.key = 'shareholder_name';"}
]

all_ddl = """
CREATE TABLE IF NOT EXISTS pages (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT,
    preprocessed TEXT,
    page_number INTEGER,
    image_width DOUBLE PRECISION,
    image_height DOUBLE PRECISION,
    lines TEXT,
    words TEXT,
    bboxes TEXT,
    normalized_bboxes TEXT,
    tokens TEXT,
    words_for_clf TEXT,
    processing_time DOUBLE PRECISION,
    clf_type TEXT,
    page_label TEXT,
    page_confidence DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create the 'extracted' table
CREATE TABLE IF NOT EXISTS extracted (
    id BIGSERIAL PRIMARY KEY,
    key TEXT,
    label TEXT,
    label_bbox TEXT,
    label_confidence DOUBLE PRECISION,
    value TEXT,
    value_bbox TEXT,
    value_confidence DOUBLE PRECISION,
    page_num INTEGER,
    annotated_image_path TEXT,
    filename TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create the 'extracted2' table
CREATE TABLE IF NOT EXISTS extracted2 (
    id BIGSERIAL PRIMARY KEY,
    key TEXT,
    value TEXT,
    filename TEXT,
    page_label TEXT,
    page_confidence DOUBLE PRECISION,
    page_num INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create the 'entities' table
CREATE TABLE IF NOT EXISTS entities (
    entity_id BIGSERIAL PRIMARY KEY,
    entity_type TEXT,
    entity_name TEXT,
    additional_info TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create the 'page_entity_crosswalk' table
CREATE TABLE IF NOT EXISTS page_entity_crosswalk (
    crosswalk_id BIGSERIAL PRIMARY KEY,
    page_id BIGINT,
    entity_id BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    FOREIGN KEY (page_id) REFERENCES pages(id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);
"""

documentation_model_crosswalk = '''
This function outlines expected values in `pages.page_label` and includes definitions for each type:
def get_model(model_name: str) -> Type[BaseModel]:
    model_mapping = {
        "1120S_p1": F1120S_p1, # First page of 1120S. Contains profit & loss information.
        "1120S_k1": F1120S_k1, # K1 for 1120S. Contains shareholder information.
        "1120S_bal_sheet": BalanceSheet, # Balance sheet for 1120S.
        "1065_p1": F1065_p1, # First page of 1065. Contains profit & loss information.
        "1065_k1": F1065_k1, # K1 for 1065. Contains partner information.
        "1065_bal_sheet": BalanceSheet, # Balance sheet for 1065.
        "1120_p1": F1120_p1, # First page of 1120. Contains profit & loss information.
        "1120_bal_sheet": BalanceSheet, # Balance sheet for 1120.
        "1040_p1": F1040_p1, # First page of 1040. Contains personal information.
        "1040_sch_c": F1040_sch_c, # Schedule C for 1040. Contains business information for sole proprietors.
        "acord_25": Acord25, # Acord 25. Contains insurance information.
        "acord_28": Acord28, # Acord 28. Contains insurance information.
        "unknown": None, # Page is Unknown.
        "unknown_text_type": None, # Page is Unknown text.
        "unknown_tax_form_type": None, # Page is Unknown tax type.
        "drivers_license": DriversLicense, # Drivers license.
        "passport": Passport, # Passport.
        "lease_document": LeaseDocument, # Lease document. Should contain renter/lessee information, address, start date, end date, and term length.
        "certificate_of_good_standing": CertificateOfGoodStanding, # Certificate of good standing. Should contain business name, current standing, and date incorporated.
        "business_license": BusinessLicense, # Business license. Should contain business name, current standing, and date issued.
    }
    return model_mapping.get(model_name)
'''

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