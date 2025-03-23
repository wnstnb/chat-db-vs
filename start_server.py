from vanna.openai import OpenAI_Chat
from supabase_vector_store import Supabase_VectorStore
from vanna.base import VannaBase
from dotenv import load_dotenv
import os
import pandas as pd
from vanna.utils import deterministic_uuid
from vanna.flask import VannaFlaskApp

load_dotenv()

class MyVanna(Supabase_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Supabase_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

if __name__ == '__main__':

    vn = MyVanna(config={'api_key': os.getenv('OPENAI_API_KEY'), 'model': 'gpt-4o'})

    vn.connect_to_postgres(
        host=os.getenv('SUPABASE_HOST'),
        dbname=os.getenv('SUPABASE_DBNAME'),
        user=os.getenv('SUPABASE_USER'),
        password=os.getenv('SUPABASE_PASSWORD'),
        port=os.getenv('SUPABASE_PORT')
    )

    app = VannaFlaskApp(vn)
    app.run()