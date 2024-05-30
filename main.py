import psycopg2
from sqlalchemy import create_engine, inspect
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

db_config = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'rosterly_first'
}

engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
]
schema = CollectionSchema(fields, "A collection for company data")

collection_name = "company_data"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(collection_name, schema)

def process_table(table_name):
    try:
        query = f"SELECT * FROM test.{table_name}"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print(f"No data found in table: {table_name}")
            return
        
        pk_query = f"""
        SELECT
            kcu.column_name
        FROM
            information_schema.table_constraints tco
            JOIN information_schema.key_column_usage kcu 
              ON kcu.constraint_name = tco.constraint_name
              AND kcu.constraint_schema = tco.constraint_schema
              AND kcu.constraint_catalog = tco.constraint_catalog
        WHERE tco.constraint_type = 'PRIMARY KEY' AND kcu.table_name = '{table_name}';
        """
        pk_column = pd.read_sql(pk_query, engine).iloc[0, 0]
        
        df['combined_info'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        df['embedding'] = df['combined_info'].apply(lambda x: model.encode(x).tolist())
        
        ids = df[pk_column].tolist()
        table_names = [table_name] * len(ids)
        embeddings = df['embedding'].tolist()
        
        if not ids or not table_names or not embeddings:
            print(f"Skipping insertion for table: {table_name} due to empty data.")
            return
        
        collection.insert([ids, table_names, embeddings])
    except Exception as e:
        print(f"Error processing table {table_name}: {e}")
        return

inspector = inspect(engine)
tables = inspector.get_table_names(schema='test')

for table in tables:
    process_table(table)

