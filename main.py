import psycopg2
from sqlalchemy import create_engine, inspect
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# PostgreSQL connection details
db_config = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'rosterly_first'
}

# Create a connection to the PostgreSQL database
engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
]
schema = CollectionSchema(fields, "A collection for company data")

# Check if the collection exists and drop it if it does
collection_name = "company_data"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Create a new collection
collection = Collection(collection_name, schema)

# Function to process and insert data from a table into Milvus
def process_table(table_name):
    try:
        # Query to retrieve column names and data
        query = f"SELECT * FROM test.{table_name}"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print(f"No data found in table: {table_name}")
            return
        
        # Retrieve primary key column name
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
        
        # Combine relevant fields into a single string for embedding
        df['combined_info'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        
        # Generate embeddings for the combined_info column
        df['embedding'] = df['combined_info'].apply(lambda x: model.encode(x).tolist())  # Convert numpy array to list
        
        # Prepare data for insertion
        ids = df[pk_column].tolist()
        table_names = [table_name] * len(ids)
        embeddings = df['embedding'].tolist()
        
        # Print debug information
        print(f"Table: {table_name}")
        print(f"IDs: {ids}")
        print(f"Table Names: {table_names}")
        print(f"Embedding Shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")
        print(f"Sample Embeddings: {embeddings[:2]}")  # Print only first two embeddings for brevity
        
        if not ids or not table_names or not embeddings:
            print(f"Skipping insertion for table: {table_name} due to empty data.")
            return
        
        # Insert data into the collection
        collection.insert([ids, table_names, embeddings])
    except Exception as e:
        print(f"Error processing table {table_name}: {e}")
        return

# Get list of tables in the 'test' schema
inspector = inspect(engine)
tables = inspector.get_table_names(schema='test')

# Process each table
for table in tables:
    process_table(table)
