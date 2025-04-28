from flask import Flask, jsonify, request, render_template
import boto3
import pandas as pd
import requests
import random
import time
from boto3.dynamodb.conditions import Key

app = Flask(__name__)

# Initialize DynamoDB local instance
dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url='http://localhost:8000',  # Use DynamoDB Local
    region_name='us-west-2',
    aws_access_key_id='dummy',
    aws_secret_access_key='dummy'
)
def create_new_indexed_orders_table():
    try:
        print("Attempting to create new_indexed_orders table...")  # Log start
        table = dynamodb.create_table(
            TableName='new_indexed_orders',
            KeySchema=[
                {'AttributeName': 'order_id', 'KeyType': 'HASH'},  # Partition Key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'order_id', 'AttributeType': 'S'},  # Primary key
                {'AttributeName': 'product_id', 'AttributeType': 'S'},  # GSI
                {'AttributeName': 'order_date', 'AttributeType': 'S'},  # LSI
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10,
            },
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'ProductIndex',  # GSI on product_id
                    'KeySchema': [
                        {'AttributeName': 'product_id', 'KeyType': 'HASH'},  # GSI Partition Key
                    ],
                    'Projection': {'ProjectionType': 'ALL'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 10,
                        'WriteCapacityUnits': 10,
                    }
                }
            ],
            LocalSecondaryIndexes=[
                {
                    'IndexName': 'OrderDateIndex',  # LSI on order_date
                    'KeySchema': [
                        {'AttributeName': 'order_id', 'KeyType': 'HASH'},  # Partition Key
                        {'AttributeName': 'order_date', 'KeyType': 'RANGE'},  # Sort Key for LSI
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ]
        )
        table.meta.client.get_waiter('table_exists').wait(TableName='new_indexed_orders')
        print(f"Table 'new_indexed_orders' created successfully!")  # Log success
    except Exception as e:
        print(f"Error during table creation: {e}")  # Log errors
# Function to create the Customers table with GSI based on query needs



def create_new_indexed_customers_table():
    try:
        table = dynamodb.create_table(
            TableName='new_indexed_customers',
            KeySchema=[
                {'AttributeName': 'customer_id', 'KeyType': 'HASH'},  # Partition Key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'customer_id', 'AttributeType': 'S'},  # Primary Key
                {'AttributeName': 'status', 'AttributeType': 'S'},  # GSI for filtering by status
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10,
            },
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'StatusCustomerIndex',  # GSI on status and customer_id
                    'KeySchema': [
                        {'AttributeName': 'status', 'KeyType': 'HASH'},       # GSI Partition Key
                        {'AttributeName': 'customer_id', 'KeyType': 'RANGE'}  # GSI Sort Key
                    ],
                    'Projection': {'ProjectionType': 'ALL'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 10,
                        'WriteCapacityUnits': 10,
                    }
                }
            ]
        )
        print(f"Creating table: {table.name}...")
        table.meta.client.get_waiter('table_exists').wait(TableName='new_indexed_customers')
        print(f"Table {table.name} created successfully!")
    except Exception as e:
        print(f"Error: {e}")

def create_new_indexed_products_table():
    try:
        table = dynamodb.create_table(
            TableName='new_indexed_products',
            KeySchema=[
                {'AttributeName': 'product_id', 'KeyType': 'HASH'},  # Partition Key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'product_id', 'AttributeType': 'S'},  # Primary Key
                {'AttributeName': 'price', 'AttributeType': 'N'},  # GSI for sorting products by price (Query 2)
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10,
            },
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'PriceIndex',  # GSI on price for sorting (Query 2)
                    'KeySchema': [
                        {'AttributeName': 'price', 'KeyType': 'HASH'},  # GSI Partition Key
                    ],
                    'Projection': {'ProjectionType': 'ALL'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 10,
                        'WriteCapacityUnits': 10,
                    }
                }
            ]
        )
        print(f"Creating table: {table.name}...")
        table.meta.client.get_waiter('table_exists').wait(TableName='new_indexed_products')
        print(f"Table {table.name} created successfully!")
    except Exception as e:
        print(f"Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

# Insert Customers
@app.route('/add_customer', methods=['POST'])
def add_customer():
    table = dynamodb.Table('new_indexed_customers')
    customer_data = {
        'customer_id': request.json['customer_id'],
        'customer_zip_code_prefix': request.json['customer_zip_code_prefix'],
        'customer_city': request.json['customer_city'],
        'customer_state': request.json['customer_state']
    }
    table.put_item(Item=customer_data)
    return jsonify({"message": "Customer added successfully!"}), 200

# Insert Products
@app.route('/add_product', methods=['POST'])
def add_product():
    table = dynamodb.Table('new_indexed_products')
    product_data = {
        'product_id': request.json['product_id'],
        'product_category_name': request.json['product_category_name'],
        'product_weight_g': request.json['product_weight_g'],
        'product_length_cm': request.json['product_length_cm'],
        'product_height_cm': request.json['product_height_cm'],
        'product_width_cm': request.json['product_width_cm']
    }
    table.put_item(Item=product_data)
    return jsonify({"message": "Product added successfully!"}), 200

# Insert Orders
@app.route('/add_order', methods=['POST'])
def add_order():
    table = dynamodb.Table('new_indexed_orders')
    order_data = {
        'order_id': request.json['order_id'],
        'product_id': request.json['product_id'],
        'customer_id': request.json['customer_id'],
        'order_date': request.json['order_date'],
        'quantity': request.json['quantity'],
        'status': request.json['status'],
        'total_price': request.json['total_price']
    }
    table.put_item(Item=order_data)
    return jsonify({"message": "Order added successfully!"}), 200

# Log query execution time in a file
log_file = "execution.txt"
def log_query_time(query_name, execution_time):
    with open(log_file, "a") as f:
        f.write(f"{query_name}: {execution_time:.4f} seconds\n")

# 1. Query Orders by Product and Date using LSI (OrderDateIndex)
@app.route('/orders_by_product_and_date', methods=['GET'])
def query_orders_by_product_and_date():
    table = dynamodb.Table('new_indexed_orders')
    product_id = 'mHI4NBBHLa3q'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    start_time = time.time()
    
    # Use LSI 'OrderDateIndex' with product_id and order_date (RANGE)
    response = table.query(
        IndexName='OrderDateIndex',  # Use LSI on order_date
        KeyConditionExpression=Key('order_id').eq(product_id) & 
                             Key('order_date').between(start_date, end_date)
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    log_query_time("Query Orders by Product and Date", execution_time)
    
    return jsonify(response['Items']), 200

# 2. Sort Products by Price using GSI (PriceIndex)
@app.route('/sort_products_by_price', methods=['GET'])
def sort_products_by_price():
    table = dynamodb.Table('new_indexed_products')
    
    start_time = time.time()
    response = table.scan(IndexName='PriceIndex', ProjectionExpression="product_id, product_category_name, price")
    sorted_items = sorted(response['Items'], key=lambda item: float(item['price']))
    
    end_time = time.time()
    execution_time = end_time - start_time
    log_query_time("Sort Products by Price", execution_time)
    
    return jsonify(sorted_items), 200

# 3. Filter Orders by Status and Customer using GSI (StatusCustomerIndex)
@app.route('/filter_orders_by_status_and_customer', methods=['GET'])
def filter_orders_by_status_and_customer():
    table = dynamodb.Table('new_indexed_customers')
    status = 'delivered'
    customer_id = 'CUST002'
    
    start_time = time.time()
    response = table.query(
        IndexName='StatusCustomerIndex',  # Use GSI on status and customer_id
        KeyConditionExpression=Key('status').eq(status) & Key('customer_id').eq(customer_id)
    )
    end_time = time.time()
    execution_time = end_time - start_time
    log_query_time("Filter Orders by Status and Customer", execution_time)
    
    return jsonify(response['Items']), 200

# 4. Query Orders for a Specific Customer and Sort by Order Date using GSI (CustomerOrderDateIndex)
@app.route('/orders_by_customer', methods=['GET'])
def query_orders_by_customer():
    table = dynamodb.Table('new_indexed_orders')
    customer_id = 'CUST001'
    
    start_time = time.time()
    response = table.query(
        IndexName='OrderDateIndex',  # Use LSI on order_date
        KeyConditionExpression=Key('customer_id').eq(customer_id)
    )
    sorted_items = sorted(response['Items'], key=lambda item: item['order_date'])
    
    end_time = time.time()
    execution_time = end_time - start_time
    log_query_time("Query Orders by Customer and Sort by Order Date", execution_time)
    
    return jsonify(sorted_items), 200

# Insert Customers from CSV
def insert_customers():
    df_customers = pd.read_csv(r'G:\7Th-Semester\new_way\Dynamo-Db\Dynamo lab\Data\preprocessed_customers.csv')

    for index, row in df_customers.iterrows():
        try:
            customer_data = {
                'customer_id': row['customer_id'],
                'customer_zip_code_prefix': str(row['customer_zip_code_prefix']),
                'customer_city': str(row['customer_city']),
                'customer_state': str(row['customer_state'])
            }
            response = requests.post('http://localhost:8000/add_customer', json=customer_data)
            print(f"Inserted customer {row['customer_id']}, Response: {response.json()}")
        except Exception as e:
            print(f"An error occurred at row {index}: {e}")

# Insert Products from CSV
def insert_products():
    df_products = pd.read_csv(r'G:\7Th-Semester\new_way\Dynamo-Db\Dynamo lab\Data\preprocessed_products.csv')

    for index, row in df_products.iterrows():
        try:
            product_data = {
                'product_id': row['product_id'],
                'product_category_name': row['product_category_name'],
                'product_weight_g': row['product_weight_g'],
                'product_length_cm': row['product_length_cm'],
                'product_height_cm': row['product_height_cm'],
                'product_width_cm': row['product_width_cm']
            }
            response = requests.post('http://localhost:8000/add_product', json=product_data)
            print(f"Inserted product {row['product_id']}, Response: {response.json()}")
        except Exception as e:
            print(f"An error occurred at row {index}: {e}")

# Insert Orders from CSV
def insert_orders():
    df_orderitems = pd.read_csv(r'G:\7Th-Semester\new_way\Dynamo-Db\Dynamo lab\Data\preprocessed_OrderItems.csv')
    df_orders = pd.read_csv(r'G:\7Th-Semester\new_way\Dynamo-Db\Dynamo lab\Data\preprocessed_orders.csv')

    for index, row in df_orderitems.iterrows():
        try:
            quantity = random.randint(1, 5)
            total_price = (row['price'] * quantity) + row['shipping_charges']
            product_id = str(row['product_id'])
            order_id = str(df_orders.iloc[index]['order_id'])
            customer_id = str(df_orders.iloc[index]['customer_id'])
            order_purchase_timestamp = str(df_orders.iloc[index]['order_purchase_timestamp'])

            order_data = {
                'order_id': order_id,
                'product_id': product_id,
                'customer_id': customer_id,
                'order_date': order_purchase_timestamp,
                'quantity': quantity,
                'status': 'pending',  # Assuming default status
                'total_price': total_price
            }

            response = requests.post('http://localhost:8000/add_order', json=order_data)
            print(f"Inserted order {order_id} with product {product_id}, Response: {response.json()}")
        except Exception as e:
            print(f"An error occurred at row {index}: {e}")

# Call the functions to create the tables
#create_new_indexed_orders_table()
#create_new_indexed_customers_table()
#create_new_indexed_products_table()

#insert_customers()  # Insert data into the new_indexed_customers table
#insert_products()   # Insert data into the new_indexed_products table
#insert_orders()     # Insert data into the new_indexed_orders table


query_orders_by_product_and_date()  # Query 1: Orders by Product and Date
sort_products_by_price()            # Query 2: Sort Products by Price
filter_orders_by_status_and_customer()  # Query 3: Filter Orders by Status and Customer
query_orders_by_customer()          # Query 4: Orders by Customer and Sort by Date

if __name__ == '__main__':
    app.run(debug=True)
