import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = os.environ.get("INFLUXDB_TOKEN")
org = "vibstring"
url = "http://192.168.1.90:8086"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

# 创建bucket（实验关联）
# buckets_api = client.buckets_api() # 创建链接bucket的客户端
# created_bucket = buckets_api.create_bucket(bucket_name="miao", org=org)


bucket="miao"
write_api = client.write_api(write_options=SYNCHRONOUS)
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org=org, record=point)
  time.sleep(1) # separate points by 1 second

# query_api = client.query_api()

# query = """from(bucket: "miao")
#  |> range(start: -60m)
#  |> filter(fn: (r) => r._measurement == "measurement1")"""
# tables = query_api.query(query, org="vibstring")

# for table in tables:
#   for record in table.records:
#     print(record)