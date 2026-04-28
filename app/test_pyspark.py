from pyspark.sql import SparkSession

# 1. Initialize Spark Session
# ใช้ Builder pattern ในการสร้าง Session
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("CleanOutput") \
    .getOrCreate()

# 2. Configure Logging
# ลดความวุ่นวายของ Log ให้เหลือแค่ Error เท่านั้น
spark.sparkContext.setLogLevel("ERROR")

# 3. Create Sample Data
data = [
    ("A", 1), 
    ("B", 2)
]
columns = ["Name", "Value"]

# 4. Create and Display DataFrame
df = spark.createDataFrame(data, schema=columns)
df.show()

# 5. Stop the Session
spark.stop()