from pyspark.sql import SparkSession

# 1. Initialize Spark Session
# ใช้ Builder pattern ในการสร้าง Session
spark = SparkSession.builder \
    .appName("CleanOutput") \
    .getOrCreate()

# 2. Configure Logging
# ลดความวุ่นวายของ Log ให้เหลือแค่ Error เท่านั้น
spark.sparkContext.setLogLevel("ERROR")
read_file = spark.read.format\
("csv").option('header',True).\
load('fb_live_thailand.csv') # Read file
read_file.printSchema() # Print schema