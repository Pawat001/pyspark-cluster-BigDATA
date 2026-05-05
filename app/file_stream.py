from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.types import StructType, StringType, StructField, IntegerType


spark = SparkSession.builder.appName("FileStreaming").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. กำหนด Schema ให้ตรงกับคอลัมน์ในไฟล์ CSV จริง
user_schema = StructType([
    StructField("status_id", StringType(), True),
    StructField("status_type", StringType(), True),
    StructField("status_published", StringType(), True),
    StructField("num_reactions", IntegerType(), True),
    StructField("num_comments", IntegerType(), True),
    StructField("num_shares", IntegerType(), True),
    StructField("num_likes", IntegerType(), True),
    StructField("num_loves", IntegerType(), True),
    StructField("num_wows", IntegerType(), True),
    StructField("num_hahas", IntegerType(), True),
    StructField("num_sads", IntegerType(), True),
    StructField("num_angrys", IntegerType(), True)
])

#อ่านข้อมูลจากโฟลเดอร์แบบ Stream
lines = spark.readStream.format("csv") \
    .option("maxFilesPerTrigger", 1) \
    .option("header", True) \
    .option("path", "/app/fb_data") \
    .schema(user_schema).load()

date_df = lines.withColumn("date", split(lines["status_published"], " ").getItem(0))
date_counts = date_df.groupBy("date").count()

query = date_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()