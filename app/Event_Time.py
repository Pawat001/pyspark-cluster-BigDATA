from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# 1. สร้าง SparkSession
spark = SparkSession.builder \
    .appName("EventTimeWindowing") \
    .getOrCreate()

# 2. อ่านข้อมูลจาก Socket
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# 3. แยกคำ (Split/Explode) และเพิ่มเวลาปัจจุบัน (Timestamp)
words = lines.select(
    explode(split(lines.value, " ")).alias("word")
).withColumn("timestamp", current_timestamp())

# 4. จัดกลุ่มด้วย Sliding Window (10 วินาที เลื่อนทุก 5 วินาที) และนับคำ
windowCounts = words.groupBy(
    window(words.timestamp, "10 seconds", "5 seconds"),
    words.word
).count()

# 5. แสดงผลออกทางหน้าจอและสั่งเริ่มทำงาน
query = windowCounts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()