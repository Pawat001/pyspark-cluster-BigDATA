from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import count, countDistinct, first, last, min, max, sum, sumDistinct, avg

# 1. Initialize Spark Session
# ใช้ Builder pattern ในการสร้าง Session
spark = SparkSession.builder \
    .appName("CleanOutput") \
    .getOrCreate()

# 2. Configure Logging
# ลดความวุ่นวายของ Log ให้เหลือแค่ Error เท่านั้น
spark.sparkContext.setLogLevel("ERROR")

read_file = spark.read.format("csv").option("header", True).load("fb_live_thailand.csv")  # Read file
read_file.printSchema()  # Print schema

# --- โค้ดสไลด์ที่ 10 ---
# เลือกเฉพาะคอลัมน์ num_reactions และ num_comments
sqlDF = read_file.select(read_file.num_reactions, read_file.num_comments)
sqlDF.show(5)  # แสดงผลลัพธ์ 5 แถวแรก

# --- โค้ดสไลด์ที่ 11 ---
# เลือกคอลัมน์ status_id และ num_reactions
sqlDF = read_file.select(read_file.status_id, read_file.num_reactions) \
    .where(read_file.num_reactions.cast(IntegerType()) > 3000) \
    .withColumnRenamed("num_reactions", "Reactions") \
    .orderBy(read_file.num_reactions)

# --- โค้ดสไลด์ที่ 12---
sqlDF.show(3)  # แสดงผลลัพธ์ 3 แถว ตามตารางในสไลด์

# สร้างตารางจำลอง (Temporary View)
read_file.createOrReplaceTempView("FB_Thailand_data")

# ใช้คำสั่ง SQL ดึงข้อมูลจากตารางจำลองที่เราสร้างไว้
sqlDF = spark.sql("select * from FB_Thailand_data")

# แสดงผลลัพธ์ 3 แถวแรก
sqlDF.show(3)

# --- โค้ดสไลด์ที่ 13 ---
# บันทึกข้อมูลแบบเขียนทับ (overwrite) ลงในโฟลเดอร์ชื่อ output_data
sqlDF.write.mode("overwrite").csv("output_data")

# --- โค้ดสไลด์ที่ 14 ---
# สุ่มแบ่งข้อมูลเป็น 2 ชุด (สัดส่วน 80% และ 20%)
split = sqlDF.randomSplit([0.8, 0.2])

# แสดงผลลัพธ์ข้อมูลชุดแรก (Index 0)
print("--- Data Set 1 (60%) ---")
split[0].show(3)

# แสดงผลลัพธ์ข้อมูลชุดที่สอง (Index 1)
print("--- Data Set 2 (40%) ---")
split[1].show(3)

# --- โค้ดสไลด์ที่ 17---
sqlDF.select(count("status_published")).show()

# --- โค้ดสไลด์ที่ 18 ---
sqlDF.select(countDistinct("status_type")).show()

# --- โค้ดสไลด์ที่ 19 (first, last) ---
sqlDF.select(first("status_published"), last("status_published")).show()

# --- โค้ดสไลด์ที่ 20 (min, max, sum, sumDistinct, avg) ---
sqlDF.select(min("num_reactions"), max("num_reactions")).show()

# --- โค้ดสไลด์ที่ 21 (count, countDistinct, avg) ---
# หาผลรวมทั้งหมด
sqlDF.select(sum("num_reactions")).show()
# หาผลรวมแบบไม่ซ้ำ
sqlDF.select(sumDistinct("num_reactions")).show()
# หาค่าเฉลี่ย
sqlDF.select(avg("num_reactions")).show()

# --- ส่วนการ Join ข้อมูล (สไลด์ 24-30) ---
left_df = read_file.select("status_id", "status_type", "status_published", "num_reactions", "num_comments")
right_df = read_file.select("status_id", "status_type", "status_published", "num_reactions", "num_shares")

join_condition = left_df["status_id"] == right_df["status_id"]

def show_join_result(title, join_type, n=10):
    print("\n" + "=" * 40)
    print(f"{title}")
    print("=" * 40)
    left_df.join(right_df, join_condition, join_type).show(n, truncate=False)

# แสดงผลตามลำดับ: inner, left, right
show_join_result("Inner Join Example Result", "inner")
show_join_result("Left Outer Join Example Result", "left")
show_join_result("Right Outer Join Example Result", "right")
# ...existing code...