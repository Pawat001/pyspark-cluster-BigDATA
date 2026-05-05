# ---------------------------------------------------------
# PART 1: Spark Setup & Data Creation
# ---------------------------------------------------------
from pyspark.sql import SparkSession
from operator import add
import os

# สร้าง SparkSession 
spark = SparkSession.builder.master("local[*]").appName("LowLevelSpark").getOrCreate()

alphabet_list = [('a', 1), ('b', 2), ('c', 3), ('a', 1), ('b', 2)]
rdd = spark.sparkContext.parallelize(alphabet_list, 4) 


base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "fb_live_thailand.csv")
rdd_file = spark.sparkContext.textFile(csv_path, 5)

# ---------------------------------------------------------
# PART 2: Transformation Functions (Define New RDDs)
# ---------------------------------------------------------

# Narrow Transformations 
distinct_rdd = rdd.distinct()
filter_rdd = rdd.filter(lambda x: x[0] == 'a') 
map_rdd = rdd.map(lambda x: (x[0], x[1] * 10)) 

reduce_rdd = rdd.reduceByKey(lambda x, y: x + y) 
sorted_key = rdd.sortByKey() 
sorted_val = rdd.sortBy(lambda x: x[1], False, 5) 

# Advanced Key-Value Transformations
# aggregateByKey: รวมค่าในระดับ Partition ก่อนแล้วจึงรวมข้าม 
zero_val = (0, 0)
par_agg = lambda x, y: (x[0] + y, x[1] + 1)
allpar_agg = lambda x, y: (x[0] + y[0], x[1] + y[1])
agg_rdd = rdd.aggregateByKey(zero_val, par_agg, allpar_agg)

# foldByKey: คล้าย aggregate แต่ใช้ค่าเริ่มต้นที่เหมือนกันทั้งตอนเริ่มและตอนรวม 
fold_rdd = rdd.foldByKey(0, add)

# combineByKey: แปลง Key-Value ให้เป็นชุดข้อมูลที่ซับซ้อนขึ้น 
combine_rdd = rdd.combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)

# groupByKey: จัดกลุ่มค่าตาม Key 
group_rdd = rdd.groupByKey().mapValues(list)

# Join Operations 
rdd_extra = spark.sparkContext.parallelize([('a', 100), ('z', 999)])
join_rdd = rdd.join(rdd_extra)

# ---------------------------------------------------------
# PART 3: Action Functions (Trigger Execution & Show Results)
# ---------------------------------------------------------

print("--- RESULTS OF ACTION FUNCTIONS ---")

# แสดงผลลัพธ์พื้นฐาน
print("1. Collect:", rdd.collect()) 
print("2. Count:", rdd.count()) 
print("3. First:", rdd.first()) 
print("4. Max/Min:", rdd.max(), "/", rdd.min()) 

# แสดงผลลัพธ์ฟังก์ชัน Dictionary/Key-based
print("5. CountByKey:", rdd.countByKey())
print("6. CountByValue:", rdd.countByValue()) 
print("7. CollectAsMap:", rdd.collectAsMap()) 
print("8. Lookup 'a':", rdd.lookup('a'))

# แสดงตัวอย่างข้อมูลจากไฟล์ (เพื่อยืนยันว่าอ่านไฟล์สำเร็จ)
print("9. Sample lines from CSV:", rdd_file.take(3))

print("--- FINISHED ---")
spark.stop()