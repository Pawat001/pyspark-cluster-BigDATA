from pyspark.sql import SparkSession
from operator import add
import tempfile
import shutil


spark = SparkSession.builder.appName("LowLevelSparkAssignment").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

print("--- 0. RDD Creation (Slide 14) ---")
# สร้าง RDD จาก List
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rdd_alpha = sc.parallelize(alphabet, 4)
print("Number of partitions (alpha):", rdd_alpha.getNumPartitions())

# สร้าง RDD จาก Text File (ใช้ไฟล์ที่คุณมี)
rdd_csv = sc.textFile("fb_live_thailand.csv", 5)
print("Number of partitions (csv):", rdd_csv.getNumPartitions())

# สร้าง RDD แบบ Key-Value Pairs สำหรับใช้ทดสอบคำสั่งอื่นๆ
kv_data = [('a',1), ('b',2), ('c',3), ('a', 1), ('b',2)]
rdd_kv = sc.parallelize(kv_data, 4)

# 1. Transformations 

print("\n--- 1. Transformations ---")

#  distinct() และ count()
count_distinct = rdd_csv.distinct().count()
print("Number of distinct records in CSV: ", count_distinct)

#  filter()

filter_rdd = rdd_csv.filter(lambda x: len(x.split(',')) > 1 and x.split(',')[1] == 'link').take(5) 
print("Filter (take 5):", filter_rdd)

#  flatMap() และ map()
flatmap_rdd = rdd_csv.flatMap(lambda x: x.split(','))
pair = flatmap_rdd.map(lambda x: (x, 1))
print("flatMap & map (take 5):", pair.take(5))

#  sortByKey()
# (ใช้ rdd_kv เพื่อความรวดเร็วในการแสดงผล)
sort_data_key = rdd_kv.sortByKey().collect()
print("sortByKey:", sort_data_key)

#  sortBy()
sort_data = rdd_kv.sortBy(lambda x: x, False, 5).collect()
print("sortBy (descending):", sort_data)

#  reduceByKey()
reduce_key = rdd_kv.reduceByKey(lambda x, y: x + y)
print("reduceByKey:", reduce_key.collect())

#  aggregateByKey()
zero_val = (0, 0)
par_agg = lambda x, y: (x[0] + y, x[1] + 1)
allpar_agg = lambda x, y: (x[0] + y[0], x[1] + y[1])
agg = rdd_kv.aggregateByKey(zero_val, par_agg, allpar_agg).collect()
print("aggregateByKey:", agg)

#  foldByKey()
fold = sorted(rdd_kv.foldByKey(0, add).collect())
print("foldByKey:", fold)

#  combineByKey()
def tolist(x): return [x]
def append_val(x, y): 
    x.append(y)
    return x
def extend_val(x, y): 
    x.extend(y)
    return x
combine = sorted(rdd_kv.combineByKey(tolist, append_val, extend_val).collect())
print("combineByKey:", combine)

#  groupByKey()
group1 = sorted(rdd_kv.groupByKey().mapValues(len).collect())
print("groupByKey (len):", group1)
group2 = sorted(rdd_kv.groupByKey().mapValues(list).collect())
print("groupByKey (list):", group2)

#  join(), leftOuterJoin(), rightOuterJoin()
rdd1 = sc.parallelize([('a',1), ('b',2), ('c',3)])
rdd2 = sc.parallelize([('a',1), ('b',2), ('a',1), ('b',2)])

print("join:", rdd1.join(rdd2).collect())
print("leftOuterJoin:", rdd1.leftOuterJoin(rdd2).collect())
print("rightOuterJoin:", rdd1.rightOuterJoin(rdd2).collect())

# 2. Actions 

print("\n--- 2. Actions ---")

#  countByKey()
print("countByKey:", dict(rdd_kv.countByKey()))

#  countByValue()
print("countByValue:", dict(rdd_kv.countByValue()))

#  collectAsMap()
print("collectAsMap:", rdd_kv.collectAsMap())

#  lookup()
print("lookup('a'):", rdd_kv.lookup('a'))

#  first()
print("first:", rdd_kv.first())

#  max() และ min()
print("max:", rdd_kv.max())
print("min:", rdd_kv.min())

# 3. Save Output (Slide 32)

print("\n--- 3. Save to File ---")
#  saveAsTextFile()
folder = "textfile_output"
import os
if os.path.exists(folder):
    shutil.rmtree(folder)

rdd_kv.saveAsTextFile(folder)
print(f"Data successfully saved to folder: {folder}")