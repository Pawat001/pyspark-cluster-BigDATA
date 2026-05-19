from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
# 1. เปิดสิทธิ์การใช้งาน Spark
spark = SparkSession.builder.appName("AssociationRules").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. โหลดไฟล์ข้อมูลดิบ (นี่คือ Dataframe ที่ 1 ของคุณ)
df = spark.read.csv("groceries_data.csv", header=True, inferSchema=True)
print("=== Dataframe 1: Data groceries ===")
df.show(20)

# 3. จัดกลุ่มสินค้าเข้าตะกร้าแยกตามสมาชิก (นี่คือ Dataframe ที่ 2 ของคุณ)
basket_df = df.groupBy("Member_number").agg(collect_set("itemDescription").alias("basket")) [cite: 288, 292]
print("=== Dataframe 2: ข้อมูลที่รวมเป็นตะกร้า (basket) เรียบร้อยแล้ว ===")
basket_df.show(20)

# 4. ตั้งค่าโมเดล FPGrowth (กำหนดค่าขั้นต่ำในการยอมรับความสัมพันธ์)
# minSupport=0.01 (กลุ่มสินค้าต้องพบอย่างน้อย 1% ของทั้งหมด)
# minConfidence=0.1 (ความน่าเชื่อถือของกฎขั้นต่ำ 10%)
fp = FPGrowth(minSupport=0.01, minConfidence=0.1, itemsCol='basket', predictionCol='prediction') [cite: 300, 301]

# 5. เทรนโมเดล (Fit Model)
model = fp.fit(basket_df) 

# 6. แสดงกลุ่มสินค้าที่พบบ่อย (นี่คือ Dataframe ที่ 3 ของคุณ)
print("=== Dataframe 3: Frequent Itemsets (กลุ่มสินค้าที่พบบ่อย) ===")
model.freqItemsets.show(5, truncate=False)

# 7. แสดงกฎความสัมพันธ์ที่ได้ (นี่คือ Dataframe ที่ 4 ของคุณ)
print("=== Dataframe 4: Association Rules (กฎความสัมพันธ์พร้อมค่าสถิติ) ===")
model.associationRules.show(5, truncate=False) 

# 8. สร้างข้อมูลตะกร้าใหม่ตามที่โจทย์กำหนด (ปรับแก้ชื่อสินค้าให้ตรงกับในไฟล์ CSV)
test_data = [
    Row(basket=['fruit/vegetable juice', 'frozen fruits', 'packaged fruit/vegetables']),
    Row(basket=['mayonnaise', 'butter', 'rolls/buns'])
]
new_df = spark.createDataFrame(test_data)
print("=== ข้อมูลตะกร้าใหม่ที่รอการทำนาย ===")
new_df.show(truncate=False)

# 9. ทำนายผลสินค้าที่จะซื้อเพิ่ม ( Dataframe ที่ 5 )
predictions = model.transform(new_df)
print("=== Dataframe 5: ผลการทำนายสินค้าที่แนะนำ (Prediction) ===")
predictions.show(truncate=False)

# 10. ปิด SparkSession เมื่อทำงานเสร็จสิ้น
spark.stop()

