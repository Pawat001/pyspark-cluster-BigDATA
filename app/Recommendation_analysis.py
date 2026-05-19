# ==========================================
# ส่วนที่ 1: นำเข้า Libraries และตั้งค่าเริ่มต้น (สไลด์หน้า 13)
# ==========================================
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator   # ใช้ประเมินผล RMSE 
from pyspark.ml.recommendation import ALS               # ไลบรารีระบบแนะนำแบบ Collaborative Filtering 
from pyspark.sql.functions import desc, col

# สร้างเซสชัน Spark 
spark = SparkSession.builder.appName("BookRecommendation").getOrCreate()

# ==========================================
# ส่วนที่ 2: โหลดข้อมูลและแบ่งชุดข้อมูล (สไลด์หน้า 12)
# ==========================================
# โหลดไฟล์ข้อมูลจาก book_ratings.csv 
df = spark.read.csv("book_ratings.csv", header=True, inferSchema=True)

# สุ่มแบ่งข้อมูลเป็น 2 ส่วน: สำหรับสอนโมเดล (Training) 80% และสำหรับทดสอบ (Test) 20%
(training, test) = df.randomSplit([0.8, 0.2], seed=42)

# ==========================================
# ส่วนที่ 3: สร้างและสอนโมเดล ALS (สไลด์หน้า 14)
# ==========================================
# กำหนดค่าพารามิเตอร์ของ ALS ตามโครงสร้างข้อมูลจริงในไฟล์ csv 
# ใช้ coldStartStrategy="drop" เพื่อลบแถวที่เป็น NaN ป้องกันโมเดลประเมินผลไม่ได้ 
als = ALS(
    maxIter=10, 
    userCol="user_id",       # กำหนดคอลัมน์ผู้ใช้ 
    itemCol="book_id",       # กำหนดคอลัมน์หนังสือ 
    ratingCol="rating",      # กำหนดคอลัมน์เรตติ้ง 
    coldStartStrategy="drop" # จัดการปัญหา Cold Start 
)

# สอนโมเดล (Fit Model) ด้วยข้อมูลฝั่ง Training
model = als.fit(training)

# ทำนายผลเรตติ้ง (Transform) โดยใช้ข้อมูลฝั่ง Test
predictions = model.transform(test)

# ==========================================
# ส่วนที่ 4: การประเมินผลและการแสดงผลลัพธ์เพื่อส่งงาน (สไลด์หน้า 15-20)
# ==========================================

# ----- 🎯 ผลลัพธ์ที่ 1: แสดงค่า RMSE (สไลด์หน้า 15) -----
# ตั้งค่าตัวประเมินผลโดยใช้สูตร RMSE 
evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="rating", 
    predictionCol="prediction"
)
# คำนวณค่า RMSE ออกมา 
rmse = evaluator.evaluate(predictions)
print(f"\n=========================================")
print(f"🎯 [RESULT] Root Mean Square Error (RMSE) = {rmse}")
print(f"=========================================\n")


# ----- 📊 DataFrame ชุดที่ 1: แสดงการทำนายผลของ User ID = 53 (สไลด์หน้า 16-17) -----
print("--- 📊 DataFrame 1: Prediction for User ID = 53 ---")
# 1. กรองข้อมูลจริงในชุด Test เฉพาะอันที่เป็นของ User ID 53 มาตรวจสอบ 
user_53_actual = test.filter(col("user_id") == 53)
print("Actual Data for User 53:")
user_53_actual.show() # แสดงผลข้อมูลจริงก่อนทำนาย

# 2. นำข้อมูลของ User ID 53 ไปให้โมเดลทำนายเรตติ้ง (Transform) และเรียงลำดับจากคะแนนทำนายมากไปน้อย 
user_53_predict = model.transform(user_53_data if 'user_53_data' in locals() else user_53_actual)
print("Predicted Ratings for User 53:")
# เลือกแสดงคอลัมน์ book_id, user_id, rating และ prediction ตามโจทย์สไลด์หน้า 12 
user_53_predict.select("book_id", "user_id", "rating", "prediction").orderBy(desc("prediction")).show()


# ----- 📊 DataFrame ชุดที่ 2: แนะนำหนังสือ 5 เล่มให้ผู้ใช้ทุกคน (สไลด์หน้า 18) -----
print("--- 📊 DataFrame 2: 5 Recommended Books for All Users ---")
# สร้างรายการแนะนำหนังสือ 5 อันดับแรกให้ผู้ใช้ทุกคน และตั้งค่า truncate=False เพื่อให้แสดงผลเต็มคอลัมน์
userRecs = model.recommendForAllUsers(5)
userRecs.show(truncate=False)


# ----- 📊 DataFrame ชุดที่ 3: แนะนำผู้ใช้ 5 คนให้หนังสือทุกเล่ม (สไลด์หน้า 19) -----
print("--- 📊 DataFrame 3: 5 Recommended Users for All Books ---")
# สร้างรายการแนะนำผู้ใช้ 5 อันดับแรกที่มีแนวโน้มจะชอบหนังสือแต่ละเล่ม 
bookRecs = model.recommendForAllItems(5)
bookRecs.show(truncate=False)