# ====================================================================
# STEP 1: นำเข้าเครื่องมือที่จำเป็นทั้งหมด (Import Libraries)
# ====================================================================
# SparkSession: ตัวจัดการหลักในการเปิดใช้งานระบบ Apache Spark
from pyspark.sql import SparkSession

# IntegerType: ใช้สำหรับแปลงประเภทข้อมูลของคะแนนรีวิวให้เป็นตัวเลขจำนวนเต็ม
from pyspark.sql.types import IntegerType

# นำเข้าฟังก์ชันจัดการตารางทั่วไป (ในที่นี้เราจะเรียกใช้ฟังก์ชัน trim)
from pyspark.sql.functions import *

# Tokenizer: ตัวตัดคำ, StopWordsRemover: ตัวลบคำฟุ่มเฟือย, HashingTF: ตัวแปลงคำเป็นตัวเลขเชิงสถิติ
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF

# Pipeline: ตัวเชื่อมโยงขั้นตอนการเตรียมข้อมูลทั้งหมดให้ทำงานร่วมกันเป็นขั้นตอนเดียว
from pyspark.ml import Pipeline

# LogisticRegression: ตัวแบบโมเดลที่จะใช้ในการเรียนรู้และพยากรณ์ข้อมูล
from pyspark.ml.classification import LogisticRegression

# MulticlassClassificationEvaluator: เครื่องมือวัดประสิทธิภาพและความแม่นยำของโมเดล
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ====================================================================
# STEP 2: เริ่มต้นระบบและโหลดข้อมูล (Initialize Spark & Load Data)
# ====================================================================
# สร้างเซสชันเริ่มต้นสำหรับการรันโปรแกรม
spark = SparkSession.builder.appName("TextAnalyticsAssignment").getOrCreate()

# ตั้งค่า Log ให้แสดงเฉพาะคำเตือน (WARN) ผลลัพธ์หน้าจอจะได้สะอาดและดูง่าย
spark.sparkContext.setLogLevel("WARN")

# อ่านข้อมูลรีวิวจากไฟล์ reviews_rated.csv
# header=True (ใช้แถวแรกเป็นหัวตาราง), inferSchema=True (ให้ระบบเดาประเภทข้อมูลพื้นฐานอัตโนมัติ)
raw_df = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)


# ====================================================================
# STEP 3: ทำความสะอาดและแปลงประเภทข้อมูล (Data Preprocessing)
# ====================================================================
# คัดเลือกเฉพาะคอลัมน์คำรีวิวและคะแนน พร้อมทั้งเคลียร์ช่องว่างและแปลงชนิดข้อมูลตามสั่ง
# 1. trim(col("Review Text")) -> ลบช่องว่างส่วนเกินที่อยู่หน้าและท้ายข้อความออก
# 2. col("Rating").cast(IntegerType()) -> แปลงคะแนนรีวิวจากตัวหนังสือ/ทศนิยม ให้กลายเป็นเลขจำนวนเต็ม
cleaned_df = raw_df.select(
    trim(col("Review Text")).alias("review_text"),
    col("Rating").cast(IntegerType()).alias("label")  # ตั้งชื่อว่า label เพื่อให้ตัวแบบเรียนรู้ได้ง่าย
)

# กันข้อมูลว่าง/ค่า null ที่ทำให้ Tokenizer ล้ม (NullPointerException)
cleaned_df = cleaned_df.filter(
    col("review_text").isNotNull()
    & (length(col("review_text")) > 0)
    & col("label").isNotNull()
)

# [ผลลัพธ์ที่ 1] สั่งแสดงตารางข้อมูลที่ทำความสะอาดเรียบร้อยแล้ว
print("=== 1. Cleaned Dataframe ===")
cleaned_df.show(truncate=False)


# ====================================================================
# STEP 4: ประกอบ Pipeline และแปลงข้อมูลฟีเจอร์ (Feature Engineering)
# ====================================================================
# 1. สร้างตัวตัดคำ (Tokenizer): รับข้อความรีวิวเข้ามาตัดแบ่งออกเป็นคำ ๆ แยกย่อย
tokenizer = Tokenizer(inputCol="review_text", outputCol="review_words")

# 2. สร้างตัวลบคำฟุ่มเฟือย (StopWordsRemover): รับคำที่ตัดแล้วมาลบคำที่ไม่สื่อความหมายออก (เช่น is, am, are, the)
# โดยดึงชื่อคอลัมน์ผลลัพธ์จากตัวตัดคำมาใส่โดยอัตโนมัติผ่านฟังก์ชัน .getOutputCol()
stopwords_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="meaningful_words")

# 3. สร้างตัวคำนวณความถี่คำ (HashingTF): แปลงกลุ่มคำที่มีความหมายให้กลายเป็นเวกเตอร์ตัวเลขเชิงสถิติ (Features) เพื่อส่งให้ AI
hashing_tf = HashingTF(inputCol=stopwords_remover.getOutputCol(), outputCol="features")

# นำเอาทั้ง 3 ขั้นตอนด้านบนมาประกอบร่างร้อยเรียงเข้าด้วยกันเป็น Pipeline เดียวกัน
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf])


# ====================================================================
# STEP 5: จัดเตรียมชุดข้อมูลและประมวลผลผ่าน Pipeline (Data Splitting)
# ====================================================================
# ทำการแบ่งข้อมูลเป็น 2 ชุด: ชุดเทรนสำหรับสอน AI (70%) และชุดทดสอบสำหรับวัดผล (30%)
# seed=42 คือการล็อกสุ่ม เพื่อให้เวลากดรันโค้ดกี่ครั้งก็ได้การแบ่งข้อมูลเหมือนเดิมเป๊ะ
train_data, test_data = cleaned_df.randomSplit([0.7, 0.3], seed=42)

# [ผลลัพธ์ที่ 2] สั่งแสดงหน้าตาของชุดข้อมูลที่ใช้สอน AI (Train Dataset)
print("=== 2. Train Dataset Sample ===")
train_data.show(truncate=False)

# ส่งชุดข้อมูล Train เข้าไปให้เครื่องมือ Pipeline คำนวณจำจดขั้นตอนและแกะรอยคำศัพท์
pipeline_model = pipeline.fit(train_data)

# แปลงข้อมูลดิบของทั้งชุด Train และชุด Test ให้กลายเป็นเวกเตอร์ตัวเลข (Features) พร้อมใช้งาน
final_train_df = pipeline_model.transform(train_data)
final_test_df = pipeline_model.transform(test_data)


# ====================================================================
# STEP 6: ฝึกสอน AI และพยากรณ์ผลลัพธ์ (Model Training & Prediction)
# ====================================================================
# สร้างวัตถุโมเดลวิเคราะห์ความถดถอยโลจิสติก (Logistic Regression) สำหรับจำแนกประเภท
lr = LogisticRegression(featuresCol="features", labelCol="label")

# สั่งให้โมเดล AI ทำการเรียนรู้และฝึกฝนจากข้อมูลตาราง Train ที่ผ่านการแปลงฟีเจอร์แล้ว
lr_model = lr.fit(final_train_df)

# นำโมเดล AI ที่ฝึกเสร็จแล้ว ไปใช้คาดเดาและพยากรณ์คะแนนกับชุดข้อมูลทดสอบ (Test Dataframe)
predictions_df = lr_model.transform(final_test_df)

# [ผลลัพธ์ที่ 3] สั่งแสดงตารางเปรียบเทียบคำที่มีความหมาย, คะแนนจริง (label) และคำที่ AI เดา (prediction)
print("=== 3. Prediction Results Dataframe ===")
predictions_df.select("meaningful_words", "label", "prediction").show(truncate=False)


# ====================================================================
# STEP 7: ประเมินค่าความแม่นยำของตัวแบบ (Model Evaluation)
# ====================================================================
# สร้างตัววัดผลการจำแนกประเภทข้อมูล
# กำหนดให้เปรียบเทียบคอลัมน์คำตอบจริง (labelCol) กับคอลัมน์ที่ AI ทาย (predictionCol) โดยใช้เกณฑ์วัดความแม่นยำ (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

# คำนวณหาค่าความแม่นยำออกมาเป็นตัวเลขทศนิยม
accuracy_value = evaluator.evaluate(predictions_df)

# [ผลลัพธ์ที่ 4] แสดงผลค่าความแม่นยำ (Accuracy) ตัวเลขสุดท้ายออกทางหน้าจอ
print("=" * 50)
print(f"ค่าความแม่นยำของโมเดล (Model Accuracy): {accuracy_value}")
print("=" * 50)