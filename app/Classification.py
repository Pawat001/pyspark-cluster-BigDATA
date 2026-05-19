import os
# สไลด์หน้า 13, 19: นำเข้าไลบรารีพื้นฐานสำหรับเปิดใช้งาน SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType

# สไลด์หน้า 13, 19: นำเข้าเครื่องมือจัดการ Feature (StringIndexer, VectorAssembler)
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
# สไลด์หน้า 13, 19: นำเข้า Pipeline เพื่อใช้รวมขั้นตอนการทำงานเข้าด้วยกัน
from pyspark.ml import Pipeline
# สไลด์หน้า 13, 19: นำเข้าโมเดลที่จะใช้ทำ Assignment (Logistic Regression และ Decision Tree)
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
# สไลด์หน้า 13, 19: นำเข้าตัวประเมินผลประสิทธิภาพของโมเดล (MulticlassClassificationEvaluator)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# สไลด์หน้า 14, 20: เริ่มต้นเปิดใช้งาน SparkSession และตั้งชื่อแอปพลิเคชันว่า "Classification"
spark = SparkSession.builder.appName("Classification").getOrCreate()
# ซ่อนข้อความเตือนระบบ (Log) ที่ไม่จำเป็น ให้แสดงเฉพาะความผิดพลาดระดับ "ERROR" เท่านั้น
spark.sparkContext.setLogLevel("ERROR")

# สไลด์หน้า 14, 20: ทำการตรวจสอบ Path และโหลดไฟล์ข้อมูล FBLiveTH (fb_live_thailand.csv) เข้าสู่ DataFrame
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "fb_live_thailand.csv")
if not os.path.exists(csv_path):
    csv_path = "/app/fb_live_thailand.csv"

df = spark.read.format("csv").option("header", True).load(csv_path)

# สไลด์หน้า 14, 20: กำหนดรายชื่อคอลัมน์ของคุณลักษณะข้อมูล (Features/Inputs) ที่เราต้องการใช้ในการทำนาย
feature_cols = [
    "num_reactions",
    "num_comments",
    "num_shares",
    "num_likes",
    "num_loves",
    "num_wows",
    "num_hahas",
    "num_sads",
    "num_angrys",
]

# เลือกเฉพาะคอลัมน์เป้าหมาย (status_type) และคอลัมน์คุณลักษณะ (feature_cols) มาใช้งาน
select_cols = ["status_type"] + feature_cols
df = df.select(*select_cols)

# แปลงชนิดข้อมูลของคอลัมน์คุณลักษณะทั้งหมดให้เป็นตัวเลขทศนิยม (DoubleType) เพื่อให้โมเดลสามารถคำนวณได้
for col_name in feature_cols:
    df = df.withColumn(col_name, df[col_name].cast(DoubleType()))

# จัดการข้อมูลสูญหาย (Missing Value) โดยหากช่องไหนเป็นค่าว่าง (Null) ให้เติมเลข 0 แทนลงไป
df = df.na.fill(0)

# สไลด์หน้า 16, 22: แบ่งข้อมูลออกแบบสุ่ม (randomSplit) เป็นข้อมูลสำหรับฝึกสอน 80% และข้อมูลสำหรับทดสอบ 20%
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# สไลด์หน้า 14, 20: ใช้ StringIndexer แปลงคอลัมน์กลุ่มที่เป็นข้อความ (status_type) ให้กลายเป็นดัชนีตัวเลข (label)
label_indexer = StringIndexer(inputCol="status_type", outputCol="label", handleInvalid="keep")

# สไลด์หน้า 14, 21: ใช้ VectorAssembler รวมคอลัมน์คุณลักษณะหลายๆ ตัวเข้าด้วยกันให้เป็น Vector ตัวเดียว (features)
vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ทำการปรับค่าคุณลักษณะให้อยู่ในมาตรฐานเดียวกัน (StandardScaler) ป้องกันไม่ให้ตัวเลขที่มากเกินไปส่งผลต่อโมเดล
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# สไลด์หน้า 17, 23: ตั้งค่าตัวประเมินผลโดยชี้เป้าไปที่คอลัมน์ "label" (ค่าจริง) และคอลัมน์ "prediction" (ค่าที่ทายได้)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")


print("=" * 50)
print("=== 1. Logistic Regression ===")
print("=" * 50)

# สไลด์หน้า 12, 15: สร้างโมเดล Logistic Regression โดยกำหนดให้เรียนรู้จากคอลัมน์คุณลักษณะที่ทำสเกลแล้ว และระบุรอบการเทรนสูงสุด (maxIter)
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=50)

# สไลด์หน้า 15: สร้าง Pipeline เพื่อมัดรวมขั้นตอนการทำงาน (จัดกลุ่มดัชนี -> รวมเวกเตอร์ -> ปรับสเกล -> เข้าโมเดล LR)
lr_pipeline = Pipeline(stages=[label_indexer, vec_assembler, scaler, lr])

# สไลด์หน้า 16: ส่งข้อมูลฝึกสอน (train_df) เข้าไปเรียนรู้ใน Pipeline จนได้ออกมาเป็น Pipeline Model
lr_model = lr_pipeline.fit(train_df)

# สไลด์หน้า 16: นำโมเดลที่ฝึกเสร็จแล้วมาทำการพยากรณ์ข้อมูลทดสอบ (test_df) จะได้ผลการทำนายในคอลัมน์ prediction
lr_predictions = lr_model.transform(test_df)

# สไลด์หน้า 17: ดึงค่ามาตรวัดประสิทธิภาพต่างๆ ออกมาคำนวณผ่านตัว evaluator
lr_accuracy = evaluator.setMetricName("accuracy").evaluate(lr_predictions)            # สไลด์หน้า 7: คำนวณ Accuracy
lr_precision = evaluator.setMetricName("weightedPrecision").evaluate(lr_predictions)    # สไลด์หน้า 8: คำนวณ Precision
lr_recall = evaluator.setMetricName("weightedRecall").evaluate(lr_predictions)          # สไลด์หน้า 9: คำนวณ Recall
lr_f1 = evaluator.setMetricName("f1").evaluate(lr_predictions)                        # สไลด์หน้า 10: คำนวณ F1-Score

# แสดงค่าผลลัพธ์การประเมินของ Logistic Regression ทางหน้าจอ (ฟอร์แมตทศนิยม 4 ตำแหน่ง)
print(f"Logistic Regression Accuracy:   {lr_accuracy:.4f}")
print(f"Logistic Regression Precision:  {lr_precision:.4f}")
print(f"Logistic Regression Recall:     {lr_recall:.4f}")
print(f"Logistic Regression F1 Measure: {lr_f1:.4f}")

# สไลด์หน้า 16: แสดงตัวอย่างตารางผลลัพธ์เปรียบเทียบระหว่างข้อมูลจริง (status_type) กับสิ่งที่ทาย (prediction) 10 แถวแรก
print("\nSample Predictions (Top 10 rows):")
lr_predictions.select("status_type", "prediction").show(10, truncate=False)


print("=" * 50)
print("=== 2. Decision Tree Classification ===")
print("=" * 50)

# สไลด์หน้า 18, 22: สร้างโมเดล Decision Tree สำหรับการจำแนกประเภทข้อมูล
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# สไลด์หน้า 21: สร้าง Pipeline สรุปลำดับขั้นตอนของ Decision Tree (จัดกลุ่มดัชนี -> รวมเวกเตอร์ -> เข้าโมเดล DT)
dt_pipeline = Pipeline(stages=[label_indexer, vec_assembler, dt])

# สไลด์หน้า 21, 22: ส่งข้อมูลฝึกสอน (train_df) เข้าไปสร้างแผนภูมิต้นไม้การตัดสินใจ (Fit Model)
dt_model = dt_pipeline.fit(train_df)

# สไลด์หน้า 21, 22: นำต้นไม้การตัดสินใจมาทายผลข้อมูลทดสอบ (test_df)
dt_predictions = dt_model.transform(test_df)

# สไลด์หน้า 23: คำนวณหามาตรวัดความแม่นยำรวมถึงค่าความผิดพลาดตามเงื่อนไขของโจทย์
dt_accuracy = evaluator.setMetricName("accuracy").evaluate(dt_predictions)            # สไลด์หน้า 7: คำนวณ Accuracy
dt_precision = evaluator.setMetricName("weightedPrecision").evaluate(dt_predictions)    # สไลด์หน้า 8: คำนวณ Precision
dt_recall = evaluator.setMetricName("weightedRecall").evaluate(dt_predictions)          # สไลด์หน้า 9: คำนวณ Recall
dt_f1 = evaluator.setMetricName("f1").evaluate(dt_predictions)                        # สไลด์หน้า 10: คำนวณ F1-Score
dt_test_error = 1.0 - dt_accuracy                                                      # สไลด์หน้า 23: คำนวณค่า Test Error (1.0 - accuracy)

# แสดงค่าผลลัพธ์การประเมินของ Decision Tree ทางหน้าจอ
print(f"Decision Tree Accuracy:   {dt_accuracy:.4f}")
print(f"Decision Tree Precision:  {dt_precision:.4f}")
print(f"Decision Tree Recall:     {dt_recall:.4f}")
print(f"Decision Tree F1 Measure: {dt_f1:.4f}")
print(f"Decision Tree Test Error: {dt_test_error:.4f}")

# สไลด์หน้า 22: แสดงตัวอย่างตารางผลลัพธ์เปรียบเทียบของโมเดล Decision Tree 10 แถวแรก
print("\nSample Predictions (Top 10 rows):")
dt_predictions.select("status_type", "prediction").show(10, truncate=False)

# สั่งปิดและคืนหน่วยความจำระบบให้กับ SparkSession เมื่อโปรแกรมทำงานเสร็จสิ้นทั้งหมด
spark.stop()