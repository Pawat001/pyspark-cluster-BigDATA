import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, desc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Create SparkSession & Load Data (หน้า 13)
spark = SparkSession.builder.appName("LinearRegressionAssign").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "fb_live_thailand.csv")
if not os.path.exists(csv_path):
        csv_path = "/app/fb_live_thailand.csv"

df = spark.read.csv(csv_path, header=True, inferSchema=True)

# 2. StringIndexer (หน้า 13)
indexer_rxn = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind", handleInvalid="keep")
indexer_love = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind", handleInvalid="keep")

# 3. Create train and test datasets (หน้า 15)
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# 4. VectorAssembler (หน้า 13) - อิงตามสไลด์ที่ให้นำ index มารวมกัน 
# *หมายเหตุ: ปกติ Feature จะไม่รวม Label (num_loves_ind) เข้าไป แต่เขียนตามสไลด์คำสั่งครับ
assembler = VectorAssembler(inputCols=["num_reactions_ind"], outputCol="features")

# 5. Create Linear Regression (หน้า 14)
lr = LinearRegression(labelCol="num_loves_ind", featuresCol="features", 
                      maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 6. Create Pipeline & Fit/Transform (หน้า 14-15)
pipeline = Pipeline(stages=[indexer_rxn, indexer_love, assembler, lr])
pipeline_model = pipeline.fit(train_data)
predictions = pipeline_model.transform(test_data)

# Show 5 rows of predictions (หน้า 15) 
predictions.select("num_loves_ind", "prediction").show(5)

# 7. Evaluate Model (MSE และ R2) (หน้า 16) 
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

mse = evaluator.setMetricName("mse").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R2): {r2}")

# 8. Plot กราฟ (หน้า 18)
plot_data = predictions.select(col("num_loves_ind").cast(IntegerType()).alias("num_loves"), 
                               col("prediction").cast(IntegerType()).alias("prediction")) \
                       .orderBy(desc("prediction"))
pdf = plot_data.toPandas()

sns.lmplot(x="num_loves", y="prediction", data=pdf)
plt.title("Linear Regression: Actual vs Prediction")

output_img = os.path.join(base_dir, "output_linear_regression.png")
plt.savefig(output_img)
print(f"-> Saved plot successfully to {output_img}")

spark.stop()