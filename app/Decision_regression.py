from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# 1. Create SparkSession & Load Data (หน้า 21)
spark = SparkSession.builder.appName("DTreeRegressionAssign").getOrCreate()
df = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# 2. StringIndexer (หน้า 21) 
indexer_rxn = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_love = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# 3. OneHotEncoder (หน้า 21) 
encoder_rxn = OneHotEncoder(inputCol="num_reactions_ind", outputCol="num_reactions_vec")
encoder_love = OneHotEncoder(inputCol="num_loves_ind", outputCol="num_loves_vec")

# 4. VectorAssembler (หน้า 22) 
assembler = VectorAssembler(inputCols=["num_reactions_vec"], outputCol="features")

# 5. Create pipeline (Indexer -> Encoder -> Assembler) (หน้า 22)
prep_pipeline = Pipeline(stages=[indexer_rxn, indexer_love, encoder_rxn, encoder_love, assembler])
prep_model = prep_pipeline.fit(df)
df_prepared = prep_model.transform(df)

# 6. Create train and test datasets (หน้า 23)
train_data, test_data = df_prepared.randomSplit([0.7, 0.3], seed=42)

# 7. Create Decision Tree Regression & Fit/Transform (หน้า 23)
dt = DecisionTreeRegressor(labelCol="num_loves_ind", featuresCol="features")
dt_model = dt.fit(train_data)
predictions = dt_model.transform(test_data)

# Show 5 rows
predictions.select("num_loves_ind", "prediction").show(5)

# 8. Evaluate Model (R2) (หน้า 24) 
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R-Squared (R2): {r2}")