from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("testKMeans").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("csv").option("header", True).load("/app/fb_live_thailand.csv")

df = df.select(df.num_sads.cast(DoubleType()), df.num_reactions.cast(DoubleType()))

vec_assembler = VectorAssembler(inputCols=["num_sads", "num_reactions"], outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

k_values = []
print("=== Finding Best k ===")
for i in range(2, 5):
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=i)
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
    model = pipeline.fit(df)
    output = model.transform(df)
    evaluator = ClusteringEvaluator(predictionCol="prediction_col", featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
    score = evaluator.evaluate(output)
    k_values.append(score)
    print(f"k = {i} | Silhouette Score: {score}")

best_k = k_values.index(max(k_values)) + 2
print(f"\n-> The best k is {best_k} with score {max(k_values)}\n")

print("=== Training Final Model ===")
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=best_k)

pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
model = pipeline.fit(df)

predictions = model.transform(df)
evaluator = ClusteringEvaluator(predictionCol="prediction_col", featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance = {silhouette}")

clustered_data_pd = predictions.toPandas()

plt.scatter(clustered_data_pd["num_reactions"], clustered_data_pd["num_sads"], c=clustered_data_pd["prediction_col"])
plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("K-means Clustering")
plt.colorbar().set_label("Cluster")

output_img = "/app/output_Kmean/kmeans_plot.png"
plt.savefig(output_img)
print(f"-> Saved plot successfully to {output_img}")

spark.stop()