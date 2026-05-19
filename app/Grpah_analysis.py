# --- ส่วนที่ 1: นำเข้า Libraries และสร้าง Spark Session ---
import os
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc

spark = SparkSession.builder.appName("GraphAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- ส่วนที่ 2: สร้างข้อมูลปม (Vertices) และเส้นเชื่อม (Edges) ---
v = spark.createDataFrame([
    ("Alice", 45), ("Jacob", 43), ("Roy", 21),
    ("Ryan", 49), ("Emily", 24), ("Sheldon", 52)
], ["id", "age"])

e = spark.createDataFrame([
    ("Sheldon", "Alice", "Sister"), ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"), ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"), ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son")
], ["src", "dst", "relation"])

g = GraphFrame(v, e)

# --- ส่วนที่ 3: เริ่มต้นทำ Assignment (หน้า 5-12) ---
# หมายเหตุ: โค้ดด้านล่างเรียงตามหัวข้อในสไลด์เพื่อให้อ่านง่าย

# 1. Querying and Filtering (หน้า 5)
print("--- 1. Querying ---")
g.edges.groupBy("src", "dst").count().orderBy(desc("count")).show(5)

print("--- 2. Filtering ---")
g.edges.where("src = 'Alice' OR dst = 'Alice'").groupBy("src", "dst").count().orderBy(desc("count")).show(5)

# 2. Subgraph (หน้า 6)
alice_edges = g.edges.where("src = 'Alice' OR dst = 'Alice'")
alice_subgraph = GraphFrame(g.vertices, alice_edges)
print("--- 2. Subgraph (Alice) ---")
alice_subgraph.edges.show(truncate=False)

# 3. Motif Finding (หน้า 7-8)
print("--- 3. Motif Finding ---")
motifs = g.find("(a) - [ab] -> (b)")
motifs.show(5)

# 4. PageRank (หน้า 9)
print("--- 4. PageRank ---")
rank = g.pageRank(resetProbability=0.15, maxIter=5)
rank.vertices.orderBy(desc("pagerank")).show(5)

# 5. In-Degree (หน้า 10)
print("--- 5. In-Degree ---")
in_deg = g.inDegrees
in_deg.orderBy(desc("inDegree")).show(5)

# 6. Connected Components (หน้า 11)
print("--- 6. Strongly Connected Components ---")
checkpoint_dir = os.path.join("/tmp", "checkpoints")
spark.sparkContext.setCheckpointDir(checkpoint_dir)
scc = g.stronglyConnectedComponents(maxIter=5)
scc.show()

# 7. Breadth-First Search (หน้า 12)
print("--- 7. Breadth-First Search ---")
g.bfs(fromExpr = "id = 'Alice'", toExpr = "id = 'Jacob'", maxPathLength = 2).show()

spark.stop()