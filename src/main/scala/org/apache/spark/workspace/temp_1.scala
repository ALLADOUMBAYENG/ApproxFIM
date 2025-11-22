package org.apache.spark.workspace

import breeze.linalg.{*, DenseMatrix, DenseVector, diag, eigSym, norm, sum}
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{collect_list, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

object temp_1 {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val K_true = 3
    val dataDF: DataFrame = spark.read.parquet("data/classification_100_2_0.54_4_32M.parquet")
    import spark.implicits._
    dataDF.show()
    println(dataDF.rdd.getNumPartitions)
    val kmeans = new KMeans().setK(K_true).setSeed(9L)
    val model = kmeans.fit(dataDF)
    val predictions = model.transform(dataDF)
    val centers = model.clusterCenters
    val features = centers.union(centers)
    val predictions1 = spectralClustering(features, spark, K_true)
    val data = predictions1.withColumnRenamed("prediction", "label")
    predictions1.show()
    // 定义一个UDF来将Vector转换为数组
    val vectorToArray: UserDefinedFunction = udf((vec: Vector) => vec.toArray)

    // 将Vector转换为数组
    val dataWithArray = data.withColumn("featureArray", vectorToArray($"features"))

    // 自定义聚合函数来计算数组逐元素均值
    val avgArrayUDF: UserDefinedFunction = udf((arrays: Seq[Seq[Double]]) => {
      if (arrays.isEmpty) {
        Seq.empty[Double]
      } else {
        val n = arrays.length
        val sumArray = arrays.reduce((a, b) => a.zip(b).map { case (x, y) => x + y })
        sumArray.map(_ / n)
      }
    })

    // 计算每个label的feature均值
    val mergedDF = dataWithArray
      .groupBy("label")
      .agg(avgArrayUDF(collect_list($"featureArray")).alias("meanFeatureArray"))

    // 定义一个UDF将数组转换回Vector
    val arrayToVector: UserDefinedFunction = udf((array: Seq[Double]) => Vectors.dense(array.toArray))

    // 将均值数组转换回Vector
    val finalDF = mergedDF.withColumn("features", arrayToVector($"meanFeatureArray"))
      .drop("meanFeatureArray")

    // 显示结果
    finalDF.show(truncate = false)

  }

  def spectralClustering(features: Array[org.apache.spark.ml.linalg.Vector], spark: SparkSession, K_true: Int): DataFrame = {
    import spark.implicits._
    // 计算相似性矩阵
    val n = features.length
    val similarityMatrix = DenseMatrix.zeros[Double](n, n)
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        val dist = norm(DenseVector(features(i).toArray) - DenseVector(features(j).toArray))
        similarityMatrix(i, j) = math.exp(-dist * dist / 2.0)
      }
    }
    println(similarityMatrix(1, 2))

    // 计算度矩阵
    val degreeMatrix = diag(similarityMatrix(*, ::).map(row => sum(row)))

    // 计算归一化的拉普拉斯矩阵
    val laplacianMatrix = degreeMatrix - similarityMatrix
    val normalizedLaplacian = laplacianMatrix

    // 计算拉普拉斯矩阵的特征值和特征向量

    val eigSym.EigSym(eigenvalues, eigenvectors) = eigSym(normalizedLaplacian)

    // 选择前k个最小的特征向量

    val selectedEigenvectors = eigenvectors(::, 0 until K_true)

    // 将特征向量作为新特征，并应用K-means进行聚类
    val featureRows1 = for (i <- 0 until selectedEigenvectors.rows) yield (i, selectedEigenvectors(i, ::).t.toArray)
    val eigenDF = featureRows1.map { case (id, vec) => (id, Vectors.dense(vec)) }.toDF("id", "features")

    val kmeans1 = new KMeans().setK(K_true).setSeed(1L)
    val kmeansModel1 = kmeans1.fit(eigenDF)

    // 显示聚类结果
    val predictions1 = kmeansModel1.transform(eigenDF)
    predictions1.select("features", "prediction")

  }
}
