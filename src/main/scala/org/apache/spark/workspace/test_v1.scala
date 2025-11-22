package org.apache.spark.workspace

// Spark code of A scalable framework for cluster ensembles
import breeze.linalg.{*, DenseMatrix, diag, eigSym, sum}
import breeze.numerics.{abs, exp}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.RspContext.NewRDDFunc
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{SingularValueDecomposition, distributed}
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id, row_number}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import smile.math.matrix.Matrix

import java.util.Date


object test_v1 {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession

    // 获取 Hadoop 文件系统对象
    val hadoopConf = new Configuration()
    val hadoopFs = FileSystem.get(hadoopConf)
    // 指定要检查的目录路径
    val outputPath = new Path("output")
    // 检查目录是否存在
    if (hadoopFs.exists(outputPath)) {
      println(s"Directory $outputPath exists. Deleting...")
      // 删除目录
      hadoopFs.delete(outputPath, true)
      println(s"Directory $outputPath has been deleted.")
    } else {
      println(s"Directory $outputPath does not exist.")
    }
    val now_start: Date=new Date()
    val sparkConf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    val K_true = 10
    //    val dataDF = spark.read.parquet("data/classification_100_2_0.54_4_32M.parquet").rdd.toRSP(100)
    //    val dataDF1 = dataDF.map(a => (a.getInt(0).toDouble, a.getAs[DenseVector]("features"))).toDF("label","features")
    //    val dataDF2 = spark.read.parquet("data/classification_100_2_0.54_4_32M.parquet")
    //    dataDF1.show(100)
    //    println(dataDF1.rdd.getNumPartitions)
    val dataRDD = spark.read.text("data/extendMnistPCA86.txt").rdd
    val dataRspRDD:RspRDD[Row] = dataRDD.toRSP(126)
    val dataDF2 = rddtodataframe(dataRDD,spark)
    print(dataRDD.count())

    val sampleClustering = rddtodataframe(dataRspRDD.coalesce(1, false, Option(new SonPartitionCoalescer(Array(0)))),spark)
    val samplenum = sampleClustering.count().toInt
    sampleClustering.persist()
    val numberOfSample:Int = 5
    var flag = 0
    val array: Array[Array[Int]] = Array.ofDim[Int](numberOfSample, sampleClustering.count().toInt)
    while(flag<numberOfSample){
      val predictions1 = spectralClustering(rddtodataframe(dataRspRDD.coalesce(1, false, Option(new SonPartitionCoalescer(Array(flag)))),spark),spark,K_true)
      val finalPredictionsResult = predictions1.withColumnRenamed("prediction", "label")
      array(flag) = anchorMethor(finalPredictionsResult,sampleClustering,spark,K_true)
      val outputPath = "output/centersLabel/"+flag.toString+"/" +".parquet"
      finalPredictionsResult.write.parquet(outputPath)
      //      predictions1.show(10)
      //      predictions1.select("label","prediction").coalesce(1).write.csv("output/result")
      flag = flag+1
    }

    val binaryMatrixArray: Array[Array[Array[Int]]] = array.map(ele => {
      val matrix: Array[Array[Int]] = Array.ofDim[Int](ele.size, K_true)
      for (i <- 0 until ele.size) {
        matrix(i)(ele(i)) += 1
      }
      matrix
    })
    // 创建一个全0的二维数组
    val zeroArray: Array[Array[Double]] = Array.fill(samplenum, samplenum)(0.0)
    // 使用 Matrix 类创建矩阵
    var Bmatrix_temp = new Matrix(zeroArray)
    var flag1 = 0
    while(flag1<numberOfSample){
      val temp = binaryMatrixArray(flag1)
      val rows = temp.length
      val cols = temp(0).length
      val matrix_temp = temp.map(_.map(_.toDouble))
      val matrix = new Matrix(rows,cols, matrix_temp)
      val matrix1 = matrix.mt(matrix)
      for (i <- 0 until samplenum) {
        for (j <- 0 until samplenum) {
          Bmatrix_temp(i,j) = matrix1.get(i, j) + Bmatrix_temp.get(i, j)
        }
      }
      flag1 = flag1+1
    }

    for (i <- 0 until samplenum) {
      for (j <- 0 until samplenum) {
        Bmatrix_temp(i,j) = Bmatrix_temp(i,j)/samplenum
      }
    }

    var flag2 = 0
    var result_temp = 0.0
    val re = Array.ofDim[Double](samplenum)
    while(flag2<numberOfSample){
      val temp = binaryMatrixArray(flag2)
      val rows = temp.length
      val cols = temp(0).length
      val matrix_temp = temp.map(_.map(_.toDouble))
      val matrix = new Matrix(rows,cols, matrix_temp)
      val matrix1 = matrix.mt(matrix)
      // 执行矩阵减法操作（手动元素级操作）
      val result = new Matrix(matrix1.nrows, matrix1.ncols)
      for (i <- 0 until matrix1.nrows) {
        for (j <- 0 until matrix1.ncols) {
          result_temp = abs(matrix1(i, j) - Bmatrix_temp(i, j))+result_temp
        }
      }
      re(flag2) = result_temp
      result_temp = 0.0
      flag2 = flag2+1
    }
    var flagfinal = 0
    for(i<-0 until numberOfSample-1){
      print(re(i))
      print("\n")
      if(re(i)>re(i+1)){
        flagfinal = i+1
      }
    }
    val finalPredictionsResultensemble = spark.read.parquet("output/centersLabel/" + flagfinal.toString + "/" + ".parquet")
    val finalresult = finalPrediction(finalPredictionsResultensemble, dataDF2, spark, K_true)
    finalresult.select("label","prediction").coalesce(1).write.csv("output/result")

    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("===============================================" +
      "\n"+"Running time = "+ runningTime +"ms"
      +"\n"+"=================================================="+"\n")

  }

  def anchorMethor(featureClustering: DataFrame, sampleClustering: DataFrame, spark: SparkSession, K_true: Int): Array[Int] = {
    //用kmeans的方法
    val kmeans1 = new KMeans().setK(K_true * 10).setSeed(1L)
    val kmeansModel1 = kmeans1.fit(featureClustering)
    val centroids = kmeansModel1.clusterCenters
    // 显示聚类结果
    val centroidsDF: DataFrame = spark.createDataFrame(centroids.zipWithIndex.map {
      case (vector, index) =>
        // 将每个聚类中心转换为 DenseVector
        val denseVector = new DenseVector(vector.toArray)
        (index, denseVector)
    }).toDF("cluster", "features")
    //将样本的结果传给聚类中心
    val knn = new KNNClassifier()
      .setTopTreeSize(5)
      .setK(1)
    val knnModel = knn.fit(featureClustering.withColumn("label", col("label").cast("double")))
    val centerPredictions = knnModel.transform(centroidsDF)
    centerPredictions.select("features", "prediction").show()
    //将聚类中心的结果传给输入数据
    val knn1 = new KNNClassifier()
      .setTopTreeSize(5)
      .setK(1)
    val knnModel1 = knn1.fit(centerPredictions.select("features", "prediction").withColumnRenamed("prediction", "label"))
    val dataPredictions = knnModel1.transform(sampleClustering)
    dataPredictions.select("features", "prediction").show()
    val rows = dataPredictions.select("prediction").collect().map(k => k.getDouble(0)).map(value=>value.toInt)
    rows

  }

  def finalPrediction(featureClustering: DataFrame, sampleClustering: DataFrame, spark: SparkSession, K_true: Int): DataFrame = {
    //用kmeans的方法
    val kmeans1 = new KMeans().setK(K_true * 10).setSeed(1L)
    val kmeansModel1 = kmeans1.fit(featureClustering)
    val centroids = kmeansModel1.clusterCenters
    // 显示聚类结果
    val centroidsDF: DataFrame = spark.createDataFrame(centroids.zipWithIndex.map {
      case (vector, index) =>
        // 将每个聚类中心转换为 DenseVector
        val denseVector = new DenseVector(vector.toArray)
        (index, denseVector)
    }).toDF("cluster", "features")
    //将样本的结果传给聚类中心
    val knn = new KNNClassifier()
      .setTopTreeSize(5)
      .setK(1)
    val knnModel = knn.fit(featureClustering.withColumn("label", col("label").cast("double")))
    val centerPredictions = knnModel.transform(centroidsDF)
    //centerPredictions.select("features", "prediction").show()
    //将聚类中心的结果传给输入数据
    val knn1 = new KNNClassifier()
      .setTopTreeSize(5)
      .setK(1)
    val knnModel1 = knn1.fit(centerPredictions.select("features", "prediction").withColumnRenamed("prediction", "label"))
    val dataPredictions = knnModel1.transform(sampleClustering)
    dataPredictions.select("features", "prediction","label")
  }

  def rddtodataframe(dataRspRDD: RDD[Row], spark: SparkSession): DataFrame={
    //格式化Row为(Array[Int], Array[Array[Double]])
    import spark.implicits._
    val wrapperRDD = new RspRDD(dataRspRDD.glom().map(f => (
      f.map(r => {
        val doubles = r.toString().replace('[', ' ').replace(']', ' ').trim.split(",").map(_.toDouble)
        doubles(doubles.length - 1).toInt
      }),
      f.map(r => {
        val doubles = r.toString().replace('[', ' ').replace(']', ' ').trim.split(",").map(_.toDouble)
        doubles.dropRight(1)
      })
    )))

    val value1: RDD[(Array[Int], Array[Array[Double]])] = wrapperRDD.map(row => (row._1, row._2))
    //    val value_all: RDD[(Array[Int], Array[Array[Double]])] = value1.coalesce(1, false, Option(new SonPartitionCoalescer(Array(0))))
    val dataAll: DataFrame = value1.flatMap { case (labels, features) =>
      labels.zip(features).map { case (label, feature) =>
        (label, Vectors.dense(feature))
      }
    }.toDF("label", "features")
    dataAll
  }

  def spectralClustering(featuresDF: DataFrame, spark: SparkSession, K_true: Int): DataFrame = {
        // 提取特征向量并转换为 Array[Double]
        import spark.implicits._
        val features = featuresDF.select("features").collect().map {
          case Row(vector: Vector) => vector
        }
       val value: RDD[Vector] = spark.sparkContext.parallelize(features)
        // 计算相似性矩阵
        //    val n = features.length
        //    val symmetricMatrix = DenseMatrix.zeros[Double](n, n)
        //    for (i <- 0 until n) {
        //      for (j <- 0 until n) {
        //        val dist = norm(DenseVector(features(i)) - DenseVector(features(j)))
        //        symmetricMatrix(i, j) = math.exp(-dist * dist / 2.0)
        //      }
        //    }
        val n = features.length
        val similarityMatrix = DenseMatrix.zeros[Double](n, n)
        // 计算每对点的欧几里得距离并存储
        val distances = Array.ofDim[Double](n, n)
        for (i <- 0 until n) {
          for (j <- 0 until n) {
            // 计算欧几里得距离
            distances(i)(j) = Vectors.sqdist(features(i), features(j)) // 使用 sqdist 计算欧几里得距离的平方
          }
        }
        val kMax = 30
        // 对每个点的距离进行处理
        for (i <- 0 until n) {
          val sortedDistances = distances(i).zipWithIndex.sortBy(_._1).map { case (dist, idx) => (dist, idx) }
          // 计算相似度
          for (j <- 0 until n if j != i) {
            val distanceIJ = distances(i)(j)
            val kMaxDistance = sortedDistances(kMax)._1 // kMax+1是索引kMax
            if (distanceIJ <= kMaxDistance) {
              val sumDistances = sortedDistances.slice(1, kMax + 1).map(_._1).sum // 从l=2开始
              val numerator = sumDistances - (kMax - 1) * distanceIJ
              similarityMatrix(i, j) = math.max(numerator / sumDistances, 0.0)
            } else {
              similarityMatrix(i, j) = 0.0
            }
          }
        }
        println(similarityMatrix(1, 2))
        val symmetricMatrix = (similarityMatrix + similarityMatrix.t) / 2.0

        val degreeMatrix: DenseMatrix[Double] = diag(symmetricMatrix(*, ::).map(row => 1/math.sqrt(sum(row))))
        val symmetricMatrixNormalized = degreeMatrix*symmetricMatrix*degreeMatrix
        val breezeMatrix=symmetricMatrixNormalized
        // 将 Breeze DenseMatrix 转换为 Spark MLlib Matrix
        val rows = breezeMatrix.rows
        val cols = breezeMatrix.cols
        // 使用 Spark MLlib 的 `Vector` 类型来创建每一行的数据
        val data = for (i <- 0 until rows) yield {
          org.apache.spark.mllib.linalg.Vectors.dense(breezeMatrix(i, ::).t.toArray)  // 转换为 Spark MLlib 的 Vector 类型
        }
        // 将数据转换为 Spark 的 RDD
        val rdd = spark.sparkContext.parallelize(data)
        // 使用 RowMatrix 的 fromRDD 方法来构造 RowMatrix
        val rowMatrix = new RowMatrix(rdd)
        // 使用 Spark MLlib 执行奇异值分解（SVD）
        val eigen: SingularValueDecomposition[RowMatrix, linalg.Matrix] = rowMatrix.computeSVD(K_true, computeU = true)
        println("Singular values: " + eigen.s)

        val U_dense = eigen.U.rows.map { row =>
          // 将每行转换为 DenseVector
          new DenseVector(row.toArray)
        }
        //eigen.U.rows.foreach(println)
        // 转换成 DataFrame
        val U_df = spark.createDataFrame(U_dense.map(Tuple1.apply)).toDF("features")
        U_df.show()
        print("数据量"+U_df.count())

        val kmeans1 = new KMeans().setK(K_true).setSeed(1L)
        val kmeansModel1 = kmeans1.fit(U_df)
        val predictions1: DataFrame = kmeansModel1.transform(U_df)
        val predDF: DataFrame = predictions1.rdd.map(a => a.getInt(1)).zipWithIndex().toDF("prediction", "index")
        val feature: DataFrame = value.zipWithIndex().toDF("features", "index")
        val predDFfinal = predDF.join(feature.select("features","index"),"index").drop("index")
        predDFfinal

//内存计算的方法
        // 计算度矩阵
//        val degreeMatrix: DenseMatrix[Double] = diag(symmetricMatrix(*, ::).map(row => sum(row)))
//        val degreeMatrixnormalize: DenseMatrix[Double] = diag(symmetricMatrix(*, ::).map(row => 1/math.sqrt(sum(row))))
//        // 计算归一化的拉普拉斯矩阵
//        val laplacianMatrix = degreeMatrix - symmetricMatrix
//        val normalizedLaplacian = degreeMatrixnormalize*laplacianMatrix*degreeMatrixnormalize
//            // 计算拉普拉斯矩阵的特征值和特征向量
//            val eigSym.EigSym(eigenvalues, eigenvectors) = eigSym(normalizedLaplacian)
//            // 选择前K个最小的特征向量
//            val selectedEigenvectors = eigenvectors(::, 0 until K_true)
//            // 将特征向量作为新特征，并应用K-means进行聚类
//            val featureRows1 = for (i <- 0 until selectedEigenvectors.rows) yield {
//              Row(i, Vectors.dense(selectedEigenvectors(i, ::).t.toArray))
//            }
//            // 定义 schema，包括 Vector 类型
//            val schema = StructType(Array(
//              StructField("id", IntegerType, false),
//              StructField("features", new VectorUDT, false)
//            ))
//            // 创建 DataFrame
//            val eigenDF = spark.createDataFrame(spark.sparkContext.parallelize(featureRows1), schema)
//            val kmeans1 = new KMeans().setK(K_true).setSeed(1L)
//            val kmeansModel1 = kmeans1.fit(eigenDF)
//            // 显示聚类结果
//            val predictions1 = kmeansModel1.transform(eigenDF)
//            val predDF: DataFrame = predictions1.rdd.map(a => a.getInt(2)).zipWithIndex().toDF("prediction", "index")
//            val feature: DataFrame = value.zipWithIndex().toDF("features", "index")
//            val predDFfinal = predDF.join(feature.select("features","index"),"index").drop("index")
//            predDFfinal

//            val predDF = predictions1.select("features","prediction").withColumn("index", row_number().over(Window.orderBy(lit(1))))
//            val feature = featuresDF.withColumn("index", row_number().over(Window.orderBy(lit(1))))
//            val predDFfinal = predDF.select("prediction","index").join(feature.select("label","features","index"),"index").drop("index")
//            predDFfinal
//kmeans的方法
//    val kmeans1 = new KMeans().setK(K_true).setSeed(1L)
//    val kmeansModel1 = kmeans1.fit(featuresDF)
//    val predDFfinal = kmeansModel1.transform(featuresDF)
//    predDFfinal
  }

}
