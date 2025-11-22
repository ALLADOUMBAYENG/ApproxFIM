package org.apache.spark.workspace

import org.apache.spark.RspContext.RspRDDFunc
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

//import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{NewRDDFunc, SparkSessionFunc}
//import org.apache.spark.sql.RspContext._
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.sql.{Row, SparkSession}
import smile.clustering.{KMeans => smileKMeans}
import smile.math.matrix.Matrix

object test_v3 {
  def main(args: Array[String]): Unit = {
    //变量
    val K_true = 10
    val K = 10*15
    val H = 10
    //初始化环境
    val sparkConf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    //读取数据
    val dataRDD: RspRDD[Row] = spark.rspRead.text("data/extendMnistPCA86.txt").rdd
    //Rsp化
    val rspRDD: RspRDD[Row] = dataRDD.toRSP(30)
    println(rspRDD.getNumPartitions)
    //格式化Row为(Array[Int], Array[Array[Double]])
    val wrapperRDD = new RspRDD(rspRDD.glom().map(f => (
      f.map(r => {
        val doubles = r.toString().replace('[', ' ').replace(']', ' ').trim.split(",").map(_.toDouble)
        doubles(doubles.length - 1).toInt
      }),
      f.map(r => {
        val doubles = r.toString().replace('[', ' ').replace(']', ' ').trim.split(",").map(_.toDouble)
        doubles.dropRight(1)
      })
    )))
    //取出部分数据块, 前面四个用于训练，后面用于测试
    val subRddArray: Array[RspRDD[(Array[Int], Array[Array[Double]])]] = wrapperRDD.getTwoRandomPartitions(H+1, 1)
    val trainRDD: RspRDD[(Array[Int], Array[Array[Double]])] = subRddArray(0)
    val testRDD = subRddArray(1)
    //进行LO操作,调用smile库里面的kmeans
    val centerRDD: RDD[smileKMeans] = trainRDD.mapPartitions(f => {
      val partitionRes = smileKMeans.lloyd(f.next()._2, K)
      if (f.hasNext) throw new Exception("The input iterator of LO should only has one element.")
      Iterator.single(partitionRes)
    })
    val modelArray: Array[smileKMeans] = centerRDD.collect()
    // 取出测试数据
    val dataRSP: Array[Array[Double]] = testRDD.collect()(0)._2
    val array: Array[Array[Int]] = Array.ofDim[Int](H, dataRSP.length)
    val dataRSPRDD: RDD[Array[Double]] = spark.sparkContext.makeRDD(dataRSP, numSlices = 1)
    val schema2 = StructType(Array(
      StructField("featuresArray", ArrayType(DoubleType), nullable = false)
    ))
    val datarowRDD = dataRSPRDD.map { case (center) =>
      Row(center.toSeq)
    }
    // 将RDD转换为DataFrame
    val trainfeature = spark.createDataFrame(datarowRDD, schema2)
    val arrayToVector = udf((array: Seq[Double]) => org.apache.spark.ml.linalg.Vectors.dense(array.toArray))
    val trainDataDF: DataFrame = trainfeature.withColumn("features", arrayToVector(col("featuresArray"))).drop("featuresArray")
    trainfeature.show()
    println(trainfeature.count())
    trainfeature.persist()
    var flag0 = 1
    while(flag0<H+1) {
      // 提取模型中保存的中心
      val kmeansCenters: Array[Array[Array[Double]]] = modelArray.map(center => center.centroids)
      val centerseq: Seq[Array[Double]] = kmeansCenters(flag0).toSeq
      // 指定保存路径
      val outputPath = "output/centers"+ flag0.toString +".parquet"
      val centerseqRDD: RDD[Array[Double]] = spark.sparkContext.makeRDD(centerseq,numSlices=1)
      val schema = StructType(Array(
        StructField("centers", ArrayType(DoubleType), nullable = false)
      ))
      val rowRDD = centerseqRDD.map { case (center) => Row(center.toSeq)}
      // 将RDD转换为DataFrame
      val center_row = spark.createDataFrame(rowRDD, schema)
      val center_row_label: DataFrame = center_row.withColumn("label", row_number().over(Window.orderBy(lit(1))) - 1)
        .withColumn("index", row_number().over(Window.orderBy(lit(1))) - 1)
      center_row_label.select("centers","index").write.parquet(outputPath)
      val center_train: DataFrame = center_row_label.withColumnRenamed("centers", "featuresArray").drop("index")
      // 定义 UDF 将 array<double> 转换为 org.apache.spark.ml.linalg.Vector
      val arrayToVector = udf((array: Seq[Double]) => org.apache.spark.ml.linalg.Vectors.dense(array.toArray))
      // 应用 UDF 转换 features 列
      val frame: DataFrame = center_train.withColumn("features", arrayToVector(col("featuresArray"))).select("features","label")
      val frame1: DataFrame = frame.withColumn("label", col("label").cast("double"))
      // 使用knn算法分配类标给用于集成的数据
      val knn = new KNNClassifier()
        .setTopTreeSize(1)
        .setK(1)
      val knnModel = knn.fit(frame1)
      val predicted = knnModel.transform(trainDataDF)
      array(flag0-1) = predicted.select("prediction").collect().map(data => data.getDouble(0).toInt)
      flag0 = flag0+1
    }
    //建立二值矩阵
    val binaryMatrixArray: Array[Array[Array[Int]]] = array.map(ele => {
      val matrix: Array[Array[Int]] = Array.ofDim[Int](ele.size, K)
      for (i <- 0 until ele.size) {
        matrix(i)(ele(i)) += 1
      }
      matrix
    })
    //二值矩阵按列拼接
    val resMatrix: Array[Array[Int]] = binaryMatrixArray.transpose.map(_.flatten)
    //归一化
    //第一步,将矩阵的每个元素除以 sqrt(H)
    val arr1: Array[Array[Double]] = resMatrix.map(r => {
      r.map(a => a / Math.sqrt(H))
    })
    //计算矩阵每一列的和
    // 先转置
    val transposedMatrix = resMatrix.transpose
    //累加为行向量
    val arr2: Array[Double] = transposedMatrix.map(column => column.reduce(_ + _)).map(_ + 0.000001)
    val rightArray = Array.ofDim[Double](arr2.size, arr2.size)
    for (i <- 0 until arr2.size) {
      rightArray(i)(i) = 1 / math.sqrt(arr2(i))
    }
    //两个矩阵做乘积
    val left = new Matrix(resMatrix.length, resMatrix(0).length, arr1)
    val right = new Matrix(resMatrix(0).length, resMatrix(0).length, rightArray)
    println(resMatrix.length)
    println(resMatrix(0).length)
    val matrixZ = left.mt(right.transpose())
    val matrixNcut: Matrix = matrixZ.ata()
    val svd = matrixNcut.svd()
    val SVDarray: Array[Array[Double]] = svd.V.toArray.map(row => row.slice(0, K_true))
    //val s: Array[Double] = svd.s.toArray
    //println(s.mkString(", "))
    val kmeansV = smileKMeans.lloyd(SVDarray, K_true)
    val y: Array[Int] = kmeansV.y
    var flag = 1
    while(flag<H+1) {
      val yselect: Array[Int] = y.slice((flag-1)*K, flag * K)
      val yselectRDD: RDD[Int] = spark.sparkContext.makeRDD(yselect,numSlices=1)
      val schema1 = StructType(Array(
        StructField("label", IntegerType, nullable = false)
      ))
      // 将RDD[(Array[Array[Double]], Int)]转换为RDD[Row]
      val rowRDD = yselectRDD.map { case (label) =>
        Row(label)
      }
      // 将RDD转换为DataFrame
      val df = spark.createDataFrame(rowRDD, schema1).withColumn("index",row_number().over(Window.orderBy(lit(1)))-1)
      val inputPath = "output/centers"+ flag.toString +".parquet"
      val centerdf: DataFrame = spark.read.parquet(inputPath)
      val dfWithLabels = centerdf.join(df, "index").drop("index")
      // 指定保存路径
      val outputPath = "output/centersLabel"+ flag.toString +".parquet"
      dfWithLabels.write.parquet(outputPath)
      dfWithLabels.show()
      dfWithLabels.unpersist()
      flag = flag+1
    }

    val feature: RDD[Array[Double]] = wrapperRDD.getSubPartitions(2).map(row=>row._2).flatMap(a=>a)//wrapperRDD.map(row => row._2).map(feature=>feature(0))
    val centerlabeldf: DataFrame = spark.read.parquet("output/centersLabel" + 1 + ".parquet")
    val schema1 = StructType(Array(
      StructField("featuresArray", ArrayType(DoubleType), nullable = false)
    ))
    // 将RDD[(Array[Array[Double]], Int)]转换为RDD[Row]
    val rowRDD = feature.map { case (center) =>
      Row(center.toSeq)
    }
    // 将RDD转换为DataFrame
    val testfeature = spark.createDataFrame(rowRDD, schema1)
    testfeature.persist()
    val testfeatureDF: DataFrame = testfeature.withColumn("features", arrayToVector(col("featuresArray"))).drop("featuresArray")
    val centerlabelDF: DataFrame = centerlabeldf.withColumn("features", arrayToVector(col("centers"))).select("features","label")
    .withColumn("label", col("label").cast("double"))
    //
    val knn = new KNNClassifier()
      .setTopTreeSize(1)
      .setK(1)
    val knnModel = knn.fit(centerlabelDF)
    val predicted = knnModel.transform(testfeatureDF)
    val labeldf: DataFrame = predicted.select("prediction","features").withColumnRenamed("prediction","label")
    //val labeldf: DataFrame = knn(centerlabeldf, testfeature,1)
    labeldf.show()
    labeldf.write.parquet("output/temp/clusterResult1")
    centerlabeldf.unpersist()
    labeldf.unpersist()
    var flag1 = 1
        while(flag1<H+1) {
          val centerlabeldf: DataFrame = spark.read.parquet("output/centersLabel" + flag1 + ".parquet")
          val centerlabelDF: DataFrame = centerlabeldf.withColumn("features", arrayToVector(col("centers"))).select("features","label")
          .withColumn("label", col("label").cast("double"))
          //
          val knn = new KNNClassifier()
            .setTopTreeSize(1)
            .setK(1)
          val knnModel = knn.fit(centerlabelDF)
          val predicted = knnModel.transform(testfeatureDF)
          val labeldf_temp: DataFrame = predicted.select("prediction", "features").withColumnRenamed("prediction","label")
          val Ldf_temp1: DataFrame = labeldf_temp.withColumnRenamed("label","label"+(flag1-1).toString)
          val Ldf_temp2: DataFrame = spark.read.parquet("output/temp/clusterResult" + flag1.toString)
          val labeldf: DataFrame = Ldf_temp1.join(Ldf_temp2, "features")
          labeldf.write.parquet("output/temp/clusterResult"+(flag1+1).toString)
          labeldf.show(50)
          println(labeldf.count())
          labeldf.unpersist()
          centerlabeldf.unpersist()
          flag1=flag1+1
        }

    val Ldf_temp2: DataFrame = spark.read.parquet("output/temp/clusterResult" + (flag1).toString).drop("features","label")
    // 定义UDF来计算每行的众数
    val majorityVote: UserDefinedFunction = udf((row: Row) => {
      val predictions = row.toSeq.map(_.asInstanceOf[Double])
      predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
    })

    // 使用withColumn将计算后的最终类别添加到DataFrame中
    val finalPredictions: DataFrame = Ldf_temp2.withColumn("final_prediction", majorityVote(struct(Ldf_temp2.columns.map(col): _*)))

    // 打印最终预测结果
    println("Final Predictions DataFrame:")
    finalPredictions.show(100)
    //    val r: Array[Row] = Ldf_temp2.select("label0").collect()
//    val ints: Array[Double] = r.map(a => a.getDouble(0))
//    println(ints.mkString(", "))
  }

}
