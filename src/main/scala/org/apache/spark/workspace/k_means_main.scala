package org.apache.spark.workspace

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.RspContext.RspRDDFunc
import org.apache.spark.RspContext.{RspRDDFunc, SparkSessionFunc}
import org.apache.spark.{RspContext, SparkConf}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame}

import java.util.Date
import scala.math.sqrt
import scala.util.Random

//import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.NewRDDFunc
//import org.apache.spark.sql.RspContext._
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.sql.{Row, SparkSession}
import smile.clustering.{KMeans => smileKMeans}
import smile.math.matrix._

object k_means_main {
  def main(args: Array[String]): Unit = {
    //变量
    val K_true = 13
    //初始化环境
    val sparkConf = new SparkConf().setAppName("Clustering").setMaster("yarn")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    ////// 读取txt数据 //////
    val start_time = new Date()
    //val dataRDD: RspRDD[Row] = spark.rspRead.text(args(0)).rdd
    val dataRDD: RDD[Row] = spark.read.text(args(0)).rdd
    //Rsp化
    //val rspRDD: RspRDD[Row] = dataRDD.toRSP(30)
    //println(rspRDD.getNumPartitions)
    //格式化Row为(Array[Int], Array[Array[Double]])
    val wrapperRDD = new RspRDD(dataRDD.glom().map(f => (
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
    import spark.implicits._
    val data: DataFrame = value1.flatMap { case (labels, features) =>
      labels.zip(features).map { case (label, feature) =>
        (label, Vectors.dense(feature))
      }
    }.toDF("label", "features")
    data.show()

    ////读取parquet数据//////
    //  val data = spark.read.parquet(args(0))
//   //////读取svm数据//////
//    val data = spark.read.format("libsvm").load(args(0))
    //////读取csv数据//////
    //val data_row: DataFrame = spark.read.csv(args(0))

//    val allColumns = data_row.columns
//    // 第一列是label列
//    val labelColumn = allColumns.head
//    // 剩余的列是feature列
//    val featureColumns = allColumns.tail
//    // 将特征列转换为数值类型
//    val dfWithNumericFeatures = featureColumns.foldLeft(data_row) { (tempDF, colName) =>
//      tempDF.withColumn(colName, col(colName).cast("double"))
//    }
//    // 创建VectorAssembler
//    val assembler = new VectorAssembler()
//      .setInputCols(featureColumns)
//      .setOutputCol("features")
//    // 使用VectorAssembler进行转换
//    val data = assembler.transform(dfWithNumericFeatures)
//      .withColumnRenamed(labelColumn, "label")
//      .select("label", "features")

    //读取数据
    //val data = spark.read.parquet(args(0))
    //////kmeans算法
//    val kmeans = new KMeans().setK(K_true).setSeed(9L)
//    val model = kmeans.fit(data)
//    val predictions = model.transform(data)
    //////bkm算法
//    val bkm = new BisectingKMeans().setK(K_true).setSeed(9L)
//    val model = bkm.fit(data)
//    // Make predictions
//    val predictions = model.transform(data)
    /////GMM算法
    val gmm = new GaussianMixture().setK(K_true).setSeed(9L)
    val model = gmm.fit(data)
    val predictions: DataFrame = model.transform(data)
//    predictions.show()

    val dfInt: DataFrame = predictions.select("label", "prediction")
    dfInt.show()
    dfInt.coalesce(1).write.csv("result/Flower20/GMM/")

//    val purity = calculatePurity(dfInt,spark)
//    val nmi = calculateNMI(dfInt,spark)
//    println("purity结果为："+purity)
//    println("nmi结果为："+nmi)
    val end_time = new Date()
    val runningtime = end_time.getTime - start_time.getTime
    println("running time:" + runningtime)
  }


  def calculatePurity(df: DataFrame,spark:SparkSession): Double = {
    import spark.implicits._
    // 计算每个簇中每个标签的计数
    val counts = df.groupBy("prediction", "label").count()

    // 找到每个簇中占多数的标签
    val maxCounts = counts.withColumn("max_count", max("count").over(Window.partitionBy("prediction")))
      .filter($"count" === $"max_count")
      .groupBy("prediction")
      .agg(first("count").as("count")) // 取第一个最大值

    // 计算纯度
    val total = df.count()
    val purity = maxCounts.agg(sum("count")).as[Long].first().toDouble / total

    purity
  }

  def safeReduce(arr: Array[Double]): Double = {
    if (arr.isEmpty) 0.0 else arr.reduce(_ + _)
  }

  def calculateNMI(df: DataFrame,spark:SparkSession): Double = {
    import spark.implicits._
    // 计算每个标签和每个预测的计数
    val labelCounts = df.groupBy("label").count().withColumnRenamed("count", "label_count")
    val predictionCounts = df.groupBy("prediction").count().withColumnRenamed("count", "prediction_count")

    // 计算联合计数
    val jointCounts = df.groupBy("label", "prediction").count().withColumnRenamed("count", "joint_count")

    val total = df.count()

    // 计算互信息
    val mutualInfoArray = jointCounts
      .join(labelCounts, "label")
      .join(predictionCounts, "prediction")
      .select($"joint_count", $"label_count", $"prediction_count")
      .as[(Long, Long, Long)]
      .map { case (jointCount, labelCount, predictionCount) =>
        val pij = jointCount.toDouble / total
        val pi = labelCount.toDouble / total
        val pj = predictionCount.toDouble / total
        pij * math.log(pij / (pi * pj))
      }
      .collect()

    val mutualInfo = safeReduce(mutualInfoArray)

    // 计算标签熵和预测熵
    val entropyLabelArray = labelCounts.select($"label_count")
      .as[Long]
      .map { count =>
        val p = count.toDouble / total
        -p * math.log(p)
      }
      .collect()

    val entropyLabel = safeReduce(entropyLabelArray)

    val entropyPredictionArray = predictionCounts.select($"prediction_count")
      .as[Long]
      .map { count =>
        val p = count.toDouble / total
        -p * math.log(p)
      }
      .collect()

    val entropyPrediction = safeReduce(entropyPredictionArray)

    // 计算 NMI
    val nmi = if (entropyLabel == 0.0 || entropyPrediction == 0.0) 0.0 else mutualInfo / sqrt(entropyLabel * entropyPrediction)

    nmi
  }
  //不放回抽样所生成的数组
  def sampling_Without_Replacement(total: Int, subNum: Int) = {
    //这个生成不重复随机数的意思是arr是一个保留有全部数字的数字，每次都从arr里取一个数，取出了之后就删掉，保证不取重复
    var arr = 0 to total toArray
    var outList: List[Int] = Nil
    var border = arr.length //随机数范围
    for (i <- 0 to subNum - 1) { //生成n个数
      val index = (new Random).nextInt(border)
      //println(index)
      outList = outList ::: List(arr(index))
      arr(index) = arr.last //将最后一个元素换到刚取走的位置
      arr = arr.dropRight(1) //去除最后一个元素
      border -= 1
    }
    outList
  }
}
