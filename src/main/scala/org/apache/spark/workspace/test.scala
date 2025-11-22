package org.apache.spark.workspace

import breeze.linalg.{DenseMatrix, InjectNumericOps}
import breeze.linalg.Matrix.castOps
import breeze.linalg.Vector.castOps
import breeze.numerics.sqrt
import breeze.storage.ConfigurableDefault.fromV
import org.apache.spark.RspContext.RspRDDFunc
import org.apache.spark.SparkConf
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, desc, monotonically_increasing_id, row_number, udf}
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType, StructField, StructType}

//import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{NewRDDFunc, SparkSessionFunc}
import smile.clustering.KMeans
import smile.math.distance.EuclideanDistance
//import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.clustering.{KMeans => smileKMeans}
import smile.math.matrix.Matrix


object test {
  def main(args: Array[String]): Unit = {
    //变量
    val K_true = 10
    val K = 10*3
    val H = 3
    //初始化环境
    val sparkConf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
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

    //    val wrapperRDD = new RspRDD(rspRDD.map(r => {
    //      val doubles = r.toString().replace('[', ' ').replace(']', ' ').trim.split(" ").map(_.toDouble)
    //      (doubles(doubles.length - 1).toInt, doubles.dropRight(1))
    //    }))
    //    val wrapperRDD: RspRDD[(Array[Int], Array[Array[Double]])] = BasicWrappers.toMatrixRDD(rspRDD)
    //取出部分数据块, 前面四个用于训练，后面用于测试
    val subRddArray: Array[RspRDD[(Array[Int], Array[Array[Double]])]] = wrapperRDD.getTwoRandomPartitions(H, 1)
    //val trainRDD: RspRDD[(Array[Int], Array[Array[Double]])] = wrapperRDD.getSubPartitions(1)
    val trainRDD: RspRDD[(Array[Int], Array[Array[Double]])] = subRddArray(0)
    //val testRDD: RspRDD[(Array[Int], Array[Array[Double]])] = wrapperRDD.getSubPartitions(2)
    val testRDD = subRddArray(1)
    //进行LO操作,调用smile库里面的kmeans
    val centerRDD: RDD[smileKMeans] = trainRDD.mapPartitions(f => {
      val partitionRes = smileKMeans.fit(f.next()._2, K)
      if (f.hasNext) throw new Exception("The input iterator of LO should only has one element.")
      Iterator.single(partitionRes)
    })
    val modelArray: Array[smileKMeans] = centerRDD.collect()
    //每一个模型均对测试数据块做预测
    val test_feature = testRDD.collect()(0)._2
    val array = modelArray.map(ele => {
      val predictArray = test_feature.map(f => {
        ele.predict(f)
      })
      predictArray
    })
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
    val doubleMatrix = transposedMatrix.map(_.map(_.toDouble))
    //累加为行向量
    val arr2 = doubleMatrix.map(column => column.reduce(_ + _)).map(_ + 0.000001)
    //建立对角矩阵
    val rightArray = Array.ofDim[Double](arr2.size, arr2.size)
    for(i <- 0 until  arr2.size) {
      rightArray(i)(i) = 1/math.sqrt(arr2(i))
    }

    //两个矩阵做乘积
    val left = new Matrix(resMatrix.length, resMatrix(0).length, arr1.map(_.map(_.toDouble)))
    val right = new Matrix(resMatrix(0).length, resMatrix(0).length, rightArray.map(_.map(_.toDouble)))
    val matrix = left.mm(right)
    val matrixNcut: Matrix = matrix.transpose().mm(matrix)
    val svd = matrixNcut.svd()
    //println(svd.U)
    val SVDarray: Array[Array[Double]] = svd.U.toArray.map(row => row.slice(1, K_true))
    val kmeans = smileKMeans.fit(SVDarray, K_true)
    val y: Array[Int] = kmeans.y

    var flag = 1
    while(flag<H+1) {
      val yselect: Array[Int] = y.slice((flag-1)*K, flag * K)
      val centercelected: Array[Array[Double]] = modelArray.map(model => model.centroids).map(center=>center(flag))
      val rddcenter: Seq[(Array[Double], Int)] = centercelected.toSeq.zip(yselect)
      val centerrdd: RDD[(Array[Double], Int)] = spark.sparkContext.makeRDD(rddcenter)
      val schema = StructType(Array(
        StructField("centers", ArrayType(DoubleType), nullable = false),
        StructField("label", IntegerType, nullable = false)
      ))
      // 将RDD[(Array[Array[Double]], Int)]转换为RDD[Row]
      val rowRDD = centerrdd.map { case (center, label) =>
        Row(center.toSeq, label)
      }
      // 将RDD转换为DataFrame
      val df = spark.createDataFrame(rowRDD, schema)
      // 指定保存路径
      val outputPath = "output/centersLabel"+ flag.toString +".parquet"
      df.write.parquet(outputPath)
      df.show()
      centerrdd.unpersist()
      flag = flag+1
    }
    var flag1 = 2
    val feature: RDD[Array[Double]] = wrapperRDD.map(row => row._2).map(feature=>feature(0))
    val centerlabeldf: DataFrame = spark.read.parquet("output/centersLabel" + flag1 + ".parquet")
    val schema1 = StructType(Array(
      StructField("features", ArrayType(DoubleType), nullable = false)
    ))
    // 将RDD[(Array[Array[Double]], Int)]转换为RDD[Row]
    val rowRDD = feature.map { case (center) =>
      Row(center.toSeq)
    }
    // 将RDD转换为DataFrame
    val testfeature = spark.createDataFrame(rowRDD, schema1)
    val labeldf: DataFrame = knn(centerlabeldf, testfeature,1)
    val Ldf: DataFrame = labeldf.withColumn("id", monotonically_increasing_id())
//    while(flag1<H+1) {
//      val centerlabeldf: DataFrame = spark.read.parquet("output/centersLabel" + flag1 + ".parquet")
//      val schema1 = StructType(Array(
//        StructField("features", ArrayType(DoubleType), nullable = false)
//      ))
//      // 将RDD[(Array[Array[Double]], Int)]转换为RDD[Row]
//      val rowRDD = feature.map { case (center) =>
//        Row(center.toSeq)
//      }
//      // 将RDD转换为DataFrame
//      val testfeature = spark.createDataFrame(rowRDD, schema1)
//      val labeldf_temp: DataFrame = knn(centerlabeldf, testfeature,1)
//      val Ldf_temp: DataFrame = labeldf_temp.withColumn("id", monotonically_increasing_id())
//      val Ldf: DataFrame = Ldf.join(Ldf_temp, "id")
//      flag1=flag1+1
//    }
    Ldf.show()
  }

  val euclideanDistance = udf((p1: Seq[Double], p2: Seq[Double]) => {
    scala.math.sqrt(p1.zip(p2).map { case (x, y) => (x - y) * (x - y) }.sum)
  })

  def knn(data: DataFrame, points: DataFrame, k: Int): DataFrame = {
    import data.sparkSession.implicits._

    // 计算每个点到数据集中所有点的距离，并添加到一个新的DataFrame中
    val distanceDF = points.crossJoin(data)
      .withColumn("distance", euclideanDistance(col("centers"), col("features")))
      .select(col("centers").alias("input_point"), col("label"), col("distance"))

    // 找到每个点的最近k个邻居
    val windowSpec = Window.partitionBy("input_point").orderBy("distance")
    val nearestNeighbors = distanceDF.withColumn("rank", row_number().over(windowSpec)).where($"rank" <= k)

    // 对每个点的k个邻居进行投票
    val majorityLabels = nearestNeighbors.groupBy("input_point", "label").count()
      .withColumn("rank", row_number().over(Window.partitionBy("input_point").orderBy(desc("count"))))
      .where($"rank" === 1)
      .select("label")

    majorityLabels
  }
}
