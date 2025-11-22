package org.apache.spark

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.classification.{DecisionTree, cart}
import smile.data.DataFrame
import smile.data.formula._
import smile.validation.Accuracy


object Test_Smile {
  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()

    println("环境测试成功.........")

    //    var frame: org.apache.spark.sql.DataFrame = spark.read.parquet("datas/classification_50_2_0.54_5_64M.parquet")

    val frame: RspDataset[Row] = spark.rspRead.parquet("data/classification_100_2_0.54_4_32M.parquet")

    frame.show(1)
    println("数据读取成功.........")

    val Array(
    trainRDD: RDD[(Array[Int], Array[Array[Double]])],
    testRDD: RDD[(Array[Int], Array[Array[Double]])]
    ) = frame.rdd.randomSplit(Array(0.9, 0.1)).map(rdd => etl(rdd))


    val value: RDD[Array[Array[Double]]] = frame.rdd.glom().map(f => f.map(r => r.get(1).asInstanceOf[DenseVector].toArray))


    val value2: RDD[(Array[Int], Array[Array[Double]])] = frame.rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )



    value2.map(
      f => {
        val labelFrame = DataFrame.of(f._1.map(l => Array(l)),"Y")
        val featureFrame = DataFrame.of(f._2)
        val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
        val trainFrame = featureFrame.merge(labelFrame)
        trainFrame
      }
    )




    def trainer2(sample: (Array[Int], Array[Array[Double]])): DecisionTree = {
      val trainLabel: Array[Array[Int]] = sample._1.map(l => Array(l))
      val trainFeatures: Array[Array[Double]] = sample._2
      val featureFrame: DataFrame = DataFrame.of(trainFeatures) //特征列
      val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y") //标签列
      val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
      val trainFrame = featureFrame.merge(labelFrame)

      val tree: DecisionTree = cart(formula, trainFrame, maxDepth = 3, nodeSize = 10)
      tree
    }

//        val value: RDD[(Array[Int], Array[Array[Double]])] = frame.rdd.glom().map(
//          f => (
//            f.map(r => r.getInt(0)),
//            f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
//          )
//        )
    //
    //    val resultRDD2 = value.map(trainer)
    //    val modelRDD2 = resultRDD2.map(_._1)
    //    val timeRDD2 = resultRDD2.map(_._2)
    //
    //    println("模型数量：" + modelRDD2.partitions.length)
    //    timeRDD2.foreach(time => println("建模时间:" + time))
    //
    //
    //    println("==================")

    val resultRDD = trainRDD.map(trainer)
    val modelRDD = resultRDD.map(_._1)
    val timeRDD = resultRDD.map(_._2)

    val predictRDD = modelRDD.cartesian(testRDD).map(
      f => estimator(predictor(f._1, f._2._2), f._2._1)
    )

    val accuracies = predictRDD.collect()
    printf("%s\n", accuracies.map(_.toString).reduce(_ + ", " + _))


    //    modelRDD.foreach(println)

    println("模型数量：" + modelRDD.partitions.length)
    timeRDD.foreach(time => println("建模时间:" + time))

    //TODO 尝试创建Smile算法对象

    spark.close()
  }

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = {

    rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  def trainer(sample: (Array[Int], Array[Array[Double]])): (DecisionTree, Double) = {
    val trainLabel: Array[Array[Int]] = sample._1.map(l => Array(l))
    val trainFeatures: Array[Array[Double]] = sample._2
    val featureFrame: DataFrame = DataFrame.of(trainFeatures) //特征列
    val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y") //标签列
    val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
    val startTime = System.nanoTime
    val trainFrame = featureFrame.merge(labelFrame)
    val tree: DecisionTree = cart(formula, trainFrame, maxDepth = 3, nodeSize = 10)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (tree, duration)
  }

  def etl2(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = {

    rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }


  def trainer2(sample: (Array[Int], Array[Array[Double]])): DecisionTree = {
    val trainLabel: Array[Array[Int]] = sample._1.map(l => Array(l))
    val trainFeatures: Array[Array[Double]] = sample._2
    val featureFrame: DataFrame = DataFrame.of(trainFeatures) //特征列
    val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y") //标签列
    val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
    val trainFrame = featureFrame.merge(labelFrame)

    val tree: DecisionTree = cart(formula, trainFrame, maxDepth = 3, nodeSize = 10)
    tree
  }







  def predictor(model: DecisionTree, features: Array[Array[Double]]): Array[Int] = {
    val testFeatures: Array[Array[Double]] = features
    val testFrame: DataFrame = DataFrame.of(testFeatures)
    return model.predict(testFrame)
  }

  def estimator(prediction: Array[Int], label: Array[Int]): Double = {
    Accuracy.of(label, prediction)
  }


}



