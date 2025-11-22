package org.apache.spark

import org.apache.spark.Test_Smile.{estimator, predictor}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, SparkSessionFunc}
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.classification.{DecisionTree, cart}
import smile.data.DataFrame
import smile.data.formula._
import smile.validation.Accuracy

object testofficialsmile {
  def main(args: Array[String]): Unit = {

    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()

    val inputData: RspDataset[Row] = spark.rspRead.parquet("datas/classification_50_2_0.54_5_64M.parquet")

    val testInputData: RspDataset[Row] = inputData.getSubDataset(1)

    val trainRDD: RDD[DataFrame] = inputData.rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    ).map(
      f => {
        val labelFrame = DataFrame.of(f._1.map(l => Array(l)), "Y")
        val featureFrame = DataFrame.of(f._2)
        val trainFrame = featureFrame.merge(labelFrame)
        trainFrame
      }
    )

    val modelRDD: RDD[DecisionTree] = trainRDD.map(trainer)

    val testRDD: RDD[(Array[Int], Array[Array[Double]])] = testInputData.rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )

    val predictRDD: RDD[(DecisionTree, Double)] = modelRDD.cartesian(testRDD).map(
      f => {
        (f._1, estimator(predictor(f._1, f._2._2), f._2._1))
      }
    )

    predictRDD.foreach(x => println("准确率为" + x._2))

    val sortedPredictRDD = predictRDD.sortBy(pair => pair._2)

    val totalCount = sortedPredictRDD.count()
    println("模型总数：" + totalCount)

    val num = sortedPredictRDD.partitions.length
    println("分区数量：" + num)

    val filterModel: RDD[(DecisionTree, Double)] = sortedPredictRDD.zipWithIndex().filter {
      case (_, index) =>
        index >= totalCount * 0.1 && index <= (1 - 0.1) * totalCount - 1
    }.map {
      case (pair, _) => pair
    }

    println("过滤后保留的分区数量：" + filterModel.partitions.length)
    println("过滤后保留的模型数量：" + filterModel.count())


  }

  def trainer(trainFrame: DataFrame): DecisionTree = {
    val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
    val tree: DecisionTree = cart(formula, trainFrame, maxDepth = 3, nodeSize = 10)
    tree
  }
}
