package org.apache.spark

import org.apache.spark.RspContext.{RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rsp.RspRDD
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
//import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}
import smile.classification.{DecisionTree, RandomForest, cart, randomForest}
import smile.data.DataFrame
import smile.data.formula.Formula

import java.util.stream.{Collectors, Stream}
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rdd.RDD

object Logo_Voting {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // TODO 数据加载并进行数据封装
    val inputDataFrame: RspRDD[DataFrame] = spark.rspRead.parquet("datas/classification_50_2_0.54_5_64M.parquet")
      .getSubDataset(3)
      .dataWrapper(Smile_Parquet_Classification)
    println("------数据加载成功，并完成数据封装-------")

    // TODO LOGO阶段自定义训练器的执行方式
    def DT(trainFrame2: DataFrame): DecisionTree = {
      val tree: DecisionTree = cart(Formula.lhs("Y"), trainFrame2, maxDepth = 2, nodeSize = 2)
      tree
    }

    val modelRDD: RspRDD[DecisionTree] = inputDataFrame.LO(DT)

    println("-------------LO建模成功---------------")

    println(modelRDD.foreach(println))
    println("第一种方式的RSP模型的数量：" + modelRDD.partitions.length)
    println("第二种方式的RSP模型的数量：" + modelRDD.count())

    val inputData: RspDataset[Row] = spark.rspRead.parquet("datas/classification_50_2_0.54_5_64M.parquet")
      .getSubDataset(1)

    val value: RDD[(Array[Int], Array[Array[Double]])] = inputData.rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )

    value.first()._2




  }
}
