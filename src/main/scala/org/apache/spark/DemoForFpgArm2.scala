package org.apache.spark

import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, SparkSession}
import smile.association.{ItemSet, fpgrowth}
import smile.classification.{DecisionTree, RandomForest, cart, randomForest}
import smile.data.DataFrame
import smile.data.formula.Formula

import java.util.stream.Stream

object DemoForFpgArm2 {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    import spark.implicits._

//    val value: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("data/TH1_260W_10_26W").dataWrapper(Smile_Parquet_FPG)
//    val transactions: RspRDD[Array[Array[Int]]] = value.getSubPartitions(10).toRSP(10)
    val value1: RspRDD[Row] = spark.rspRead.parquet("data/TH1_260W_10_26W").rdd.toRSP(50).getSubPartitions(15)
    spark.createDataFrame(value1, spark.rspRead.parquet("data/TH1_260W_10_26W").schema).write.parquet("data/TH1")

    val transactions: RspRDD[Array[Array[Int]]]  = spark.rspRead.parquet("data/TH1")
      .dataWrapper(Smile_Parquet_FPG)

    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * 0.01).toInt, trainDF)
    )

    println("第一种方式的RSP模型的数量：" + localTable.count())
    println("第二种方式的RSP模型的数量：" + localTable.partitions.length)

    println("-------------LO建模成功---------------")

    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)

    println("-------------GO集成成功---------------")

//    println(goTable.toDebugString)
    goTable.foreach(println)
    spark.close()

  }
}
