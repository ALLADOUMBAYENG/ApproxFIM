package org.apache.spark

import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream

object DemoForTestDataset {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // 开始时间
    val startTime = System.nanoTime()

    val transactions: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("datas/T40I10D100K_RSP_1blocks")
      .dataWrapper(Smile_Parquet_FPG)

    println(transactions.count())

    val value: RspDataset[Row] = spark.rspRead.parquet("datas/T40I10D100K_RSP_1blocks")
    println(value.rdd.count())


    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * 0.015).toInt, trainDF)
    )

    println("-------------LO建模成功---------------")

    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)

    println("-------------GO集成成功---------------")

    goTable.foreach(println)
    goTable.count()

    // 结束时间
    val endTime = System.nanoTime()
    // 计算执行时间（以毫秒为单位）
    val executionTimeMs = (endTime - startTime) / 1000000

//    goTable.saveAsTextFile("datas/T40I10D100K_result000")

    println("---------文件已保存-----------")

    // 打印执行时间
    println(s"程序执行时间：$executionTimeMs 毫秒")

    println(goTable.count())

    spark.close()

  }
}
