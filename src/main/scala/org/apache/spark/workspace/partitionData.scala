package org.apache.spark.workspace

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object partitionData {
  def main(args: Array[String]): Unit = {

    // 创建SparkSession
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()

    // 读取Parquet数据文件
    val originalData = spark.read.parquet("datas/T40I10D100K_RSP_10blocks_1W.parquet")

    // 定义不同分区数的列表
//    val partitionCounts = List(10, 20, 30, 40, 50)

    // 重新分区为50个，并保存为Parquet格式
    val repartitionedData = originalData.repartition(21)

    repartitionedData.write.parquet("datas/T40I10D100K_RSP_21blocks")

    // 逐个重新分区并保存为Parquet格式
//    partitionCounts.foreach { numPartitions =>
//      val repartitionedData = originalData.repartition(numPartitions)
//      repartitionedData.write.parquet(s"datas/kosarak_RSP_$numPartitions")
//    }

    spark.close()

  }
}
