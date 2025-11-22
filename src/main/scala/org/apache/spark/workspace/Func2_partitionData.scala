package org.apache.spark.workspace

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object Func2_partitionData {
  def main(args: Array[String]): Unit = {

    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()

    // 循环生成RSP数据
    for (i <- 1 to 21 by 2) {
      rePartitionData("datas/T10I4D100K_RSP_10blocks",i,spark)
    }

    spark.close()

  }

  def rePartitionData(path: String, num: Int, spark: SparkSession): Unit = {

    // 读取Parquet数据文件
    val originalData = spark.read.parquet(path)

    // 重新分区为50个，并保存为Parquet格式
    val repartitionedData = originalData.repartition(num)

    val fileName = path.split("_")(0) + "_RSP_" + num + "blocks"

    println("写入文件：" + fileName)

    repartitionedData.write.parquet(fileName)

  }


}
