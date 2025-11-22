package org.apache.spark

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc}
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.functions.{col, concat_ws}
import org.apache.spark.sql.{DataFrame, Dataset, Row, RspDataset, SparkSession, functions}
import smile.association.{ItemSet, fpgrowth}

import java.util.Date
import java.util.stream.Stream

object SparkFPFIM {
  def main(args: Array[String]): Unit = {

//    // 获取 Hadoop 文件系统对象
//    val hadoopConf = new Configuration()
//    val hadoopFs = FileSystem.get(hadoopConf)
//    // 指定要检查的目录路径
//    val outputPath = new Path("output/data")
//    // 检查目录是否存在
//    if (hadoopFs.exists(outputPath)) {
//      println(s"Directory $outputPath exists. Deleting...")
//      // 删除目录
//      hadoopFs.delete(outputPath, true)
//      println(s"Directory $outputPath has been deleted.")
//    } else {
//      println(s"Directory $outputPath does not exist.")
//    }
    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("SparkFPGrowth").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val MST = 0.009
    val now_start: Date=new Date()
    import spark.implicits._
//    val dataRead = spark.read.parquet("/user/caiyongda/realWorldDataset/TH2").repartition(120)
//    val dataRead: DataFrame = spark.read.parquet("data/Kaggle.parquet")
//    val dataRead1: DataFrame = dataRead.select("product_id").withColumnRenamed("product_id", "items")
//    val dataNew: Dataset[Row] = dataRead.union(dataRead)

    val dataRead: DataFrame = spark.read.parquet("data/T40Origin")
    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
    val model = fpgrowth.fit(dataRead)
    val FIM: DataFrame = model.freqItemsets.filter(functions.size($"items") > 1)
      FIM.show()
      print("\n"+FIM.count()+"\n")

    FIM.write.parquet("SparkFIMResult1/T40Origin"+MST)
    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("===============================================" +
      "\n"+"Running time = "+ runningTime +"ms"
      +"\n"+"==================================================")
  }
}
