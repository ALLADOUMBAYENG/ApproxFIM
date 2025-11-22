package org.apache.spark

import breeze.linalg.split
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession, functions}

import java.util.Date

object realWorldDataGen {
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
    val MST = 0.005
    val now_start: Date=new Date()
    var Transaction_temp: DataFrame = spark.emptyDataFrame
    import spark.implicits._
    // Kaggle
    val dataRead: DataFrame = spark.read.parquet("data/Kaggle.parquet").repartition(60)
    val dataRead1: DataFrame = dataRead.select("product_id").withColumnRenamed("product_id", "items")
////      dataRead1.show()
//
//    var i = 1
//    while(i<11){
//      if (Transaction_temp.isEmpty) {
//        Transaction_temp = dataRead1
//      } else {
//        Transaction_temp = Transaction_temp.union(dataRead1)
//      }
//      i = i+1
//    }
//    Transaction_temp.write.parquet("data/Kaggle")
//    print(Transaction_temp.rdd.getNumPartitions)
//    //T10I4D100K
//    val dataRead: DataFrame = spark.read.text("data/T40I10D100K.txt")
//    // 转换为 RDD 并处理每一行
//    val rdd = dataRead.rdd.map(row => row.getString(0).split(" ")).repartition(5)
//    // 转回 DataFrame
//    val dataRead1 = rdd.map(arr => arr.map(_.toInt)).toDF("items")
////    var i = 1
////    while(i<101){
////        if (Transaction_temp.isEmpty) {
////            Transaction_temp = dataRead1
////        } else {
////            Transaction_temp = Transaction_temp.union(dataRead1)
////        }
////      i = i+1
////    }
////      Transaction_temp.write.parquet("data/T40Origin")
//    dataRead1.write.parquet("data/T40Origin")
//    //T40I10D100K

//    val dataNew: Dataset[Row] = dataRead.union(dataRead)
//    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
//    val model = fpgrowth.fit(dataRead1)
//    val FIM: DataFrame = model.freqItemsets.filter(functions.size($"items") > 1)
//      FIM.show()
//      print("\n"+FIM.count()+"\n")
//
////    FIM.write.parquet("SparkFIMResult1/TH1"+MST)
//    val now_end: Date=new Date()
//    val runningTime = now_end.getTime - now_start.getTime
//    print("===============================================" +
//      "\n"+"Running time = "+ runningTime +"ms"
//      +"\n"+"==================================================")
  }
}
