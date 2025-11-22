package org.apache.spark

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc}
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{Row, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.Date
import java.util.stream.Stream

object GenD_subDataset {
  def main(args: Array[String]): Unit = {

    // 获取 Hadoop 文件系统对象
    val hadoopConf = new Configuration()
    val hadoopFs = FileSystem.get(hadoopConf)
    // 指定要检查的目录路径
    val outputPath = new Path("output/data")
    // 检查目录是否存在
    if (hadoopFs.exists(outputPath)) {
      println(s"Directory $outputPath exists. Deleting...")
      // 删除目录
      hadoopFs.delete(outputPath, true)
      println(s"Directory $outputPath has been deleted.")
    } else {
      println(s"Directory $outputPath does not exist.")
    }
    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("yarn")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val MST = args(0).toDouble
    val now_start: Date=new Date()
    import spark.implicits._
    val dataRead = spark.rspRead.parquet("/user/caimeng/Items_10E_10000_100000.parquet").getSubDataset(5000)
    val value1: RspRDD[Row] = dataRead.rdd.toRSP(500)
    spark.createDataFrame(value1, spark.rspRead.parquet("/user/caimeng/Items_10E_10000_100000.parquet").schema).write.parquet("output/data")
    val numberOfData: Double = value1.getSubPartitions(1).count().toDouble
    val transactions: RspRDD[Array[Array[Int]]]  = spark.rspRead.parquet("output/data")
      .dataWrapper(Smile_Parquet_FPG)

    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * MST).toInt, trainDF)
    )

    println("第一种方式的RSP模型的数量：" + localTable.count())
    println("第二种方式的RSP模型的数量：" + localTable.partitions.length)

    println("-------------LO建模成功---------------")

    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)
    val df1: sql.DataFrame = goTable.toDF("items", "freq")
//    df1.show()
    df1.write.csv("ScaDistFIM/GenD"+MST)

//    val df1: sql.DataFrame = goTable.toDF("items", "freq").withColumn("supportValue", col("freq") / numberOfData).drop("freq")
////    df1.show()
//    df1.repartition(1).write.csv("ScaDistFIM/TH2"+MST)
    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("===============================================" +
      "\n"+"Running time = "+ runningTime +"ms"
      +"\n"+"==================================================")
  }
}
