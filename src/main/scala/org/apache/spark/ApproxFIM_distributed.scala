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
import scala.collection.mutable.ArrayBuffer

object ApproxFIM_distributed {
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
    val sparkconf = new SparkConf().setAppName("ApproxFIM").setMaster("yarn")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val MST = args(0).toDouble
    val numberOfSampleBlocks = args(1).toInt
    val now_start: Date=new Date()
    import spark.implicits._
    val dataRead = spark.rspRead.parquet("data/TH2_260W_260_26W")
    val value1: RspRDD[Row] = dataRead.rdd.getRandomPartitions(numberOfSampleBlocks).toRSP(numberOfSampleBlocks*5)
    spark.createDataFrame(value1, spark.rspRead.parquet("data/TH2_260W_260_26W").schema).write.parquet("output/data")
    val numberOfData = value1.count().toInt
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
//    val df1: sql.DataFrame = goTable.toDF("items", "freq").withColumn("supportValue", col("freq")/numberOfData ).drop("freq")
    val df1: sql.DataFrame = goTable.toDF("items", "freq")
    df1.repartition(1).write.csv("ApproxFIM/GenD"+MST)
//    val Result1: collection.Map[String, Int] = goTable.map(row => row._1 -> row._2).coalesce(1).collectAsMap()

    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("===============================================" +
      "\n"+"Running time = "+ runningTime +"ms"
      +"\n"+"=================================================="+"\n")

//    val frame: DataFrame = spark.read.parquet("data/TH20.01")
//    val value: RDD[(String, Int)] = frame.rdd.map { row =>
//      // 使用 getAs 获取 items，并确保它是 Array[String] 类型
//      val items = row.getAs[mutable.WrappedArray[String]](0).mkString(",")
//      // 获取 freq 字段并转换为 Int
//      val freq = row.getLong(1).toInt
//      (items, freq)
//    }
//    frame.show(50)
//    print(frame.count())

//    val Result2: collection.Map[String, Int] = value.map(row => row._1 -> row._2).coalesce(1).collectAsMap()
//
//    GetInfo(Result1,Result2,numberOfData,transactions.getNumPartitions)

//    spark.close()
  }

  private def GetInfo(LOGO_Map: collection.Map[String, Int], Spark_Map: collection.Map[String, Int], TotalNum: Int, Block_Num: Int): Unit = {

    val SparkConvertedMap = Spark_Map.map {
      case (key, value) =>
        (key.replaceAll("[{}]", "").split(",").map(_.toInt).toSet, value)
    }
    val LOGOConvertedMap = LOGO_Map.map {
      case (key, value) =>
        (key.replaceAll("[{}]", "").split(",").map(_.toInt).toSet, value)
    }

    val Spark_FIM_List = SparkConvertedMap.keys.toList
    val LOGO_FIM_List = LOGOConvertedMap.keys.toList
    val TP_FIM = Spark_FIM_List.intersect(LOGO_FIM_List)

    // 设置统计信息
    val Real_Num: Int = Spark_Map.size
    val LOGO_Num: Int = LOGO_Map.size
    val TP: Int = TP_FIM.size

    println("Spark结果：" + Real_Num)
    println("LOGO结果：" + LOGO_Num)
    println("TP=" + TP)
    println("FP=" + (Real_Num - TP))
    println("FN=" + ((LOGO_Num - Real_Num + TP)-Real_Num))
    println("Precision：" + TP.toDouble / LOGO_Num)
    println("Recall：" + TP.toDouble / Real_Num)


    val table = ArrayBuffer[(Set[Int], Int)]()


    for (elem <- TP_FIM) {
      table.append((elem, math.abs(SparkConvertedMap.getOrElse(elem, 0) - LOGOConvertedMap.getOrElse(elem, 0) * Block_Num)))
    }

    val sum: Int = table.map(_._2).sum
    //println("Error误差：" + sum.toDouble / TP)
    println("Support误差：" + sum.toDouble / TP / TotalNum)
  }
}
